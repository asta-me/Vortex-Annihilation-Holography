#%%
"""
Signal-region-matched smooth-random INITIALIZATION for the irrotational projector.

Motivation / criterion
----------------------
A random start seeds a huge number of vortices and lands in a poor GS basin (RMSE ~9 on marmo).
A QUADRATIC (lens) init lowers RMSE a lot (~4.9) but is TARGET-DEPENDENT (a lens tuned to a
square/uniform target) and not diverse. A brute-force sweep of a smooth-random init found a
smooth diffuser can even BEAT the quadratic (RMSE ~4.2 on marmo), but the optimum looked sharply
peaked and target-dependent.

This script replaces the brute-force sweep with a PRINCIPLED CRITERION:

    choose the smooth phase so that ITS FAR-FIELD LANDS INSIDE THE SIGNAL REGION.

Physically the initial phase is a random DIFFUSER; its far-field spread must match the SR -- not
smaller (light stays concentrated -> speckle) nor larger (light spills outside the SR -> lost).
The right spread is a GEOMETRIC quantity (grid N vs SR M), hence TARGET-INDEPENDENT.

Construction (no brute force):
  1. smooth real field sm = low-pass(white noise) with a fixed moderate correlation length,
  2. AUTO-SCALE its amplitude 'a' (bisection) so the far-field RMS radius of exp(i*a*sm) equals a
     target radius ~ (M/2)*FILL, i.e. the diffuser just fills the SR. We report the resulting
     in-SR energy fraction as the criterion check.
Because it is random, several seeds give diverse (multiplexable) vortex-free-friendly starts.

Results (marmo.tif, floor 5e-3, alpha 0.5, 300 iters)
-----------------------------------------------------
    random init         RMSE ~8.97
    quadratic (c=1500)  RMSE ~4.88   (target-dependent lens, not diverse)
    matched diffuser    RMSE ~4.8-5.6 over 3 seeds (best seed 4.81 < quadratic), auto amp ~20,
                        in-SR energy fraction ~0.92 (criterion satisfied), init vortices ~50-170
                        (low), final ~0. Target-INDEPENDENT (fill is geometric) and seed-DIVERSE
                        (multiplexable). No brute-force tuning: amp is auto-scaled by the criterion.
See __main__ printout for the exact numbers on the configured target.

Env: vortex (conda), GPU (CuPy). PIL + cv2 (no skimage).
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import cupy as cp
import cv2
from PIL import Image

ALT_PROJ_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "1_Alternative_projection")
sys.path.insert(0, ALT_PROJ_DIR)
from function_vortex_detection_accegpu import function_vortex_detection_accegpu

#%% Config
dh = 0.00374; loop = 300; seed = 42
input_tiff = "marmo.tif"; target_floor_rel = 5e-3; alpha = 0.5
quad_c = 1500.0
corr_ks = 8.0                 # low-pass width for the smooth field (moderate correlation length)
fill = 0.65                   # target far-field RMS radius = (M/2)*fill  (fill the SR)
diffuser_seeds = [1, 2, 3]
n = m = 512; nn = n + 2*(n//4); mm = m + 2*(m//4)
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "output_smooth_init"); os.makedirs(output_dir, exist_ok=True)

#%% Grid + target
bandlim_spe = cp.zeros((nn, mm), cp.float32); bandlim_spe[nn//4:3*nn//4, mm//4:3*mm//4] = 1.0
bandlim_in = cp.zeros((nn, mm), cp.float32); bandlim_in[(nn-n)//2:(nn+n)//2, (mm-m)//2:(mm+m)//2] = 1.0
bandlim_ou = 1.0 - bandlim_in
sr_r0, sr_r1 = (nn-n)//2, (nn+n)//2; sr_c0, sr_c1 = (mm-m)//2, (mm+m)//2
w = 0.26
ox, oy = cp.meshgrid(cp.linspace(-dh*mm/2, dh*mm/2, mm), cp.linspace(-dh*nn/2, dh*nn/2, nn))
incident = cp.exp(-((ox**2)+(oy**2))/w) * bandlim_spe
UU = cp.broadcast_to(cp.linspace(-0.5,0.5,mm).reshape(1,mm),(nn,mm)); VV = cp.broadcast_to(cp.linspace(-0.5,0.5,nn).reshape(nn,1),(nn,mm)); RR2 = UU**2+VV**2
Ygrid = (cp.arange(nn).reshape(nn,1) - nn/2); Xgrid = (cp.arange(mm).reshape(1,mm) - mm/2); R2pix = Xgrid**2 + Ygrid**2
_ii = cp.arange(n).reshape(n,1); _jj = cp.arange(m).reshape(1,m)
_denom = 2*cp.cos(2*cp.pi*_ii/n) + 2*cp.cos(2*cp.pi*_jj/m) - 4; _denom[0,0]=1.0
p = os.path.join(ALT_PROJ_DIR, input_tiff)
F1 = np.array(Image.open(p)); F1 = F1[...,0] if F1.ndim==3 else F1
F1 = cv2.resize(F1.astype(np.float32), (m,n), interpolation=cv2.INTER_AREA)
F1 = np.maximum(F1, target_floor_rel*(np.max(F1)+1e-12))
E = float(np.sum(F1)); El = 0.5*E
F = np.pad(np.abs(np.sqrt(F1)), ((n//4,n//4),(m//4,m//4)), mode="constant")
F_gpu = cp.asarray(F); E_gpu = cp.asarray(E); El_gpu = cp.asarray(El); F1_gpu = cp.asarray(F1)

#%% Projector + reconstruction
def irrotational_phase(pha):
    dx = cp.zeros((n,m)); dy = cp.zeros((n,m))
    dx[:,:-1] = cp.mod(pha[:,1:]-pha[:,:-1]+cp.pi, 2*cp.pi)-cp.pi
    dy[:-1,:] = cp.mod(pha[1:,:]-pha[:-1,:]+cp.pi, 2*cp.pi)-cp.pi
    rho = cp.zeros((n,m)); rho[:,0]=dx[:,0]; rho[:,1:]=dx[:,1:]-dx[:,:-1]
    rho[0,:]+=dy[0,:]; rho[1:,:]+=dy[1:,:]-dy[:-1,:]
    rh = cp.fft.fft2(rho); rh[0,0]=0.0
    return cp.real(cp.fft.ifft2(rh/_denom))

def project_phase(pha):
    pc = pha[sr_r0:sr_r1, sr_c0:sr_c1]
    psi = irrotational_phase(pc); psi = psi + cp.angle(cp.sum(cp.exp(1j*(pc-psi))))
    pn = pha.copy(); pn[sr_r0:sr_r1, sr_c0:sr_c1] = cp.angle((1-alpha)*cp.exp(1j*pc)+alpha*cp.exp(1j*psi))
    return pn

def final_reconstruction(E2_k):
    Rec = cp.fft.fftshift(cp.fft.ifft2(cp.fft.fftshift(incident*cp.exp(1j*cp.angle(E2_k)))))[sr_r0:sr_r1, sr_c0:sr_c1]
    I = cp.abs(Rec)**2; return cp.asnumpy(E_gpu*I/(cp.sum(I)+1e-12))

#%% Signal-region-matched diffuser (the criterion)
def smooth_field(s, ks):
    cp.random.seed(s); noise = cp.random.randn(nn,mm)
    ky = (cp.fft.fftfreq(nn)*nn).reshape(nn,1); kx = (cp.fft.fftfreq(mm)*mm).reshape(1,mm)
    lp = cp.exp(-(kx**2+ky**2)/(2*ks**2))
    sm = cp.real(cp.fft.ifft2(cp.fft.fft2(noise)*lp)); return (sm-sm.mean())/(sm.std()+1e-12)

def farfield_rms_radius(phi):
    Rec = cp.fft.fftshift(cp.fft.ifft2(cp.fft.fftshift(incident*cp.exp(1j*phi))))
    I = cp.abs(Rec)**2; s = cp.sum(I)+1e-12
    return float(cp.sqrt(cp.sum(I*R2pix)/s).get())

def in_sr_fraction(phi):
    Rec = cp.fft.fftshift(cp.fft.ifft2(cp.fft.fftshift(incident*cp.exp(1j*phi))))
    I = cp.abs(Rec)**2; return float((cp.sum(I*bandlim_in)/(cp.sum(I)+1e-12)).get())

def matched_diffuser(s):
    """Auto-scale a smooth field so its far-field RMS radius fills the SR (criterion)."""
    sm = smooth_field(s, corr_ks)
    target_r = (m/2.0)*fill
    lo, hi = 0.05, 400.0
    for _ in range(28):                       # bisection: far-field radius is monotone in amp
        a = 0.5*(lo+hi)
        if farfield_rms_radius(a*sm) < target_r: lo = a
        else: hi = a
    phi = cp.exp(1j*a*sm)
    return phi, a, in_sr_fraction(a*sm)

#%% GS run with projector
def run(phi):
    cp.random.seed(seed); amp = cp.random.rand(nn,mm)
    RMSE = np.zeros(loop); NUM = np.zeros(loop, dtype=int)
    for i in range(1, loop):
        amp = bandlim_in*F_gpu + bandlim_ou*amp
        E2 = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(amp*phi)))
        E2_k = cp.sqrt((E_gpu+El_gpu)*incident**2/cp.sum(incident**2)) * cp.exp(1j*cp.angle(E2))
        es = cp.fft.fftshift(cp.fft.ifft2(cp.fft.fftshift(E2_k)))
        amp = cp.abs(es); a_in = bandlim_in*amp; a_ou = bandlim_ou*amp
        amp = cp.sqrt(E_gpu)*(a_in/(cp.sqrt(cp.sum(a_in**2))+1e-12)) + cp.sqrt(El_gpu)*(a_ou/(cp.sqrt(cp.sum(a_ou**2))+1e-12))
        Isr = amp[sr_r0:sr_r1, sr_c0:sr_c1]**2; Isr = E_gpu*Isr/(cp.sum(Isr)+1e-12)
        RMSE[i] = float(cp.sqrt(cp.mean((Isr-F1_gpu)**2)).get())
        pha = cp.angle(es)
        po, ne = function_vortex_detection_accegpu(pha[sr_r0:sr_r1, sr_c0:sr_c1], dh, use_cupy=True); NUM[i]=po+ne
        phi = cp.exp(1j*project_phase(pha))
    return dict(RMSE=RMSE, NUM=NUM, I_final=final_reconstruction(E2_k))

def init_vort(phi):
    pc = cp.angle(phi)[sr_r0:sr_r1, sr_c0:sr_c1]
    po, ne = function_vortex_detection_accegpu(pc, dh, use_cupy=True); return int(po+ne)

#%% Run
tname = os.path.splitext(input_tiff)[0]
print(f"Signal-region-matched smooth init on {input_tiff} (floor={target_floor_rel}, alpha={alpha})", flush=True)
cp.random.seed(1)
res = {}
res["random"] = run(cp.exp(1j*2*cp.pi*cp.random.rand(nn,mm)))
res["quadratic"] = run(cp.exp(1j*quad_c*RR2))
print(f"  random     RMSE={res['random']['RMSE'][-1]:8.4f}")
print(f"  quadratic  RMSE={res['quadratic']['RMSE'][-1]:8.4f}")
diff_amps = []
for s in diffuser_seeds:
    phi0, a, frac = matched_diffuser(s); diff_amps.append(a)
    r = run(phi0); res[f"matched_s{s}"] = r
    print(f"  matched_s{s} RMSE={r['RMSE'][-1]:8.4f}  (auto amp={a:.2f}, in-SR frac={frac:.3f}, "
          f"init_vort={init_vort(phi0)}, final_vort={r['NUM'][-1]})", flush=True)

#%% Plots
fig = plt.figure()
for k in ("random", "quadratic", "matched_s1", "matched_s2", "matched_s3"):
    plt.plot(res[k]["RMSE"][1:], label=k)
plt.xlabel("Iteration"); plt.ylabel("RMSE (SR)"); plt.title(f"SR-matched smooth init — {tname}")
plt.legend(fontsize=8)
fig.savefig(os.path.join(output_dir, f"{tname}_rmse.png"), dpi=150, bbox_inches="tight"); plt.show()

panels = [("target", F1), ("random", res["random"]["I_final"]), ("quadratic", res["quadratic"]["I_final"]),
          ("matched_s1", res["matched_s1"]["I_final"]), ("matched_s2", res["matched_s2"]["I_final"])]
fig, ax = plt.subplots(1, len(panels), figsize=(3.2*len(panels), 4))
for a_, (t, im) in zip(ax, panels):
    a_.imshow(im, cmap="gray"); a_.set_title(t, fontsize=8); a_.axis("off")
plt.tight_layout(); fig.savefig(os.path.join(output_dir, f"{tname}_recon.png"), dpi=150, bbox_inches="tight"); plt.show()
print(f"Saved to {output_dir}")
# %%
