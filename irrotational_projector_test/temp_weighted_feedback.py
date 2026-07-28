#%%
"""
TEMP: weighted-feedback GS (adaptive target amplitude) + projector.

The residual error is a broadly-distributed under/over-reconstruction in the bright signal region.
Weighted/feedback GS (Wu et al.) fights exactly this: keep a WORKING target amplitude T that is
adaptively re-weighted each iteration to boost under-reconstructed pixels:
    T <- clip( T * (target_amp / recon_amp)^beta )   (then energy-renormalized)
and use T (not the fixed target) as the SR amplitude constraint. RMSE is always measured against
the TRUE target. beta=0 reproduces the standard hard-constraint GS. Projector ON in all runs.

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
beta_list = [0.0, 0.2, 0.5, 1.0]
clip_lo, clip_hi = 0.5, 2.0
n = m = 512; nn = n + 2*(n//4); mm = m + 2*(m//4)
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "output_temp_feedback"); os.makedirs(output_dir, exist_ok=True)

#%% Grid
bandlim_spe = cp.zeros((nn, mm), cp.float32); bandlim_spe[nn//4:3*nn//4, mm//4:3*mm//4] = 1.0
bandlim_in = cp.zeros((nn, mm), cp.float32); bandlim_in[(nn-n)//2:(nn+n)//2, (mm-m)//2:(mm+m)//2] = 1.0
bandlim_ou = 1.0 - bandlim_in
sr_r0, sr_r1 = (nn-n)//2, (nn+n)//2; sr_c0, sr_c1 = (mm-m)//2, (mm+m)//2
w = 0.26
ox, oy = cp.meshgrid(cp.linspace(-dh*mm/2, dh*mm/2, mm), cp.linspace(-dh*nn/2, dh*nn/2, nn))
incident = cp.exp(-((ox**2)+(oy**2))/w) * bandlim_spe
_ii = cp.arange(n).reshape(n,1); _jj = cp.arange(m).reshape(1,m)
_denom = 2*cp.cos(2*cp.pi*_ii/n) + 2*cp.cos(2*cp.pi*_jj/m) - 4; _denom[0,0] = 1.0

#%% Target
p = os.path.join(ALT_PROJ_DIR, input_tiff)
F1 = np.array(Image.open(p)); F1 = F1[...,0] if F1.ndim==3 else F1
F1 = cv2.resize(F1.astype(np.float32), (m,n), interpolation=cv2.INTER_AREA)
F1 = np.maximum(F1, target_floor_rel*(np.max(F1)+1e-12))
E = float(np.sum(F1)); El = 0.5*E
F = np.pad(np.abs(np.sqrt(F1)), ((n//4,n//4),(m//4,m//4)), mode="constant")
F_gpu = cp.asarray(F); E_gpu = cp.asarray(E); El_gpu = cp.asarray(El); F1_gpu = cp.asarray(F1)
Atgt_sr = cp.asarray(np.sqrt(F1).astype(np.float32))   # true target amplitude on the SR (512x512)

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
    field = (1-alpha)*cp.exp(1j*pc) + alpha*cp.exp(1j*psi)
    pn = pha.copy(); pn[sr_r0:sr_r1, sr_c0:sr_c1] = cp.angle(field)
    return pn

def final_reconstruction(E2_k):
    hologram = incident * cp.exp(1j*cp.angle(E2_k))
    Rec = cp.fft.fftshift(cp.fft.ifft2(cp.fft.fftshift(hologram)))[sr_r0:sr_r1, sr_c0:sr_c1]
    I = cp.abs(Rec)**2; return cp.asnumpy(E_gpu*I/(cp.sum(I)+1e-12))


def run_feedback(beta):
    cp.random.seed(seed)
    phi = cp.exp(1j*2*cp.pi*cp.random.rand(nn,mm)); amp = cp.random.rand(nn,mm)
    RMSE = np.zeros(loop); NUM = np.zeros(loop, dtype=int)
    Twork = F_gpu.copy()                       # working (padded) SR target amplitude, adaptive
    for i in range(1, loop):
        amp = bandlim_in*Twork + bandlim_ou*amp     # SR uses the (adaptive) working target
        E2 = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(amp*phi)))
        E2_k = cp.sqrt((E_gpu+El_gpu)*incident**2/cp.sum(incident**2)) * cp.exp(1j*cp.angle(E2))
        es = cp.fft.fftshift(cp.fft.ifft2(cp.fft.fftshift(E2_k)))
        amp = cp.abs(es); a_in = bandlim_in*amp; a_ou = bandlim_ou*amp
        amp = cp.sqrt(E_gpu)*(a_in/(cp.sqrt(cp.sum(a_in**2))+1e-12)) + cp.sqrt(El_gpu)*(a_ou/(cp.sqrt(cp.sum(a_ou**2))+1e-12))
        # RMSE vs TRUE target
        Isr = amp[sr_r0:sr_r1, sr_c0:sr_c1]**2; Isr = E_gpu*Isr/(cp.sum(Isr)+1e-12)
        RMSE[i] = float(cp.sqrt(cp.mean((Isr-F1_gpu)**2)).get())
        pha = cp.angle(es)
        po, ne = function_vortex_detection_accegpu(pha[sr_r0:sr_r1, sr_c0:sr_c1], dh, use_cupy=True); NUM[i]=po+ne
        phi = cp.exp(1j*project_phase(pha))
        # feedback update of the working target amplitude (SR only)
        if beta > 0.0:
            a_rec = cp.sqrt(Isr)                                   # recon amplitude (energy E) on SR
            ratio = cp.clip(Atgt_sr / (a_rec + 1e-6), clip_lo, clip_hi) ** beta
            Tw_sr = Twork[sr_r0:sr_r1, sr_c0:sr_c1] * ratio
            Tw_sr = cp.sqrt(E_gpu) * Tw_sr / (cp.sqrt(cp.sum(Tw_sr**2)) + 1e-12)   # renormalize energy E
            Twork = cp.zeros((nn, mm)); Twork[sr_r0:sr_r1, sr_c0:sr_c1] = Tw_sr
    return dict(RMSE=RMSE, NUM=NUM, I_final=final_reconstruction(E2_k))


#%% Run
print(f"Weighted-feedback GS on {input_tiff} (alpha={alpha}, loop={loop})", flush=True)
runs = {}
for b in beta_list:
    runs[f"beta{b}"] = run_feedback(b)
    r = runs[f"beta{b}"]
    print(f"  beta={b:<4} RMSE={r['RMSE'][-1]:8.4f}  vort={r['NUM'][-1]:5d}", flush=True)

#%% Plots
fig = plt.figure()
for name, r in runs.items():
    plt.plot(r["RMSE"][1:], label=name)
plt.xlabel("Iteration"); plt.ylabel("RMSE (SR)"); plt.title(f"Weighted-feedback GS — {os.path.splitext(input_tiff)[0]}")
plt.legend(fontsize=8)
fig.savefig(os.path.join(output_dir, "rmse.png"), dpi=150, bbox_inches="tight"); plt.show()

panels = [("target", F1)] + [(nm, r["I_final"]) for nm, r in runs.items()]
fig, ax = plt.subplots(1, len(panels), figsize=(3.2*len(panels), 4))
for a, (t, im) in zip(ax, panels):
    a.imshow(im, cmap="gray"); a.set_title(t, fontsize=8); a.axis("off")
plt.tight_layout(); fig.savefig(os.path.join(output_dir, "recon.png"), dpi=150, bbox_inches="tight"); plt.show()
print(f"Saved to {output_dir}")
# %%
