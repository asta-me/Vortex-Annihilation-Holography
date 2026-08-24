#%%
"""
DCT-projector INITIALIZATION comparison for vortex-annihilation holography.

All three cases run the SAME Gerchberg-Saxton loop with the irrotational (vortex-free) PROJECTOR
applied every iteration, solved with a DCT-II / Neumann Poisson solve (the natural zero-flux BC,
no guard-padding artifact). Only the INITIAL phase differs:

  1. random          -- exp(i*2*pi*U): a flat random phase. Seeds ~1e5 vortices and lands in a poor
                        GS basin. Run for 3 seeds (plotted dashed, red).
  2. quadratic (Chen)-- exp(i*c*(u^2+v^2)) with c* = pi*SR (Chen bandwidth criterion: the lens
                        fringes fill the pad margin). Deterministic, ~0 vortices (solid, green).
  3. random filtered -- a smooth random phase whose spectrum FITS THE SLM aperture: low-pass white
     (SLM-fit)         noise with amplitude capped at Nyquist (max|grad phi| < grad_cap*pi). Being
                        single-valued AND alias-free it has ZERO vortices BY CONSTRUCTION (no
                        projection needed). Run for 3 seeds (dashed, blue).

Why (3) works: a vortex is a wrapped-phase circulation of +-2*pi; if every per-pixel gradient stays
below pi the wrapped gradient equals the true gradient, so a single-valued smooth field carries no
vortices. The low-pass lowers the peak/RMS gradient ratio, so the Nyquist cap still lets the far-
field fill the signal region (and the image*phi spectrum fill the SLM aperture -- verified at the
end). Typical result on Lenna: random ~7.9, quadratic ~5.4, filtered ~4.5 (best), consistent across
seeds and vortex-free from iteration 0.

Env: vortex (conda), GPU (CuPy). PIL + cv2 (no skimage).
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import cupy as cp
from cupyx.scipy.fft import dctn, idctn
import cv2
from PIL import Image

ALT_PROJ_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "1_Alternative_projection")
sys.path.insert(0, ALT_PROJ_DIR)
from function_vortex_detection_accegpu import function_vortex_detection_accegpu

#%% Config
dh = 0.00374; loop = 300; seed = 42

# Target: uncomment ONE line (files live in ./targets/, or give an absolute path).
input_tiff = "marmo.tif"
# input_tiff = "object_grayscale_from_mat.tif"
# input_tiff = "Lenna.tif"
# input_tiff = "Baboon.tif"
# input_tiff = "valentini.tif"

target_floor_rel = 5e-3; alpha = 1
corr_ks = 8.0                 # low-pass width for the smooth field (moderate correlation length)
grad_cap = 0.9                # Nyquist cap: max|grad phi| < grad_cap*pi -> alias-free -> zero vortices
diffuser_seeds = [1, 2, 3]

# Chen quadratic (lens) init -- tunable knobs, defaulted to the values we found.
QUAD_C_MODE = "chen"          # "chen" -> c* = pi*SR (bandwidth criterion); "manual" -> QUAD_C_MANUAL
QUAD_C_MANUAL = 1500.0        # used only when QUAD_C_MODE == "manual" (empirical optimum ~1500 on many targets)

#%% Geometry (fix the hologram size and the SR fraction; the rest is derived)
HOLOGRAM_SIZE = 384           # SLM aperture side [px] (= half the work grid; 2x oversampling)
WORK_SIZE = 2 * HOLOGRAM_SIZE  # computational grid side [px]
SR_FRACTION = 2 / 3           # signal-region side as a fraction of WORK_SIZE (image size)
SR_SIZE = int(round(SR_FRACTION * WORK_SIZE)); SR_SIZE -= SR_SIZE % 2
assert 0 < SR_SIZE <= WORK_SIZE, "SR_FRACTION out of range: signal region must fit the grid"
n = m = SR_SIZE; nn = mm = WORK_SIZE
pad_each = (WORK_SIZE - SR_SIZE) // 2          # center the SR inside the work grid
ap0 = (WORK_SIZE - HOLOGRAM_SIZE) // 2         # SLM aperture offset
quad_c = float(np.pi * n) if QUAD_C_MODE == "chen" else float(QUAD_C_MANUAL)
script_dir = os.path.dirname(os.path.abspath(__file__))
TARGETS_DIR = os.path.join(script_dir, "targets")

#%% Grid + target
bandlim_spe = cp.zeros((nn, mm), cp.float32); bandlim_spe[ap0:ap0+HOLOGRAM_SIZE, ap0:ap0+HOLOGRAM_SIZE] = 1.0
bandlim_in = cp.zeros((nn, mm), cp.float32); bandlim_in[pad_each:pad_each+n, pad_each:pad_each+m] = 1.0
bandlim_ou = 1.0 - bandlim_in
sr_r0, sr_r1 = (nn-n)//2, (nn+n)//2; sr_c0, sr_c1 = (mm-m)//2, (mm+m)//2
w = 0.26
ox, oy = cp.meshgrid(cp.linspace(-dh*mm/2, dh*mm/2, mm), cp.linspace(-dh*nn/2, dh*nn/2, nn))
incident = cp.exp(-((ox**2)+(oy**2))/w) * bandlim_spe
UU = cp.broadcast_to(cp.linspace(-0.5,0.5,mm).reshape(1,mm),(nn,mm)); VV = cp.broadcast_to(cp.linspace(-0.5,0.5,nn).reshape(nn,1),(nn,mm)); RR2 = UU**2+VV**2
Ygrid = (cp.arange(nn).reshape(nn,1) - nn/2); Xgrid = (cp.arange(mm).reshape(1,mm) - mm/2); R2pix = Xgrid**2 + Ygrid**2
_ii = cp.arange(n).reshape(n,1); _jj = cp.arange(m).reshape(1,m)
# DCT-II (Neumann / zero-flux) Laplacian eigenvalues -- the natural BC, no guard-padding artifact.
_denom = (2*cp.cos(cp.pi*_ii/n) - 2) + (2*cp.cos(cp.pi*_jj/m) - 2); _denom[0,0]=1.0
p = input_tiff if os.path.isabs(input_tiff) else os.path.join(TARGETS_DIR, input_tiff)
F1 = np.array(Image.open(p)); F1 = F1[...,0] if F1.ndim==3 else F1
F1 = cv2.resize(F1.astype(np.float32), (m,n), interpolation=cv2.INTER_AREA)
F1 = np.maximum(F1, target_floor_rel*(np.max(F1)+1e-12))
E = float(np.sum(F1)); El = 0.5*E
F = np.pad(np.abs(np.sqrt(F1)), ((pad_each,pad_each),(pad_each,pad_each)), mode="constant")
F_gpu = cp.asarray(F); E_gpu = cp.asarray(E); El_gpu = cp.asarray(El); F1_gpu = cp.asarray(F1)

#%% Projector + reconstruction
def irrotational_phase(pha):
    dx = cp.zeros((n,m)); dy = cp.zeros((n,m))
    dx[:,:-1] = cp.mod(pha[:,1:]-pha[:,:-1]+cp.pi, 2*cp.pi)-cp.pi
    dy[:-1,:] = cp.mod(pha[1:,:]-pha[:-1,:]+cp.pi, 2*cp.pi)-cp.pi
    rho = cp.zeros((n,m)); rho[:,0]=dx[:,0]; rho[:,1:]=dx[:,1:]-dx[:,:-1]
    rho[0,:]+=dy[0,:]; rho[1:,:]+=dy[1:,:]-dy[:-1,:]
    rh = dctn(rho, type=2, norm="ortho") / _denom; rh[0,0]=0.0
    return idctn(rh, type=2, norm="ortho")

def project_phase(pha):
    pc = pha[sr_r0:sr_r1, sr_c0:sr_c1]
    psi = irrotational_phase(pc); psi = psi + cp.angle(cp.sum(cp.exp(1j*(pc-psi))))
    pn = pha.copy(); pn[sr_r0:sr_r1, sr_c0:sr_c1] = cp.angle((1-alpha)*cp.exp(1j*pc)+alpha*cp.exp(1j*psi))
    return pn

def final_reconstruction(E2_k):
    Rec = cp.fft.fftshift(cp.fft.ifft2(cp.fft.fftshift(incident*cp.exp(1j*cp.angle(E2_k)))))[sr_r0:sr_r1, sr_c0:sr_c1]
    I = cp.abs(Rec)**2; return cp.asnumpy(E_gpu*I/(cp.sum(I)+1e-12))

#%% Initializations (band-limited smooth-random potential + Nyquist cap)
def smooth_field(s, ks):
    cp.random.seed(s); noise = cp.random.randn(nn,mm)                       # random scalar potential
    ky = (cp.fft.fftfreq(nn)*nn).reshape(nn,1); kx = (cp.fft.fftfreq(mm)*mm).reshape(1,mm)
    lp = cp.exp(-(kx**2+ky**2)/(2*ks**2))                                   # Gaussian low-pass (band-limit)
    sm = cp.real(cp.fft.ifft2(cp.fft.fft2(noise)*lp)); return (sm-sm.mean())/(sm.std()+1e-12)

def in_sr_fraction(phi):
    Rec = cp.fft.fftshift(cp.fft.ifft2(cp.fft.fftshift(incident*cp.exp(1j*phi))))
    I = cp.abs(Rec)**2; return float((cp.sum(I*bandlim_in)/(cp.sum(I)+1e-12)).get())

def vortexfree_diffuser(s):
    """Smooth random phase with amplitude capped by Nyquist (max|grad phi| < grad_cap*pi).

    A smooth real field used AS the phase is single-valued (curl-free) by construction; the only
    source of vortices is aliasing when a per-pixel wrapped gradient exceeds pi. Capping the peak
    gradient below pi therefore gives ZERO vortices with no projection. The low-pass (corr_ks)
    lowers the peak/RMS gradient ratio so this cap still lets the far-field fill much of the aperture.
    """
    sm = smooth_field(s, corr_ks)
    gmax = float(cp.maximum(cp.abs(cp.diff(sm, axis=1)).max(), cp.abs(cp.diff(sm, axis=0)).max()).get())
    a = grad_cap*np.pi/(gmax+1e-12)
    phi = cp.exp(1j*a*sm)
    return phi, float(a), in_sr_fraction(a*sm)

def quadratic_init():
    """Chen quadratic (lens) initial phase over the work grid: exp(i * c * (u^2 + v^2)), u,v in [-0.5,0.5].

    The strength c sets how many lens fringes fill the pad margin. Default QUAD_C_MODE='chen' uses
    c* = pi*SR (Chen's bandwidth criterion: fringes just fill the guard band). Deterministic and
    (near) vortex-free. Tune with QUAD_C_MODE / QUAD_C_MANUAL at the top of the script.
    """
    return cp.exp(1j * quad_c * RR2)

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
tname = os.path.splitext(os.path.basename(input_tiff))[0]
print(f"DCT-projector init comparison on {input_tiff} (floor={target_floor_rel}, alpha={alpha})", flush=True)
res = {}

# (2) quadratic init, Chen bandwidth criterion -> continuous line
res["quadratic"] = run(quadratic_init())
print(f"  quadratic     RMSE={res['quadratic']['RMSE'][-1]:8.4f}  (Chen c*={quad_c:g})", flush=True)

# (1) plain random init, 3 seeds -> dashed
for s in diffuser_seeds:
    cp.random.seed(s); phir = cp.exp(1j*2*cp.pi*cp.random.rand(nn,mm))
    res[f"random_s{s}"] = run(phir)
    print(f"  random_s{s}    RMSE={res[f'random_s{s}']['RMSE'][-1]:8.4f}  (init_vort={init_vort(phir)})", flush=True)

# (3) frequency-filtered random whose spectrum fits the SLM (Nyquist-capped), 3 seeds -> dashed
for s in diffuser_seeds:
    phif, af, fracf = vortexfree_diffuser(s)
    res[f"filtered_s{s}"] = run(phif)
    print(f"  filtered_s{s}  RMSE={res[f'filtered_s{s}']['RMSE'][-1]:8.4f}  (cap amp={af:.2f}, "
          f"in-SR frac={fracf:.3f}, init_vort={init_vort(phif)})", flush=True)

#%% Results table (simple markdown, regenerated each run)
tbl = [f"# Init comparison — {tname}",
       f"geometry {HOLOGRAM_SIZE}/{WORK_SIZE}/{SR_SIZE}, loop={loop}, alpha={alpha}, floor={target_floor_rel}, Chen c*={quad_c:g}",
       "", "| init | final RMSE (SR) |", "|---|---|",
       f"| quadratic (Chen) | {res['quadratic']['RMSE'][-1]:.4f} |"]
for s in diffuser_seeds:
    tbl.append(f"| random_s{s} | {res[f'random_s{s}']['RMSE'][-1]:.4f} |")
for s in diffuser_seeds:
    tbl.append(f"| filtered_s{s} | {res[f'filtered_s{s}']['RMSE'][-1]:.4f} |")
table_md = "\n".join(tbl) + "\n"
table_path = os.path.join(script_dir, f"results_smooth_init_{tname}.md")
with open(table_path, "w", encoding="utf-8") as fh:
    fh.write(table_md)
print(f"\nResults table -> {table_path}", flush=True)

#%% Plots -- quadratic (Chen) solid; random (3 seeds) dashed; frequency-filtered (3 seeds) dashed
fig = plt.figure()
plt.plot(res["quadratic"]["RMSE"][1:], "-", color="tab:green", lw=2, label="quadratic (Chen)")
for j, s in enumerate(diffuser_seeds):
    plt.plot(res[f"random_s{s}"]["RMSE"][1:], "--", color="tab:red", alpha=0.85,
             label="random" if j == 0 else None)
    plt.plot(res[f"filtered_s{s}"]["RMSE"][1:], "--", color="tab:blue", alpha=0.85,
             label="random filtered (SLM-fit)" if j == 0 else None)
plt.xlabel("Iteration"); plt.ylabel("RMSE (SR)"); plt.title(f"DCT-projector init comparison — {tname}")
plt.legend(fontsize=8)

panels = [("target", F1), ("quadratic", res["quadratic"]["I_final"]),
          ("random_s1", res["random_s1"]["I_final"]), ("filtered_s1", res["filtered_s1"]["I_final"])]
fig, ax = plt.subplots(1, len(panels), figsize=(3.2*len(panels), 4))
for a_, (t, im) in zip(ax, panels):
    a_.imshow(im, cmap="gray"); a_.set_title(t, fontsize=8); a_.axis("off")
plt.tight_layout()

#%% SLM-occupancy check: does the IMAGE with the diffuser applied fill the SLM aperture in transform?
# Exactly like the Chen check in temp_quadratic_init_sweep.py, but for these diffuser inits: take the
# target amplitude F with the init phase on top (F*exp(i*phi)), transform to the SLM plane, and see
# whether the spectrum fills (not overflows) the SLM aperture. (Uses the image, NOT the bare incident.)
from matplotlib.patches import Rectangle
box_c0, box_r0, box_w, box_h = mm//4, nn//4, mm//2, nn//2      # SLM aperture (support of bandlim_spe)

def slm_spectrum(phi):
    S = cp.abs(cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(F_gpu*cp.exp(1j*phi)))))**2
    tot = cp.sum(S) + 1e-12
    infrac = float((cp.sum(S*bandlim_spe)/tot).get())         # energy inside the SLM aperture
    rms = float(cp.sqrt(cp.sum(S*R2pix)/tot).get())           # spectral RMS radius (px)
    return cp.asnumpy(cp.sqrt(S)), infrac, rms

cp.random.seed(diffuser_seeds[0]); phi_random = cp.exp(1j*2*cp.pi*cp.random.rand(nn,mm))
inits = [("random", phi_random), (f"quadratic Chen c*={quad_c:g}", cp.exp(1j*quad_c*RR2))]
for s in diffuser_seeds:
    inits.append((f"filtered_s{s}", vortexfree_diffuser(s)[0]))
specs = {name: slm_spectrum(phi) for name, phi in inits}

print(f"\nSLM-occupancy check (image*diffuser -> SLM plane), aperture half-width = {mm//4} px:")
for name, (_, infrac, rms) in specs.items():
    print(f"  {name:22s} in-aperture energy={infrac*100:5.1f}%   spectral RMS radius={rms:6.1f} px", flush=True)

# Figure: target + seed-1 representatives (keep it readable; printout above covers all seeds).
s0 = diffuser_seeds[0]
show = ["random", f"quadratic Chen c*={quad_c:g}", f"filtered_s{s0}"]
fig, ax = plt.subplots(1, len(show)+1, figsize=(3.2*(len(show)+1), 3.6))
ax[0].imshow(F1, cmap="gray"); ax[0].set_title(f"target ({tname})", fontsize=8); ax[0].axis("off")
for a_, name in zip(ax[1:], show):
    S, infrac, _ = specs[name]; vmax = np.percentile(S, 99.5)
    a_.imshow(S, cmap="inferno", vmax=vmax)
    a_.add_patch(Rectangle((box_c0, box_r0), box_w, box_h, fill=False, ec="cyan", lw=1.5))  # SLM aperture
    a_.set_title(f"{name}\nin-ap {infrac*100:.0f}%", fontsize=8); a_.axis("off")
plt.tight_layout()

plt.show()
# %%
