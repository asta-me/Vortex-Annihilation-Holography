#%%
"""
TEMP: quadratic (lens) initialization strength sweep -- refine the big win from temp_smooth_init.

Random init -> RMSE ~9.3; a quadratic (lens) init -> ~4.9 on marmo. Here we sweep the lens
strength c to find the optimum and check robustness. phi_init = exp(i * c * (u^2 + v^2)),
u,v in [-0.5,0.5]. Projector alpha fixed. Compare vs random init.

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
input_tiff = "Cat_1.tif"; 
input_tiff = "C:\\Users\\astam\\Desktop\\Target_Imgs\\Lenna.tif"; 
input_tiff = "C:\\Users\\astam\\Desktop\\Target_Imgs\\Baboon.tif"; 

target_floor_rel = 0.05
target_floor_rel = 5e-3

alpha = 0.5
alpha = 1
n = m = 512; nn = n + 2*(n//4); mm = m + 2*(m//4)
# Chen bandwidth criterion: quadratic fringes fill exactly the pad margin (n//4 per side) -> c* = pi*n
c_chen = float(np.pi * n)
c_list = sorted({0.0, 300.0, 600.0, 1000.0, 1500.0, 2200.0, 3000.0, 5000.0, c_chen})  # 0 == random baseline
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "output_temp_quadinit"); os.makedirs(output_dir, exist_ok=True)

#%% Grid + target
bandlim_spe = cp.zeros((nn, mm), cp.float32); bandlim_spe[nn//4:3*nn//4, mm//4:3*mm//4] = 1.0
bandlim_in = cp.zeros((nn, mm), cp.float32); bandlim_in[(nn-n)//2:(nn+n)//2, (mm-m)//2:(mm+m)//2] = 1.0
bandlim_ou = 1.0 - bandlim_in
sr_r0, sr_r1 = (nn-n)//2, (nn+n)//2; sr_c0, sr_c1 = (mm-m)//2, (mm+m)//2
w = 0.26
ox, oy = cp.meshgrid(cp.linspace(-dh*mm/2, dh*mm/2, mm), cp.linspace(-dh*nn/2, dh*nn/2, nn))
incident = cp.exp(-((ox**2)+(oy**2))/w) * bandlim_spe
u = cp.linspace(-0.5, 0.5, mm).reshape(1, mm); v = cp.linspace(-0.5, 0.5, nn).reshape(nn, 1)
UU = cp.broadcast_to(u, (nn, mm)); VV = cp.broadcast_to(v, (nn, mm)); RR2 = UU**2 + VV**2
_ii = cp.arange(n).reshape(n,1); _jj = cp.arange(m).reshape(1,m)
# Neumann (zero-flux) Poisson eigenvalues -> DCT-II solve (better BC than periodic/FFT)
_denom = (2*cp.cos(cp.pi*_ii/n) - 2) + (2*cp.cos(cp.pi*_jj/m) - 2); _denom[0,0] = 1.0
p = os.path.join(ALT_PROJ_DIR, input_tiff)
F1 = np.array(Image.open(p)); F1 = F1[...,0] if F1.ndim==3 else F1
F1 = cv2.resize(F1.astype(np.float32), (m,n), interpolation=cv2.INTER_AREA)
F1 = np.maximum(F1, target_floor_rel*(np.max(F1)+1e-12))
E = float(np.sum(F1)); El = 0.5*E
F = np.pad(np.abs(np.sqrt(F1)), ((n//4,n//4),(m//4,m//4)), mode="constant")
F_gpu = cp.asarray(F); E_gpu = cp.asarray(E); El_gpu = cp.asarray(El); F1_gpu = cp.asarray(F1)

def irrotational_phase(pha):
    dx = cp.zeros((n,m)); dy = cp.zeros((n,m))
    dx[:,:-1] = cp.mod(pha[:,1:]-pha[:,:-1]+cp.pi, 2*cp.pi)-cp.pi
    dy[:-1,:] = cp.mod(pha[1:,:]-pha[:-1,:]+cp.pi, 2*cp.pi)-cp.pi
    rho = cp.zeros((n,m)); rho[:,0]=dx[:,0]; rho[:,1:]=dx[:,1:]-dx[:,:-1]
    rho[0,:]+=dy[0,:]; rho[1:,:]+=dy[1:,:]-dy[:-1,:]
    phi_hat = dctn(rho, type=2, norm="ortho") / _denom; phi_hat[0,0] = 0.0
    return idctn(phi_hat, type=2, norm="ortho")

def final_reconstruction(E2_k):
    hologram = incident * cp.exp(1j*cp.angle(E2_k))
    Rec = cp.fft.fftshift(cp.fft.ifft2(cp.fft.fftshift(hologram)))[sr_r0:sr_r1, sr_c0:sr_c1]
    I = cp.abs(Rec)**2; return cp.asnumpy(E_gpu*I/(cp.sum(I)+1e-12))

def run(c):
    cp.random.seed(seed)
    phi = cp.exp(1j*2*cp.pi*cp.random.rand(nn,mm)) if c == 0.0 else cp.exp(1j*c*RR2)
    amp = cp.random.rand(nn,mm)
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
        pha = cp.angle(es); pc = pha[sr_r0:sr_r1, sr_c0:sr_c1]
        po, ne = function_vortex_detection_accegpu(pc, dh, use_cupy=True); NUM[i]=po+ne
        psi = irrotational_phase(pc); psi = psi + cp.angle(cp.sum(cp.exp(1j*(pc-psi))))
        pn = pha.copy(); pn[sr_r0:sr_r1, sr_c0:sr_c1] = cp.angle((1-alpha)*cp.exp(1j*pc) + alpha*cp.exp(1j*psi))
        phi = cp.exp(1j*pn)
    return dict(RMSE=RMSE, NUM=NUM, I_final=final_reconstruction(E2_k))

#%% Run
print(f"Quadratic-init strength sweep on {input_tiff} (alpha={alpha}, loop={loop})", flush=True)
runs = {}
for c in c_list:
    runs[c] = run(c)
    r = runs[c]
    if c == 0.0: tag = "random"
    elif c == c_chen: tag = f"chen c={c:g}"
    else: tag = f"c={c:g}"
    print(f"  {tag:14s} RMSE={r['RMSE'][-1]:8.4f}  vort={r['NUM'][-1]:5d}", flush=True)

best_c = min([c for c in c_list if c > 0], key=lambda c: runs[c]["RMSE"][-1])
chen_rank = 1 + sorted([c for c in c_list if c > 0], key=lambda c: runs[c]["RMSE"][-1]).index(c_chen)
print(f"\nBest quadratic c={best_c:g}  RMSE={runs[best_c]['RMSE'][-1]:.4f}  (random={runs[0.0]['RMSE'][-1]:.4f})")
print(f"Chen c*={c_chen:g}  RMSE={runs[c_chen]['RMSE'][-1]:.4f}  -> rank {chen_rank}/{len([c for c in c_list if c>0])} among quadratic inits")

fig = plt.figure()
plt.plot([c for c in c_list], [runs[c]["RMSE"][-1] for c in c_list], "o-")
plt.axvline(c_chen, color="r", ls="--", lw=1)
plt.plot([c_chen], [runs[c_chen]["RMSE"][-1]], "r*", ms=14, label=f"Chen c*={c_chen:g}")
plt.xlabel("quadratic strength c (0 = random)"); plt.ylabel("final RMSE (SR)"); plt.legend()
plt.title(f"Quadratic-init sweep — {os.path.splitext(input_tiff)[0]}")
fig.savefig(os.path.join(output_dir, "rmse_vs_c.png"), dpi=150, bbox_inches="tight"); plt.show()

panels = [("target", F1), ("random", runs[0.0]["I_final"]),
          (f"best c={best_c:g}", runs[best_c]["I_final"]), (f"chen c*={c_chen:g}", runs[c_chen]["I_final"])]
fig, ax = plt.subplots(1, 4, figsize=(14, 4))
for a, (t, im) in zip(ax, panels):
    a.imshow(im, cmap="gray"); a.set_title(t, fontsize=9); a.axis("off")
plt.tight_layout(); fig.savefig(os.path.join(output_dir, "recon.png"), dpi=150, bbox_inches="tight"); plt.show()

#%% Chen-criterion verification: does the target*quadratic-phase spectrum fill the SLM active area?
# SLM active area = support of the incident illumination (bandlim_spe box), central mm/2 x nn/2.
from matplotlib.patches import Rectangle
box_r0, box_c0, box_h, box_w = nn//4, mm//4, nn//2, mm//2  # active-area box (center +/- mm//4)
_yy, _xx = cp.mgrid[0:nn, 0:mm]; _r = cp.sqrt((_xx - mm/2.0)**2 + (_yy - nn/2.0)**2)
box_halfwidth = mm//4  # spectrum should spread to ~this radius to fill (not overflow) the SLM

def slm_spectrum(phi):
    S = cp.abs(cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(F_gpu*phi))))**2
    tot = cp.sum(S) + 1e-12
    infrac = float((cp.sum(S*bandlim_spe)/tot).get())      # energy inside active area
    rms = float(cp.sqrt(cp.sum(S*_r**2)/tot).get())        # spectral RMS radius (px)
    return cp.asnumpy(cp.sqrt(S)), infrac, rms

def phi_of(c):
    cp.random.seed(seed)
    return cp.exp(1j*2*cp.pi*cp.random.rand(nn,mm)) if c == 0.0 else cp.exp(1j*c*RR2)

print("\nChen-criterion check (target*phi -> SLM plane): fill active area, don't overflow")
print(f"  active-area half-width = {box_halfwidth} px")
verify = [("random", 0.0), (f"best c={best_c:g}", best_c), (f"chen c*={c_chen:g}", c_chen)]
specs = {}
for name, c in verify:
    S, infrac, rms = slm_spectrum(phi_of(c)); specs[c] = S
    print(f"  {name:16s} in-aperture energy={infrac*100:5.1f}%   spectral RMS radius={rms:6.1f} px", flush=True)

fig, ax = plt.subplots(1, 4, figsize=(15, 4))
ax[0].imshow(F1, cmap="gray"); ax[0].set_title("target", fontsize=10); ax[0].axis("off")
for a, (name, c) in zip(ax[1:], verify):
    S = specs[c]; vmax = np.percentile(S, 99.5)
    a.imshow(S, cmap="inferno", vmax=vmax)
    a.add_patch(Rectangle((box_c0, box_r0), box_w, box_h, fill=False, ec="cyan", lw=1.5))
    a.set_title(f"{name}\nSLM spectrum", fontsize=9); a.axis("off")
plt.tight_layout(); fig.savefig(os.path.join(output_dir, "chen_verify.png"), dpi=150, bbox_inches="tight"); plt.show()

print(f"Saved to {output_dir}")
# %%
