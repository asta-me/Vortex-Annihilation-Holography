#%%
"""
TEMP: sweep the SMOOTHING parameters of a smooth-random starting phase.

Earlier the "lowfreq" vortex-free init used a single (ks=6, amp=3) and did NOT beat random, while
the quadratic lens (huge phase excursion ~hundreds of rad -> strong diffractive spreading) won.
Question: can a smooth-random phase MATCH the quadratic if we tune it -- in particular with a much
LARGER amplitude (a random diffuser) and the right correlation length?

We sweep:
    ks   = Gaussian low-pass width in frequency-index units (small ks = smoother / longer correlation),
    amp  = phase excursion (radians std). Large amp = strong diffuser (but may re-introduce vortices).
phi_init = exp(i * amp * normalized_lowpass_noise). Reference: random and quadratic(c=1500).
Report final RMSE, initial vortex count, final vortex count for each (ks, amp).

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
dh = 0.00374; loop = 300; seed = 42; init_seed = 1
input_tiff = "Cat_black.tif"; target_floor_rel =0.1; alpha = 0.5; quad_c = 1500.0
ks_list = [2.0, 4.0, 8.0, 16.0, 32.0]          # low-pass width (small = smoother)
amp_list = [1.0, 3.0, 10.0, 30.0, 100.0]       # phase excursion (rad)
n = m = 512; nn = n + 2*(n//4); mm = m + 2*(m//4)
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "output_temp_smoothsweep"); os.makedirs(output_dir, exist_ok=True)

#%% Grid + target
bandlim_spe = cp.zeros((nn, mm), cp.float32); bandlim_spe[nn//4:3*nn//4, mm//4:3*mm//4] = 1.0
bandlim_in = cp.zeros((nn, mm), cp.float32); bandlim_in[(nn-n)//2:(nn+n)//2, (mm-m)//2:(mm+m)//2] = 1.0
bandlim_ou = 1.0 - bandlim_in
sr_r0, sr_r1 = (nn-n)//2, (nn+n)//2; sr_c0, sr_c1 = (mm-m)//2, (mm+m)//2
w = 0.26
ox, oy = cp.meshgrid(cp.linspace(-dh*mm/2, dh*mm/2, mm), cp.linspace(-dh*nn/2, dh*nn/2, nn))
incident = cp.exp(-((ox**2)+(oy**2))/w) * bandlim_spe
UU = cp.broadcast_to(cp.linspace(-0.5,0.5,mm).reshape(1,mm),(nn,mm)); VV = cp.broadcast_to(cp.linspace(-0.5,0.5,nn).reshape(nn,1),(nn,mm)); RR2 = UU**2+VV**2
_ii = cp.arange(n).reshape(n,1); _jj = cp.arange(m).reshape(1,m)
_denom = 2*cp.cos(2*cp.pi*_ii/n) + 2*cp.cos(2*cp.pi*_jj/m) - 4; _denom[0,0]=1.0
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
    rh = cp.fft.fft2(rho); rh[0,0]=0.0
    return cp.real(cp.fft.ifft2(rh/_denom))

def project_phase(pha):
    pc = pha[sr_r0:sr_r1, sr_c0:sr_c1]
    psi = irrotational_phase(pc); psi = psi + cp.angle(cp.sum(cp.exp(1j*(pc-psi))))
    pn = pha.copy(); pn[sr_r0:sr_r1, sr_c0:sr_c1] = cp.angle((1-alpha)*cp.exp(1j*pc)+alpha*cp.exp(1j*psi))
    return pn

def smooth_init(ks, amp, s):
    cp.random.seed(s); noise = cp.random.randn(nn,mm)
    ky = (cp.fft.fftfreq(nn)*nn).reshape(nn,1); kx = (cp.fft.fftfreq(mm)*mm).reshape(1,mm)
    lp = cp.exp(-(kx**2+ky**2)/(2*ks**2))
    sm = cp.real(cp.fft.ifft2(cp.fft.fft2(noise)*lp)); sm = (sm-sm.mean())/(sm.std()+1e-12)
    return cp.exp(1j*amp*sm)

def init_vort(phi):
    pc = cp.angle(phi)[sr_r0:sr_r1, sr_c0:sr_c1]
    po, ne = function_vortex_detection_accegpu(pc, dh, use_cupy=True); return int(po+ne)

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
    An = cp.angle(E2_k)
    Rec = cp.fft.fftshift(cp.fft.ifft2(cp.fft.fftshift(incident*cp.exp(1j*An))))[sr_r0:sr_r1, sr_c0:sr_c1]
    If = cp.abs(Rec)**2; If = cp.asnumpy(E_gpu*If/(cp.sum(If)+1e-12))
    return RMSE[-1], NUM[-1], If

#%% Run references
cp.random.seed(init_seed)
rnd_rmse, _, rnd_I = run(cp.exp(1j*2*cp.pi*cp.random.rand(nn,mm)))
quad_rmse, quad_v, quad_I = run(cp.exp(1j*quad_c*RR2))
print(f"Smoothing sweep on {input_tiff} (alpha={alpha}, loop={loop})")
print(f"  reference random    RMSE={rnd_rmse:.4f}")
print(f"  reference quadratic RMSE={quad_rmse:.4f}  (final_vort={quad_v})")
print("\n  ks/amp   " + " ".join(f"{a:>8g}" for a in amp_list))
grid = np.zeros((len(ks_list), len(amp_list)))
I_finals = {}
for r, ks in enumerate(ks_list):
    row = []
    for c, amp in enumerate(amp_list):
        phi0 = smooth_init(ks, amp, init_seed); iv = init_vort(phi0)
        rmse, fv, If = run(phi0); grid[r, c] = rmse; I_finals[(r, c)] = If
        row.append(f"{rmse:8.3f}")
    print(f"  {ks:8g} " + " ".join(row), flush=True)

best = np.unravel_index(np.argmin(grid), grid.shape)
print(f"\nBest smooth: ks={ks_list[best[0]]:g}, amp={amp_list[best[1]]:g}  RMSE={grid[best]:.4f}  "
      f"(quadratic {quad_rmse:.4f}, random {rnd_rmse:.4f})")

#%% Heatmap
fig = plt.figure(figsize=(7,5))
plt.imshow(grid, cmap="viridis_r", aspect="auto")
plt.colorbar(label="final RMSE (SR)")
plt.xticks(range(len(amp_list)), [f"{a:g}" for a in amp_list]); plt.xlabel("amp (rad)")
plt.yticks(range(len(ks_list)), [f"{k:g}" for k in ks_list]); plt.ylabel("ks (low-pass width)")
plt.title(f"Smooth-random init RMSE — {os.path.splitext(input_tiff)[0]}\n(quadratic={quad_rmse:.2f}, random={rnd_rmse:.2f})")
for r in range(len(ks_list)):
    for c in range(len(amp_list)):
        plt.text(c, r, f"{grid[r,c]:.1f}", ha="center", va="center", color="w", fontsize=8)
fig.savefig(os.path.join(output_dir, "heatmap.png"), dpi=150, bbox_inches="tight"); plt.show()

#%% Reconstruction grid over the (ks, amp) sweep
fig, axes = plt.subplots(len(ks_list), len(amp_list), figsize=(2.2*len(amp_list), 2.2*len(ks_list)))
for r in range(len(ks_list)):
    for c in range(len(amp_list)):
        ax = axes[r, c]
        ax.imshow(I_finals[(r, c)], cmap="gray"); ax.axis("off")
        ax.set_title(f"ks{ks_list[r]:g} a{amp_list[c]:g}\n{grid[r,c]:.2f}", fontsize=7)
plt.suptitle(f"Reconstructions over the smooth-init sweep — {os.path.splitext(input_tiff)[0]}")
plt.tight_layout()
fig.savefig(os.path.join(output_dir, "recon_grid.png"), dpi=150, bbox_inches="tight"); plt.show()

#%% Reference comparison: target / random / quadratic / best-smooth
br, bc = best
panels = [("target", F1), (f"random {rnd_rmse:.2f}", rnd_I), (f"quadratic {quad_rmse:.2f}", quad_I),
          (f"best smooth ks{ks_list[br]:g} a{amp_list[bc]:g} {grid[best]:.2f}", I_finals[best])]
fig, ax = plt.subplots(1, len(panels), figsize=(4*len(panels), 4))
for a, (t, im) in zip(ax, panels):
    a.imshow(im, cmap="gray"); a.set_title(t, fontsize=9); a.axis("off")
plt.tight_layout(); fig.savefig(os.path.join(output_dir, "recon_compare.png"), dpi=150, bbox_inches="tight"); plt.show()
print(f"Saved to {output_dir}")
# %%
