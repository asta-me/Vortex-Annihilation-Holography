#%%
"""
CRITERION for the low-pass width: let the data pick it.

We sweep corr_len_px (the smoothness knob) and plot the FINAL RMSE of the full projector-GS loop.
If the minimum is a broad flat plateau, the message is simple: the exact value does not matter --
pick anything in the plateau (we default to 15). Same geometry / loop as the comparison script.

Env: vortex (conda), GPU (CuPy).
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
dh = 0.00374; loop = 200; seed = 42
input_tiff = "C:\\Users\\astam\\Desktop\\Target_Imgs\\Lenna.tif"
target_floor_rel = 5e-3; alpha = 1; grad_cap = 0.9
corr_lens = np.unique(np.round(np.geomspace(4, 120, 16))).astype(float)   # sweep range [px]
init_seeds = [1, 2]
n = m = 512; nn = n + 2*(n//4); mm = m + 2*(m//4)
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "output_user_compare"); os.makedirs(output_dir, exist_ok=True)

#%% Grid + target
bandlim_spe = cp.zeros((nn, mm), cp.float32); bandlim_spe[nn//4:3*nn//4, mm//4:3*mm//4] = 1.0
bandlim_in = cp.zeros((nn, mm), cp.float32); bandlim_in[(nn-n)//2:(nn+n)//2, (mm-m)//2:(mm+m)//2] = 1.0
bandlim_ou = 1.0 - bandlim_in
sr_r0, sr_r1 = (nn-n)//2, (nn+n)//2; sr_c0, sr_c1 = (mm-m)//2, (mm+m)//2
w = 0.26
ox, oy = cp.meshgrid(cp.linspace(-dh*mm/2, dh*mm/2, mm), cp.linspace(-dh*nn/2, dh*nn/2, nn))
incident = cp.exp(-((ox**2)+(oy**2))/w) * bandlim_spe
_ii = cp.arange(n).reshape(n,1); _jj = cp.arange(m).reshape(1,m)
_denom = (2*cp.cos(cp.pi*_ii/n) - 2) + (2*cp.cos(cp.pi*_jj/m) - 2); _denom[0,0]=1.0
p = input_tiff if os.path.isabs(input_tiff) else os.path.join(ALT_PROJ_DIR, input_tiff)
F1 = np.array(Image.open(p)); F1 = F1[...,0] if F1.ndim==3 else F1
F1 = cv2.resize(F1.astype(np.float32), (m,n), interpolation=cv2.INTER_AREA)
F1 = np.maximum(F1, target_floor_rel*(np.max(F1)+1e-12))
E = float(np.sum(F1)); El = 0.5*E
F = np.pad(np.abs(np.sqrt(F1)), ((n//4,n//4),(m//4,m//4)), mode="constant")
F_gpu = cp.asarray(F); E_gpu = cp.asarray(E); El_gpu = cp.asarray(El); F1_gpu = cp.asarray(F1)

#%% Projector + init + GS
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

def vortexfree_phase(corr_len_px, s):
    cp.random.seed(s); noise = cp.random.randn(nn, mm)
    ks = nn / (2*np.pi*corr_len_px)
    ky = (cp.fft.fftfreq(nn)*nn).reshape(nn,1); kx = (cp.fft.fftfreq(mm)*mm).reshape(1,mm)
    lp = cp.exp(-(kx**2+ky**2)/(2*ks**2))
    sm = cp.real(cp.fft.ifft2(cp.fft.fft2(noise)*lp))
    gmax = float(cp.maximum(cp.abs(cp.diff(sm,axis=1)).max(), cp.abs(cp.diff(sm,axis=0)).max()).get())
    return cp.exp(1j*(grad_cap*np.pi/(gmax+1e-12))*sm)

def run_rmse(phi):
    cp.random.seed(seed); amp = cp.random.rand(nn,mm); rmse = 0.0
    for i in range(1, loop):
        amp = bandlim_in*F_gpu + bandlim_ou*amp
        E2 = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(amp*phi)))
        E2_k = cp.sqrt((E_gpu+El_gpu)*incident**2/cp.sum(incident**2)) * cp.exp(1j*cp.angle(E2))
        es = cp.fft.fftshift(cp.fft.ifft2(cp.fft.fftshift(E2_k)))
        amp = cp.abs(es); a_in = bandlim_in*amp; a_ou = bandlim_ou*amp
        amp = cp.sqrt(E_gpu)*(a_in/(cp.sqrt(cp.sum(a_in**2))+1e-12)) + cp.sqrt(El_gpu)*(a_ou/(cp.sqrt(cp.sum(a_ou**2))+1e-12))
        phi = cp.exp(1j*project_phase(cp.angle(es)))
    Isr = amp[sr_r0:sr_r1, sr_c0:sr_c1]**2; Isr = E_gpu*Isr/(cp.sum(Isr)+1e-12)
    return float(cp.sqrt(cp.mean((Isr-F1_gpu)**2)).get())

#%% Sweep
print(f"corr_len sweep on Lenna (loop={loop}, grad_cap={grad_cap}); RMSE averaged over seeds {init_seeds}\n", flush=True)
mean_rmse = np.zeros(len(corr_lens)); all_rmse = np.zeros((len(corr_lens), len(init_seeds)))
for k, L in enumerate(corr_lens):
    for j, s in enumerate(init_seeds):
        all_rmse[k, j] = run_rmse(vortexfree_phase(L, s))
    mean_rmse[k] = all_rmse[k].mean()
    print(f"  corr_len={L:6.1f} px  ->  RMSE={mean_rmse[k]:8.4f}", flush=True)

best = int(np.argmin(mean_rmse))
lo = mean_rmse.min(); plateau = corr_lens[mean_rmse <= lo*1.02]     # within 2% of the best
print(f"\nBest corr_len = {corr_lens[best]:.1f} px (RMSE={lo:.4f}); "
      f"within 2% of best: {plateau.min():.0f}-{plateau.max():.0f} px", flush=True)

#%% Plot
fig = plt.figure(figsize=(7, 4.5))
for j, s in enumerate(init_seeds):
    plt.plot(corr_lens, all_rmse[:, j], "o-", alpha=0.35, color="tab:gray", label="per seed" if j==0 else None)
plt.plot(corr_lens, mean_rmse, "o-", color="tab:blue", lw=2, label="mean")
plt.axvspan(plateau.min(), plateau.max(), color="tab:green", alpha=0.15, label="within 2% of best")
plt.axvline(15, color="tab:red", ls="--", label="default = 15")
plt.xscale("log"); plt.xlabel("corr_len_px  (smoothness knob)"); plt.ylabel("final RMSE (SR)")
plt.title("Final RMSE vs low-pass correlation length — Lenna"); plt.legend(fontsize=8)
plt.tight_layout(); fig.savefig(os.path.join(output_dir, "corrlen_sweep.png"), dpi=150, bbox_inches="tight"); plt.show()
print(f"Saved {os.path.join(output_dir, 'corrlen_sweep.png')}")
# %%
