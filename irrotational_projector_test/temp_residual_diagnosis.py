#%%
"""
TEMP diagnosis: is the residual error (at convergence, with the projector) still due to VORTICES,
or to something else (edges / amplitude-constraint tension / dark background)?

Runs the plain projector (const alpha) to convergence on `input_tiff`, then decomposes the final
SR squared-error map:
    - error inside a dilated neighborhood of the residual vortex cores vs outside,
    - error in bright vs dark target regions,
    - error near strong target EDGES (|grad target|) vs flat regions,
    - correlation of the error map with |grad target|.
If the error is NOT concentrated on the (few) residual vortices, then further RMSE gains must come
from the phase-retrieval engine (MRAF / feedback / init), not from better vortex removal.

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
n = m = 512; nn = n + 2*(n//4); mm = m + 2*(m//4)
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "output_temp_diagnosis"); os.makedirs(output_dir, exist_ok=True)

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

def irrotational_phase(pha):
    dx = cp.zeros((n,m)); dy = cp.zeros((n,m))
    dx[:,:-1] = cp.mod(pha[:,1:]-pha[:,:-1]+cp.pi, 2*cp.pi)-cp.pi
    dy[:-1,:] = cp.mod(pha[1:,:]-pha[:-1,:]+cp.pi, 2*cp.pi)-cp.pi
    rho = cp.zeros((n,m)); rho[:,0]=dx[:,0]; rho[:,1:]=dx[:,1:]-dx[:,:-1]
    rho[0,:]+=dy[0,:]; rho[1:,:]+=dy[1:,:]-dy[:-1,:]
    rh = cp.fft.fft2(rho); rh[0,0]=0.0
    return cp.real(cp.fft.ifft2(rh/_denom))

def vortex_mask(pha):
    n_, m_ = pha.shape
    gy = cp.vstack([cp.angle(cp.exp(1j*cp.diff(pha,axis=0))), cp.zeros((1,m_))])
    gx = cp.hstack([cp.angle(cp.exp(1j*cp.diff(pha,axis=1))), cp.zeros((n_,1))])
    gy_m1 = cp.hstack([gy[:,1:m_], cp.zeros((n_,1))]); gx_n1 = cp.vstack([gx[1:n_,:], cp.zeros((1,m_))])
    g = gx + gy_m1 - gx_n1 - gy; th = 2*cp.pi - 0.1
    return (g > th) | ((-g) > th)

#%% Run projector to convergence
cp.random.seed(seed)
phi = cp.exp(1j*2*cp.pi*cp.random.rand(nn,mm)); amp = cp.random.rand(nn,mm)
RMSE = np.zeros(loop); NUM = np.zeros(loop, dtype=int)
for i in range(1, loop):
    amp = bandlim_in*F_gpu + bandlim_ou*amp
    E2 = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(amp*phi)))
    E2_ave = cp.sqrt((E_gpu+El_gpu)*incident**2/cp.sum(incident**2))
    E2_k = E2_ave*cp.exp(1j*cp.angle(E2))
    es = cp.fft.fftshift(cp.fft.ifft2(cp.fft.fftshift(E2_k)))
    amp = cp.abs(es); a_in = bandlim_in*amp; a_ou = bandlim_ou*amp
    amp = cp.sqrt(E_gpu)*(a_in/(cp.sqrt(cp.sum(a_in**2))+1e-12)) + cp.sqrt(El_gpu)*(a_ou/(cp.sqrt(cp.sum(a_ou**2))+1e-12))
    I = amp[sr_r0:sr_r1, sr_c0:sr_c1]**2; I = E_gpu*I/(cp.sum(I)+1e-12)
    RMSE[i] = float(cp.sqrt(cp.mean((I-F1_gpu)**2)).get())
    pha = cp.angle(es); pc = pha[sr_r0:sr_r1, sr_c0:sr_c1]
    po, ne = function_vortex_detection_accegpu(pc, dh, use_cupy=True); NUM[i] = po+ne
    psi = irrotational_phase(pc); psi = psi + cp.angle(cp.sum(cp.exp(1j*(pc-psi))))
    field = (1-alpha)*cp.exp(1j*pc) + alpha*cp.exp(1j*psi)
    pn = pha.copy(); pn[sr_r0:sr_r1, sr_c0:sr_c1] = cp.angle(field); phi = cp.exp(1j*pn)

#%% Diagnosis on the final state
I = (amp[sr_r0:sr_r1, sr_c0:sr_c1]**2); I = E_gpu*I/(cp.sum(I)+1e-12)
err = (I - F1_gpu)**2
pc = cp.angle(es)[sr_r0:sr_r1, sr_c0:sr_c1]
vmask = vortex_mask(pc).astype(cp.float32)
vd = vmask.copy()
for _ in range(3):
    vd = cp.clip(vd + cp.roll(vd,1,0)+cp.roll(vd,-1,0)+cp.roll(vd,1,1)+cp.roll(vd,-1,1), 0, 1)
vd_bool = vd > 0
F1n = F1_gpu / (cp.max(F1_gpu)+1e-12)
bright = F1n > 0.5; dark = F1n < 0.10
gy = cp.abs(cp.diff(F1_gpu, axis=0, append=F1_gpu[-1:,:])); gx = cp.abs(cp.diff(F1_gpu, axis=1, append=F1_gpu[:,-1:]))
gmag = cp.sqrt(gx**2+gy**2); edge = gmag > float(cp.percentile(gmag, 90).get())

def frac(mask): return float((cp.sum(err*mask)/(cp.sum(err)+1e-12)).get())
def areafrac(mask): return float(cp.mean(mask.astype(cp.float32)).get())
e = (err - cp.mean(err)); g = (gmag - cp.mean(gmag))
corr = float((cp.sum(e*g)/(cp.sqrt(cp.sum(e**2))*cp.sqrt(cp.sum(g**2))+1e-12)).get())

print(f"Residual diagnosis on {input_tiff} (alpha={alpha}, loop={loop})")
print(f"  final RMSE={RMSE[-1]:.4f}  final vortices={NUM[-1]}")
print(f"  SSE in DILATED vortex cores : {frac(vd_bool)*100:6.2f}%  (area {areafrac(vd_bool)*100:5.2f}%)")
print(f"  SSE in BRIGHT (>0.5)        : {frac(bright)*100:6.2f}%  (area {areafrac(bright)*100:5.2f}%)")
print(f"  SSE in DARK   (<0.1)        : {frac(dark)*100:6.2f}%  (area {areafrac(dark)*100:5.2f}%)")
print(f"  SSE in EDGES  (top10% grad) : {frac(edge)*100:6.2f}%  (area {areafrac(edge)*100:5.2f}%)")
print(f"  corr(error, |grad target|)  : {corr:+.3f}")

#%% Figure
fig, ax = plt.subplots(1, 4, figsize=(16,4))
ax[0].imshow(F1, cmap="gray"); ax[0].set_title("target"); ax[0].axis("off")
ax[1].imshow(cp.asnumpy(I), cmap="gray"); ax[1].set_title(f"recon (RMSE {RMSE[-1]:.2f})"); ax[1].axis("off")
ax[2].imshow(cp.asnumpy(err), cmap="inferno"); ax[2].set_title("squared error map"); ax[2].axis("off")
ax[3].imshow(cp.asnumpy(gmag), cmap="viridis"); ax[3].set_title("|grad target|"); ax[3].axis("off")
plt.tight_layout(); fig.savefig(os.path.join(output_dir, "diagnosis.png"), dpi=150, bbox_inches="tight"); plt.show()
print(f"Saved to {output_dir}")
# %%
