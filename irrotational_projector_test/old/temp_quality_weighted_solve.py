#%%
"""
TEMP: quality-weighted Poisson solve (Ghiglia-Romero style) vs plain.

The earlier INTENSITY-weighted solve was unstable because it weighted the wrong thing. The phase-
unwrapping literature weights by a QUALITY / CONSISTENCY map instead: down-weight the wrapped-
gradient RHS where the local wrapped gradient is large (i.e. near residues / inconsistencies),
so noisy/singular gradients contribute less to the global solve. Here the weight is derived from
the phase's OWN wrapped-gradient magnitude (data-driven), not from the target intensity.
    w = 1 / (1 + (g/g0)^2),   g = local wrapped-gradient magnitude.
Projector alpha fixed. Compare plain vs quality-weighted, on a dark-heavy target.

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
input_tiff = "Cat_black.tif"; target_floor_rel = 1e-2; alpha = 0.5
g0 = 1.0   # wrapped-gradient scale for the quality weight [rad]
n = m = 512; nn = n + 2*(n//4); mm = m + 2*(m//4)
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "output_temp_quality"); os.makedirs(output_dir, exist_ok=True)

#%% Grid + target
bandlim_spe = cp.zeros((nn, mm), cp.float32); bandlim_spe[nn//4:3*nn//4, mm//4:3*mm//4] = 1.0
bandlim_in = cp.zeros((nn, mm), cp.float32); bandlim_in[(nn-n)//2:(nn+n)//2, (mm-m)//2:(mm+m)//2] = 1.0
bandlim_ou = 1.0 - bandlim_in
sr_r0, sr_r1 = (nn-n)//2, (nn+n)//2; sr_c0, sr_c1 = (mm-m)//2, (mm+m)//2
w = 0.26
ox, oy = cp.meshgrid(cp.linspace(-dh*mm/2, dh*mm/2, mm), cp.linspace(-dh*nn/2, dh*nn/2, nn))
incident = cp.exp(-((ox**2)+(oy**2))/w) * bandlim_spe
_ii = cp.arange(n).reshape(n,1); _jj = cp.arange(m).reshape(1,m)
_denom = 2*cp.cos(2*cp.pi*_ii/n) + 2*cp.cos(2*cp.pi*_jj/m) - 4; _denom[0,0] = 1.0
p = os.path.join(ALT_PROJ_DIR, input_tiff)
F1 = np.array(Image.open(p)); F1 = F1[...,0] if F1.ndim==3 else F1
F1 = cv2.resize(F1.astype(np.float32), (m,n), interpolation=cv2.INTER_AREA)
F1 = np.maximum(F1, target_floor_rel*(np.max(F1)+1e-12))
E = float(np.sum(F1)); El = 0.5*E
F = np.pad(np.abs(np.sqrt(F1)), ((n//4,n//4),(m//4,m//4)), mode="constant")
F_gpu = cp.asarray(F); E_gpu = cp.asarray(E); El_gpu = cp.asarray(El); F1_gpu = cp.asarray(F1)

def irrotational_phase(pha, quality=False):
    dx = cp.zeros((n,m)); dy = cp.zeros((n,m))
    dx[:,:-1] = cp.mod(pha[:,1:]-pha[:,:-1]+cp.pi, 2*cp.pi)-cp.pi
    dy[:-1,:] = cp.mod(pha[1:,:]-pha[:-1,:]+cp.pi, 2*cp.pi)-cp.pi
    if quality:
        g = cp.sqrt(dx**2 + dy**2)                # local wrapped-gradient magnitude
        wq = 1.0 / (1.0 + (g/g0)**2)              # down-weight inconsistent (large-gradient) pixels
        dx = dx*wq; dy = dy*wq
    rho = cp.zeros((n,m)); rho[:,0]=dx[:,0]; rho[:,1:]=dx[:,1:]-dx[:,:-1]
    rho[0,:]+=dy[0,:]; rho[1:,:]+=dy[1:,:]-dy[:-1,:]
    rh = cp.fft.fft2(rho); rh[0,0]=0.0
    return cp.real(cp.fft.ifft2(rh/_denom))

def final_reconstruction(E2_k):
    hologram = incident * cp.exp(1j*cp.angle(E2_k))
    Rec = cp.fft.fftshift(cp.fft.ifft2(cp.fft.fftshift(hologram)))[sr_r0:sr_r1, sr_c0:sr_c1]
    I = cp.abs(Rec)**2; return cp.asnumpy(E_gpu*I/(cp.sum(I)+1e-12))

def rmse_roughness(a, w0):
    t = a[w0:]; return float(np.mean(np.abs(np.diff(t)))) if len(t)>1 else 0.0

def run(quality):
    cp.random.seed(seed)
    phi = cp.exp(1j*2*cp.pi*cp.random.rand(nn,mm)); amp = cp.random.rand(nn,mm)
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
        psi = irrotational_phase(pc, quality=quality); psi = psi + cp.angle(cp.sum(cp.exp(1j*(pc-psi))))
        pn = pha.copy(); pn[sr_r0:sr_r1, sr_c0:sr_c1] = cp.angle((1-alpha)*cp.exp(1j*pc) + alpha*cp.exp(1j*psi))
        phi = cp.exp(1j*pn)
    return dict(RMSE=RMSE, NUM=NUM, I_final=final_reconstruction(E2_k))

#%% Run
print(f"Quality-weighted solve on {input_tiff} (floor={target_floor_rel}, alpha={alpha})", flush=True)
runs = {"plain": run(False), "quality_weighted": run(True)}
for name, r in runs.items():
    print(f"  {name:16s} RMSE={r['RMSE'][-1]:8.4f}  roughness={rmse_roughness(r['RMSE'], loop//2):.4f}  vort={r['NUM'][-1]:5d}", flush=True)

fig = plt.figure()
for name, r in runs.items():
    plt.plot(r["RMSE"][1:], label=name)
plt.xlabel("Iteration"); plt.ylabel("RMSE (SR)"); plt.title(f"Quality-weighted solve — {os.path.splitext(input_tiff)[0]}")
plt.legend(fontsize=8)
fig.savefig(os.path.join(output_dir, "rmse.png"), dpi=150, bbox_inches="tight"); plt.show()
print(f"Saved to {output_dir}")
# %%
