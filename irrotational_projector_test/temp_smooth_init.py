#%%
"""
TEMP: smooth phase initialization vs random.

Vortices are largely SEEDED by the random initial SLM phase. Starting from a SMOOTH phase should
nucleate far fewer vortices from the start (prevention over cure), and may land in a better basin.
We compare initial phases: random (baseline), flat/zero, quadratic (lens), low-frequency random.
Projector ON in all. We report final RMSE, final vortices, and the EARLY vortex count (iter 5).

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
output_dir = os.path.join(script_dir, "output_temp_init"); os.makedirs(output_dir, exist_ok=True)

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
    pn = pha.copy(); pn[sr_r0:sr_r1, sr_c0:sr_c1] = cp.angle((1-alpha)*cp.exp(1j*pc) + alpha*cp.exp(1j*psi))
    return pn

def final_reconstruction(E2_k):
    hologram = incident * cp.exp(1j*cp.angle(E2_k))
    Rec = cp.fft.fftshift(cp.fft.ifft2(cp.fft.fftshift(hologram)))[sr_r0:sr_r1, sr_c0:sr_c1]
    I = cp.abs(Rec)**2; return cp.asnumpy(E_gpu*I/(cp.sum(I)+1e-12))

def make_init(kind):
    cp.random.seed(seed)
    if kind == "random":
        return cp.exp(1j*2*cp.pi*cp.random.rand(nn,mm))
    if kind == "flat":
        return cp.ones((nn,mm), dtype=cp.complex128)
    if kind == "quadratic":
        c = 1500.0
        return cp.exp(1j*c*((ox/ (dh*mm))**2 + (oy/(dh*nn))**2))
    if kind == "lowfreq":
        r = cp.random.rand(nn,mm)
        R = cp.fft.fftshift(cp.fft.fft2(r))
        mask = cp.zeros((nn,mm)); mask[nn//2-8:nn//2+8, mm//2-8:mm//2+8] = 1.0
        low = cp.real(cp.fft.ifft2(cp.fft.ifftshift(R*mask)))
        low = (low - low.min())/(low.max()-low.min()+1e-12)
        return cp.exp(1j*2*cp.pi*low)
    raise ValueError(kind)

def run_init(kind):
    phi = make_init(kind); cp.random.seed(seed); amp = cp.random.rand(nn,mm)
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

#%% Run
print(f"Smooth-init study on {input_tiff} (alpha={alpha}, loop={loop})", flush=True)
runs = {}
for kind in ("random", "flat", "quadratic", "lowfreq"):
    runs[kind] = run_init(kind)
    r = runs[kind]
    print(f"  {kind:10s} RMSE={r['RMSE'][-1]:8.4f}  vort(final)={r['NUM'][-1]:5d}  vort(iter5)={r['NUM'][5]:6d}", flush=True)

fig = plt.figure()
for name, r in runs.items():
    plt.plot(r["NUM"][1:], label=name)
plt.xlabel("Iteration"); plt.ylabel("Vortex count (SR)"); plt.yscale("log"); plt.title(f"Init vs vortices — {os.path.splitext(input_tiff)[0]}")
plt.legend(fontsize=8)
fig.savefig(os.path.join(output_dir, "vortex.png"), dpi=150, bbox_inches="tight"); plt.show()
print(f"Saved to {output_dir}")
# %%
