#%%
"""
TEMP: vortex-FREE RANDOM initialization (target-agnostic) vs random vs quadratic.

The quadratic (lens) init nearly halves RMSE, but its optimum is likely TARGET-DEPENDENT (tuned for
square/uniform targets like marmo). We want the SAME benefit -- a smooth, vortex-free start that lands
in a good basin -- WITHOUT a target-specific tuning, and with SEED DIVERSITY so we can time-multiplex
several different holograms of the same target (speckle averaging).

Key idea: a RANDOM but VORTEX-FREE start. Two ways:
  - "irrot_random": take a random phase and project it ONCE with the irrotational Poisson solver
     -> a smooth scalar-potential phase, curl-free BY CONSTRUCTION, but seed-diverse.
  - "lowfreq":      a low-frequency Gaussian random field used as phase (smooth, few/no vortices).
Compared against: "random" (baseline) and "quadratic" (c=1500, the tuned lens).

We run the random-based inits with several seeds and report:
  - final RMSE, initial vortex count, final vortex count,
  - pairwise DIVERSITY of the final SLM holograms (low complex-correlation = good for multiplexing).

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
init_seeds = [1, 2, 3]           # for the random-based (diverse) inits
lowfreq_ks = 6.0; lowfreq_amp = 3.0
n = m = 512; nn = n + 2*(n//4); mm = m + 2*(m//4)
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "output_temp_vfinit"); os.makedirs(output_dir, exist_ok=True)

#%% Grid + target
bandlim_spe = cp.zeros((nn, mm), cp.float32); bandlim_spe[nn//4:3*nn//4, mm//4:3*mm//4] = 1.0
bandlim_in = cp.zeros((nn, mm), cp.float32); bandlim_in[(nn-n)//2:(nn+n)//2, (mm-m)//2:(mm+m)//2] = 1.0
bandlim_ou = 1.0 - bandlim_in
sr_r0, sr_r1 = (nn-n)//2, (nn+n)//2; sr_c0, sr_c1 = (mm-m)//2, (mm+m)//2
w = 0.26
ox, oy = cp.meshgrid(cp.linspace(-dh*mm/2, dh*mm/2, mm), cp.linspace(-dh*nn/2, dh*nn/2, nn))
incident = cp.exp(-((ox**2)+(oy**2))/w) * bandlim_spe
UU = cp.broadcast_to(cp.linspace(-0.5,0.5,mm).reshape(1,mm), (nn,mm))
VV = cp.broadcast_to(cp.linspace(-0.5,0.5,nn).reshape(nn,1), (nn,mm)); RR2 = UU**2+VV**2
def make_denom(nr, nc):
    ii = cp.arange(nr).reshape(nr,1); jj = cp.arange(nc).reshape(1,nc)
    d = 2*cp.cos(2*cp.pi*ii/nr) + 2*cp.cos(2*cp.pi*jj/nc) - 4; d[0,0]=1.0; return d
_denom = make_denom(n, m); _denomF = make_denom(nn, mm)
p = os.path.join(ALT_PROJ_DIR, input_tiff)
F1 = np.array(Image.open(p)); F1 = F1[...,0] if F1.ndim==3 else F1
F1 = cv2.resize(F1.astype(np.float32), (m,n), interpolation=cv2.INTER_AREA)
F1 = np.maximum(F1, target_floor_rel*(np.max(F1)+1e-12))
E = float(np.sum(F1)); El = 0.5*E
F = np.pad(np.abs(np.sqrt(F1)), ((n//4,n//4),(m//4,m//4)), mode="constant")
F_gpu = cp.asarray(F); E_gpu = cp.asarray(E); El_gpu = cp.asarray(El); F1_gpu = cp.asarray(F1)

def irrotational_phase(pha, denom):
    nr, nc = pha.shape
    dx = cp.zeros((nr,nc)); dy = cp.zeros((nr,nc))
    dx[:,:-1] = cp.mod(pha[:,1:]-pha[:,:-1]+cp.pi, 2*cp.pi)-cp.pi
    dy[:-1,:] = cp.mod(pha[1:,:]-pha[:-1,:]+cp.pi, 2*cp.pi)-cp.pi
    rho = cp.zeros((nr,nc)); rho[:,0]=dx[:,0]; rho[:,1:]=dx[:,1:]-dx[:,:-1]
    rho[0,:]+=dy[0,:]; rho[1:,:]+=dy[1:,:]-dy[:-1,:]
    rh = cp.fft.fft2(rho); rh[0,0]=0.0
    return cp.real(cp.fft.ifft2(rh/denom))

def project_phase(pha):
    pc = pha[sr_r0:sr_r1, sr_c0:sr_c1]
    psi = irrotational_phase(pc, _denom); psi = psi + cp.angle(cp.sum(cp.exp(1j*(pc-psi))))
    pn = pha.copy(); pn[sr_r0:sr_r1, sr_c0:sr_c1] = cp.angle((1-alpha)*cp.exp(1j*pc) + alpha*cp.exp(1j*psi))
    return pn

def final_reconstruction(E2_k):
    An = cp.angle(E2_k)
    hologram = incident * cp.exp(1j*An)
    Rec = cp.fft.fftshift(cp.fft.ifft2(cp.fft.fftshift(hologram)))[sr_r0:sr_r1, sr_c0:sr_c1]
    I = cp.abs(Rec)**2
    return cp.asnumpy(E_gpu*I/(cp.sum(I)+1e-12)), An

def make_init(kind, s=0):
    if kind == "random":
        cp.random.seed(s); return cp.exp(1j*2*cp.pi*cp.random.rand(nn,mm))
    if kind == "quadratic":
        return cp.exp(1j*quad_c*RR2)
    if kind == "lowfreq":
        cp.random.seed(s); noise = cp.random.randn(nn,mm)
        ky = (cp.fft.fftfreq(nn)*nn).reshape(nn,1); kx = (cp.fft.fftfreq(mm)*mm).reshape(1,mm)
        lp = cp.exp(-(kx**2+ky**2)/(2*lowfreq_ks**2))
        sm = cp.real(cp.fft.ifft2(cp.fft.fft2(noise)*lp)); sm = (sm-sm.mean())/(sm.std()+1e-12)
        return cp.exp(1j*lowfreq_amp*sm)
    if kind == "irrot_random":
        cp.random.seed(s); praw = 2*cp.pi*cp.random.rand(nn,mm)
        psi = irrotational_phase(praw, _denomF)          # project full grid -> curl-free potential
        return cp.exp(1j*psi)
    raise ValueError(kind)

def init_vortices(phi):
    pc = cp.angle(phi)[sr_r0:sr_r1, sr_c0:sr_c1]
    po, ne = function_vortex_detection_accegpu(pc, dh, use_cupy=True); return po+ne

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
    I_final, An = final_reconstruction(E2_k)
    return dict(RMSE=RMSE, NUM=NUM, I_final=I_final, An=An)

#%% Run
print(f"Vortex-free-init study on {input_tiff} (alpha={alpha}, loop={loop})", flush=True)
results = {}
# baselines
for kind in ("random", "quadratic"):
    phi0 = make_init(kind, s=1)
    r = run(phi0); r["init_vort"] = int(init_vortices(phi0)); results[kind] = r
    print(f"  {kind:16s} RMSE={r['RMSE'][-1]:8.4f}  init_vort={r['init_vort']:6d}  final_vort={r['NUM'][-1]:5d}", flush=True)
# random-based vortex-free inits over several seeds
for kind in ("lowfreq", "irrot_random"):
    for s in init_seeds:
        phi0 = make_init(kind, s=s)
        r = run(phi0); r["init_vort"] = int(init_vortices(phi0)); results[f"{kind}_s{s}"] = r
        print(f"  {kind+'_s'+str(s):16s} RMSE={r['RMSE'][-1]:8.4f}  init_vort={r['init_vort']:6d}  final_vort={r['NUM'][-1]:5d}", flush=True)

#%% Diversity of the irrot_random holograms (for time multiplexing)
def complex_corr(A, B):
    a = cp.exp(1j*cp.asarray(A)); b = cp.exp(1j*cp.asarray(B))
    return float(cp.abs(cp.mean(a*cp.conj(b))).get())
print("\nHologram diversity (|complex corr| of SLM phase; 0=diverse, 1=identical):")
ir = [f"irrot_random_s{s}" for s in init_seeds]
for i in range(len(ir)):
    for j in range(i+1, len(ir)):
        print(f"  {ir[i]} vs {ir[j]}: {complex_corr(results[ir[i]]['An'], results[ir[j]]['An']):.3f}")

# Time-multiplex (incoherent average) of the 3 irrot_random reconstructions
Imux = np.mean([results[k]["I_final"] for k in ir], axis=0)
rmse_mux = float(np.sqrt(np.mean((Imux/ (Imux.sum()+1e-12)*E - F1)**2)))
print(f"  time-multiplexed (avg of {len(ir)}) RMSE={rmse_mux:.4f}  (single ~{results[ir[0]]['RMSE'][-1]:.4f})")

#%% Plots
fig = plt.figure()
for k in ("random", "quadratic", "lowfreq_s1", "irrot_random_s1", "irrot_random_s2", "irrot_random_s3"):
    plt.plot(results[k]["RMSE"][1:], label=k)
plt.xlabel("Iteration"); plt.ylabel("RMSE (SR)"); plt.title(f"Vortex-free init vs random/quadratic — {os.path.splitext(input_tiff)[0]}")
plt.legend(fontsize=7)
fig.savefig(os.path.join(output_dir, "rmse.png"), dpi=150, bbox_inches="tight"); plt.show()

panels = [("target", F1), ("random", results["random"]["I_final"]), ("quadratic", results["quadratic"]["I_final"])] \
    + [(k, results[k]["I_final"]) for k in ir] + [("MUX avg", Imux)]
fig, ax = plt.subplots(1, len(panels), figsize=(3*len(panels), 4))
for a, (t, im) in zip(ax, panels):
    a.imshow(im, cmap="gray"); a.set_title(t, fontsize=8); a.axis("off")
plt.tight_layout(); fig.savefig(os.path.join(output_dir, "recon.png"), dpi=150, bbox_inches="tight"); plt.show()
print(f"Saved to {output_dir}")
# %%
