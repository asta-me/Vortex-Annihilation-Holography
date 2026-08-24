#%%
"""
THE VORTEX-FREE INITIALIZER, EXPLAINED STEP BY STEP.

Idea in one line: build a phase from a SMOOTH REAL field (not from the angle of a complex field),
and scale it so no neighbouring-pixel phase step ever reaches pi. A single-valued real function has
zero circulation on every loop, and staying below pi kills aliasing -> ZERO vortices, provably.

Only TWO numbers are chosen by hand, and both are physical (nothing else is hard-coded):
  * corr_len_px : the correlation length of the field in pixels = "how smooth" the phase is.
                  Larger -> smoother phase, tighter far-field spot. The ONE design knob.
  * grad_cap    : a Nyquist SAFETY factor in (0,1). We scale the phase so the largest per-pixel
                  step equals grad_cap*pi < pi. It is not tuning: it must be < 1, full stop.
Everything else (the Fourier filter width, the amplitude) is DERIVED from these + the grid.

Env: vortex (conda), GPU (CuPy).
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cupy as cp
import cv2
from PIL import Image

ALT_PROJ_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "1_Alternative_projection")
sys.path.insert(0, ALT_PROJ_DIR)
from function_vortex_detection_accegpu import function_vortex_detection_accegpu

#%% ---- Geometry (fix the hologram size and the SR fraction; the rest is derived) ----
dh = 0.00374                                   # [mm] pixel pitch (only used by the vortex detector)
HOLOGRAM_SIZE = 384; WORK_SIZE = 2 * HOLOGRAM_SIZE; SR_FRACTION = 2 / 3
SR_SIZE = int(round(SR_FRACTION * WORK_SIZE)); SR_SIZE -= SR_SIZE % 2
n = m = SR_SIZE                                 # signal-region (image) side [px]
nn = mm = WORK_SIZE                             # work grid side [px]
pad_each = (WORK_SIZE - SR_SIZE) // 2; ap0 = (WORK_SIZE - HOLOGRAM_SIZE) // 2
sr_r0, sr_r1 = pad_each, pad_each + n           # signal-region rows inside the work grid
sr_c0, sr_c1 = pad_each, pad_each + m           # signal-region cols inside the work grid

# SLM aperture = central HOLOGRAM_SIZE of the work grid (Chen 2x oversampling). Used only to VISUALISE the fill.
ap_r0, ap_r1, ap_c0, ap_c1 = ap0, ap0 + HOLOGRAM_SIZE, ap0, ap0 + HOLOGRAM_SIZE
bandlim_spe = cp.zeros((nn, mm), cp.float32); bandlim_spe[ap_r0:ap_r1, ap_c0:ap_c1] = 1.0
w = 0.26
ox, oy = cp.meshgrid(cp.linspace(-dh*mm/2, dh*mm/2, mm), cp.linspace(-dh*nn/2, dh*nn/2, nn))
incident = cp.exp(-((ox**2)+(oy**2))/w) * bandlim_spe     # gaussian beam clipped by the aperture

#%% ---- The TWO design numbers (both physical) ----
corr_len_px = 20.0     # [px] correlation length of the phase = "how smooth". Empirical RMSE optimum (plateau ~16-31).
grad_cap    = 0.9      # Nyquist SAFETY in (0,1): largest per-pixel phase step = grad_cap*pi < pi.

#%% ============================================================================================
#   THE METHOD, LINE BY LINE. Returns the phase AND every intermediate stage (for plotting).
#  ============================================================================================
def vortexfree_phase(seed):
    # 1) WHITE REAL NOISE. One independent Gaussian value per pixel. REAL, not complex: this is the
    #    whole trick -- we will never take the angle of a complex field (that is what makes vortices).
    cp.random.seed(seed)
    noise = cp.random.randn(nn, mm)

    # 2) LOW-PASS FILTER = impose a correlation length. Keep only low spatial frequencies so the
    #    field becomes smooth. The Gaussian width in frequency, ks, is DERIVED from the desired
    #    real-space correlation length (no hand-picked Fourier units): a real-space Gaussian of
    #    sigma = corr_len_px pairs with a frequency Gaussian of sigma_k = N / (2*pi*corr_len_px).
    ks = nn / (2*np.pi*corr_len_px)
    kx = cp.fft.fftfreq(mm).reshape(1, mm) * mm          # integer frequency index along x
    ky = cp.fft.fftfreq(nn).reshape(nn, 1) * nn          # integer frequency index along y
    lowpass = cp.exp(-(kx**2 + ky**2) / (2*ks**2))       # Gaussian low-pass mask

    # 3) BACK TO REAL SPACE. Filter the spectrum and inverse-transform -> a SMOOTH REAL field.
    s = cp.real(cp.fft.ifft2(cp.fft.fft2(noise) * lowpass))

    # 4) NYQUIST SCALING. Find the largest step between neighbouring pixels, then scale the WHOLE
    #    field so that this largest step becomes exactly grad_cap*pi. Being < pi, every wrapped
    #    gradient equals the true gradient -> NO aliasing -> no spurious 2*pi jump can appear.
    #    (No mean/std normalisation is needed: only this peak-step rescale matters -- dividing by
    #     std first would just cancel here, and the mean is only an irrelevant global phase.)
    max_step = cp.maximum(cp.abs(cp.diff(s, axis=1)).max(), cp.abs(cp.diff(s, axis=0)).max())
    a = grad_cap * np.pi / (float(max_step.get()) + 1e-12)

    # 5) USE THE REAL FIELD DIRECTLY AS THE PHASE. phi = a*s is a single-valued real function, so its
    #    circulation round any loop is exactly 0 -> ZERO vortices.
    phi = a * s
    return cp.exp(1j*phi), dict(noise=noise, lowpass=lowpass, s=s, a=a, phi=phi, ks=ks)

def vortex_count(field):
    pc = cp.angle(field)[sr_r0:sr_r1, sr_c0:sr_c1]
    po, ne = function_vortex_detection_accegpu(pc, dh, use_cupy=True); return int(po+ne)

#%% ---- Build one example and read off the guarantees ----
seed = 1
field, st = vortexfree_phase(seed)
nv = vortex_count(field)
max_step_px = float(cp.maximum(cp.abs(cp.diff(st["phi"], axis=1)).max(),
                               cp.abs(cp.diff(st["phi"], axis=0)).max()).get())
print(f"corr_len_px={corr_len_px:g}, grad_cap={grad_cap:g}  ->  Fourier width ks={st['ks']:.2f}, "
      f"amplitude a={st['a']:.3f}", flush=True)
print(f"max per-pixel phase step = {max_step_px:.3f} rad  (= grad_cap*pi = {grad_cap*np.pi:.3f}); "
      f"vortices in SR = {nv}", flush=True)

# Far-field the phase would produce (to see it fills the signal region, not just a dot)
farfield = cp.abs(cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(field*bandlim_spe))))**2

#%% ---- Visualise every stage ----
def crop(x): return cp.asnumpy(x)[sr_r0:sr_r1, sr_c0:sr_c1]

fig, ax = plt.subplots(2, 3, figsize=(13, 8.6))

ax[0,0].imshow(crop(st["noise"]), cmap="gray")
ax[0,0].set_title("1) white REAL noise\n(randn, one value/pixel)", fontsize=10); ax[0,0].axis("off")

# low-pass mask shown over the (shifted) spectrum magnitude of the noise
spec = cp.asnumpy(cp.abs(cp.fft.fftshift(cp.fft.fft2(st["noise"]))))
ax[0,1].imshow(np.log1p(spec), cmap="inferno")
lp_shift = cp.asnumpy(cp.fft.fftshift(st["lowpass"]))
ax[0,1].contour(lp_shift, levels=[0.5], colors="cyan", linewidths=1.5)
ax[0,1].set_title(f"2) Gaussian low-pass (cyan)\nks={st['ks']:.1f} from corr_len={corr_len_px:g}px", fontsize=10); ax[0,1].axis("off")

ax[0,2].imshow(crop(st["s"]), cmap="gray")
ax[0,2].set_title("3) smooth REAL field s\n(single-valued)", fontsize=10); ax[0,2].axis("off")

# gradient histogram with the pi wall
gx = cp.asnumpy(cp.abs(cp.diff(st["phi"], axis=1))).ravel()
gy = cp.asnumpy(cp.abs(cp.diff(st["phi"], axis=0))).ravel()
ax[1,0].hist(np.concatenate([gx, gy]), bins=120, color="tab:blue")
ax[1,0].axvline(np.pi, color="red", lw=2, label=r"$\pi$ (aliasing wall)")
ax[1,0].axvline(grad_cap*np.pi, color="green", lw=2, ls="--", label=r"$grad\_cap\cdot\pi$ (our max)")
ax[1,0].set_title("4) per-pixel phase steps\nall below pi -> no aliasing", fontsize=10)
ax[1,0].set_xlabel("|step| [rad]"); ax[1,0].legend(fontsize=8)

ax[1,1].imshow(crop(st["phi"]), cmap="twilight")
ax[1,1].set_title(f"5) phase phi = a*s\nvortices in SR = {nv}", fontsize=10); ax[1,1].axis("off")

ff = cp.asnumpy(farfield); vmax = np.percentile(ff, 99.5)
ax[1,2].imshow(ff, cmap="inferno", vmax=vmax)
ax[1,2].add_patch(Rectangle((sr_c0, sr_r0), m, n, fill=False, ec="cyan", lw=1.5))  # signal region
ax[1,2].set_title("far-field |FT|^2\n(fills the signal region, cyan)", fontsize=10); ax[1,2].axis("off")

plt.tight_layout()
plt.show()
# %%
