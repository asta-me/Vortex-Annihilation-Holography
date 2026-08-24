#%%
"""
Weighted / masked irrotational PROJECTOR (spatially varying regularization).

Two DIFFERENT intensity-driven weightings (often confused) are implemented and compared:

  (a) WEIGHTED POISSON SOLVE  --  w(x,y) proportional to target intensity, applied to the
      wrapped-gradient contributions BEFORE the FFT (dx*=w, dy*=w). This changes psi itself:
      low-intensity (dark, undefined-phase) pixels no longer inject noisy vorticity into the
      single GLOBAL Poisson solve.

  (b) MASKED APPLICATION  --  alpha(x,y) = alpha * mask(x,y), mask proportional to intensity.
      This changes HOW MUCH of psi is blended in per pixel (correct less where it is dark),
      NOT psi itself:  field = (1-alpha(x,y))*exp(i*pha) + alpha(x,y)*exp(i*psi).

(a) attacks the noise at its source inside the solve; (b) limits the correction in dark regions.
They are complementary. We compare: off, plain, weighted_solve, masked_alpha, weighted+masked,
and the paper method, on a dark-heavy target (where these matter).

Both weights are smooth sigmoids of the per-image-normalized target intensity.

STATUS: SET ASIDE (results not stable / satisfactory yet)
---------------------------------------------------------
On Cat_black (floor 0.01, alpha 0.5) the weightings do NOT give a clean, robust win:
    plain               RMSE 10.57  roughness 0.244
    weighted_solve (a)  RMSE 11.05  roughness 0.012   (mostly STABILIZES; RMSE ~unchanged)
    masked_alpha  (b)   RMSE 14.83  roughness 0.401   (WORSE on its own)
    weighted+masked     RMSE  7.19  roughness 0.028   (best projector, ~matches paper 6.66 and far
                                                       smoother, BUT sensitive to threshold/softness)
Takeaway: (a) mainly reduces the RMSE-curve jitter (roughness), not RMSE; (b) ALONE is harmful;
only (a)+(b) together approach the paper, and the result is tuning-sensitive -> NOT robust.
Raising the floor (see the floor-sweep script) is a simpler, stronger, more stable lever.
This weighting approach is kept for reference but SET ASIDE for now.

Env: vortex (conda), GPU (CuPy). PIL + cv2 (no skimage).
"""

#%% ---------- Imports ----------
import time
import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
import cupy as cp
import cv2
from PIL import Image

ALT_PROJ_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "1_Alternative_projection")
sys.path.insert(0, ALT_PROJ_DIR)
from function_vortex_detection_accegpu import function_vortex_detection_accegpu
from function_vortex_elimination_accegpu import function_vortex_elimination_accegpu

#%% ---------- Config ----------
lamda = 532e-6
dh = 0.00374
loop = 300
seed = 42

input_tiff = "marmo.tif"              # <-- CHANGE TARGET HERE (dark-heavy shows the effect)
target_floor_rel = 1e-2                    # modest floor: plain projector still a bit noisy here
alpha = 0.5                                # blend strength
paper_x = 100                             # paper reference period

# Smooth sigmoid weight/mask (relative to per-image max intensity)
W_THRESHOLD, W_SOFTNESS = 0.10, 0.05      # weighted Poisson solve  w(x,y)
M_THRESHOLD, M_SOFTNESS = 0.20, 0.05      # masked application      mask(x,y)

#%% ---------- Fixed grid ----------
n = m = 512
nn = n + 2 * (n // 4)
mm = m + 2 * (m // 4)
bandlim_spe = cp.zeros((nn, mm), dtype=cp.float32)
bandlim_spe[nn // 4:3 * nn // 4, mm // 4:3 * mm // 4] = 1.0
bandlim_in = cp.zeros((nn, mm), dtype=cp.float32)
bandlim_in[(nn - n) // 2:(nn + n) // 2, (mm - m) // 2:(mm + m) // 2] = 1.0
bandlim_ou = 1.0 - bandlim_in
sr_r0, sr_r1 = (nn - n) // 2, (nn + n) // 2
sr_c0, sr_c1 = (mm - m) // 2, (mm + m) // 2
w = 0.26
ox, oy = cp.meshgrid(cp.linspace(-dh * mm / 2, dh * mm / 2, mm), cp.linspace(-dh * nn / 2, dh * nn / 2, nn))
Gaussian = cp.exp(-((ox ** 2) + (oy ** 2)) / w)
incident = Gaussian * bandlim_spe
_ii = cp.arange(n).reshape(n, 1)
_jj = cp.arange(m).reshape(1, m)
_denom = 2 * cp.cos(2 * cp.pi * _ii / n) + 2 * cp.cos(2 * cp.pi * _jj / m) - 4
_denom[0, 0] = 1.0

script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "output_test_weighted")
os.makedirs(output_dir, exist_ok=True)

#%% ---------- Target + weight/mask maps ----------
input_tiff_path = os.path.join(ALT_PROJ_DIR, input_tiff)
if not os.path.isfile(input_tiff_path):
    raise FileNotFoundError(f"TIFF file not found: {input_tiff_path}")
_F1 = np.array(Image.open(input_tiff_path))
if _F1.ndim == 3:
    _F1 = _F1[..., 0]
_F1 = cv2.resize(_F1.astype(np.float32), (m, n), interpolation=cv2.INTER_AREA)
F1 = np.maximum(_F1, target_floor_rel * (np.max(_F1) + 1e-12))
E = float(np.sum(F1)); El = 0.5 * E
F = np.pad(np.abs(np.sqrt(F1)), ((n // 4, n // 4), (m // 4, m // 4)), mode="constant")
F_gpu = cp.asarray(F); E_gpu = cp.asarray(E); El_gpu = cp.asarray(El); F1_gpu = cp.asarray(F1)
target_name = os.path.splitext(input_tiff)[0]

F1n = cp.asarray((F1 / (np.max(F1) + 1e-12)).astype(np.float32))
poisson_weight = 1.0 / (1.0 + cp.exp(-(F1n - W_THRESHOLD) / W_SOFTNESS))   # w(x,y) in [0,1]
application_mask = 1.0 / (1.0 + cp.exp(-(F1n - M_THRESHOLD) / M_SOFTNESS))  # mask(x,y) in [0,1]

#%% ---------- Core ----------
def irrotational_phase(pha, weight=None):
    """Least-squares irrotational phase; if `weight` is given it down-weights the wrapped-gradient
    RHS (weighted Poisson SOLVE). weight=None reproduces the plain solve exactly."""
    dx = cp.zeros((n, m)); dy = cp.zeros((n, m))
    dx[:, :-1] = cp.mod(pha[:, 1:] - pha[:, :-1] + cp.pi, 2 * cp.pi) - cp.pi
    dy[:-1, :] = cp.mod(pha[1:, :] - pha[:-1, :] + cp.pi, 2 * cp.pi) - cp.pi
    if weight is not None:
        dx = dx * weight
        dy = dy * weight
    rho = cp.zeros((n, m))
    rho[:, 0] = dx[:, 0]; rho[:, 1:] = dx[:, 1:] - dx[:, :-1]
    rho[0, :] += dy[0, :]; rho[1:, :] += dy[1:, :] - dy[:-1, :]
    rho_hat = cp.fft.fft2(rho); rho_hat[0, 0] = 0.0
    return cp.real(cp.fft.ifft2(rho_hat / _denom))


def final_reconstruction(E2_k):
    An = cp.angle(E2_k)
    hologram = incident * cp.exp(1j * An)
    Rec = cp.fft.fftshift(cp.fft.ifft2(cp.fft.fftshift(hologram)))
    Rec_sr = Rec[sr_r0:sr_r1, sr_c0:sr_c1]
    I_final = cp.abs(Rec_sr) ** 2
    I_final = E_gpu * I_final / (cp.sum(I_final) + 1e-12)
    return cp.asnumpy(I_final)


def rmse_roughness(rmse_array, warmup):
    tail = rmse_array[warmup:]
    return float(np.mean(np.abs(np.diff(tail)))) if len(tail) > 1 else 0.0


def run_projector(weighted, masked):
    """Projector every iteration; weighted -> weighted solve, masked -> alpha(x,y) mask."""
    weight = poisson_weight if weighted else None
    mask = application_mask if masked else None
    cp.random.seed(seed)
    phi = cp.exp(1j * 2 * cp.pi * cp.random.rand(nn, mm))
    amp = cp.random.rand(nn, mm)
    RMSE = np.zeros(loop); NUM = np.zeros(loop, dtype=int)
    for i in range(1, loop):
        amp = bandlim_in * F_gpu + bandlim_ou * amp
        E1 = amp * phi
        E2 = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(E1)))
        E2_ave = cp.sqrt((E_gpu + El_gpu) * incident ** 2 / cp.sum(incident ** 2))
        E2_k = E2_ave * cp.exp(1j * cp.angle(E2))
        es = cp.fft.fftshift(cp.fft.ifft2(cp.fft.fftshift(E2_k)))
        amp = cp.abs(es)
        amp_in = bandlim_in * amp; amp_ou = bandlim_ou * amp
        amp = cp.sqrt(E_gpu) * (amp_in / (cp.sqrt(cp.sum(amp_in ** 2)) + 1e-12)) \
            + cp.sqrt(El_gpu) * (amp_ou / (cp.sqrt(cp.sum(amp_ou ** 2)) + 1e-12))
        I = amp[sr_r0:sr_r1, sr_c0:sr_c1] ** 2
        I = E_gpu * I / (cp.sum(I) + 1e-12)
        RMSE[i] = float(cp.sqrt(cp.mean((I - F1_gpu) ** 2)).get())
        pha = cp.angle(es); pha_crop = pha[sr_r0:sr_r1, sr_c0:sr_c1]
        po, ne = function_vortex_detection_accegpu(pha_crop, dh, use_cupy=True)
        NUM[i] = po + ne
        if alpha > 0.0:
            psi = irrotational_phase(pha_crop, weight=weight)
            psi = psi + cp.angle(cp.sum(cp.exp(1j * (pha_crop - psi))))
            eff_alpha = alpha * mask if mask is not None else alpha
            field = (1 - eff_alpha) * cp.exp(1j * pha_crop) + eff_alpha * cp.exp(1j * psi)
            pha_new = pha.copy(); pha_new[sr_r0:sr_r1, sr_c0:sr_c1] = cp.angle(field)
            phi = cp.exp(1j * pha_new)
        else:
            phi = cp.exp(1j * pha)
    return dict(RMSE=RMSE, NUM=NUM, I_final=final_reconstruction(E2_k))


def run_off():
    cp.random.seed(seed)
    phi = cp.exp(1j * 2 * cp.pi * cp.random.rand(nn, mm))
    amp = cp.random.rand(nn, mm)
    RMSE = np.zeros(loop); NUM = np.zeros(loop, dtype=int)
    for i in range(1, loop):
        amp = bandlim_in * F_gpu + bandlim_ou * amp
        E1 = amp * phi
        E2 = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(E1)))
        E2_ave = cp.sqrt((E_gpu + El_gpu) * incident ** 2 / cp.sum(incident ** 2))
        E2_k = E2_ave * cp.exp(1j * cp.angle(E2))
        es = cp.fft.fftshift(cp.fft.ifft2(cp.fft.fftshift(E2_k)))
        amp = cp.abs(es)
        amp_in = bandlim_in * amp; amp_ou = bandlim_ou * amp
        amp = cp.sqrt(E_gpu) * (amp_in / (cp.sqrt(cp.sum(amp_in ** 2)) + 1e-12)) \
            + cp.sqrt(El_gpu) * (amp_ou / (cp.sqrt(cp.sum(amp_ou ** 2)) + 1e-12))
        I = amp[sr_r0:sr_r1, sr_c0:sr_c1] ** 2
        I = E_gpu * I / (cp.sum(I) + 1e-12)
        RMSE[i] = float(cp.sqrt(cp.mean((I - F1_gpu) ** 2)).get())
        pha = cp.angle(es); pha_crop = pha[sr_r0:sr_r1, sr_c0:sr_c1]
        po, ne = function_vortex_detection_accegpu(pha_crop, dh, use_cupy=True)
        NUM[i] = po + ne
        phi = cp.exp(1j * pha)
    return dict(RMSE=RMSE, NUM=NUM, I_final=final_reconstruction(E2_k))


def run_paper(x):
    cp.random.seed(seed)
    phi = cp.exp(1j * 2 * cp.pi * cp.random.rand(nn, mm))
    amp = cp.random.rand(nn, mm)
    n_cycles = max(2, round(loop / x)); total = n_cycles * x
    RMSE = np.zeros(total); NUM = np.zeros(total, dtype=int)
    for i in range(1, total):
        amp = bandlim_in * F_gpu + bandlim_ou * amp
        E1 = amp * phi
        E2 = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(E1)))
        E2_ave = cp.sqrt((E_gpu + El_gpu) * incident ** 2 / cp.sum(incident ** 2))
        E2_k = E2_ave * cp.exp(1j * cp.angle(E2))
        es = cp.fft.fftshift(cp.fft.ifft2(cp.fft.fftshift(E2_k)))
        amp = cp.abs(es)
        amp_in = bandlim_in * amp; amp_ou = bandlim_ou * amp
        amp = cp.sqrt(E_gpu) * (amp_in / (cp.sqrt(cp.sum(amp_in ** 2)) + 1e-12)) \
            + cp.sqrt(El_gpu) * (amp_ou / (cp.sqrt(cp.sum(amp_ou ** 2)) + 1e-12))
        I = amp[sr_r0:sr_r1, sr_c0:sr_c1] ** 2
        I = E_gpu * I / (cp.sum(I) + 1e-12)
        RMSE[i] = float(cp.sqrt(cp.mean((I - F1_gpu) ** 2)).get())
        pha = cp.angle(es); pha_crop = pha[sr_r0:sr_r1, sr_c0:sr_c1]
        po, ne = function_vortex_detection_accegpu(pha_crop, dh, use_cupy=True)
        NUM[i] = po + ne
        if i % x == 0:
            vfree = function_vortex_elimination_accegpu(pha_crop, dh, use_cupy=True, gather_output=False)
            pha_new = pha.copy(); pha_new[sr_r0:sr_r1, sr_c0:sr_c1] = vfree
            phi = cp.exp(1j * pha_new)
        else:
            phi = cp.exp(1j * pha)
    return dict(RMSE=RMSE, NUM=NUM, I_final=final_reconstruction(E2_k))


#%% ---------- Run ----------
print(f"Weighted/masked projector on {input_tiff} (floor={target_floor_rel}, alpha={alpha})", flush=True)
runs = {
    "off":                run_off(),
    "plain":              run_projector(weighted=False, masked=False),
    "weighted_solve":     run_projector(weighted=True,  masked=False),
    "masked_alpha":       run_projector(weighted=False, masked=True),
    "weighted_and_masked": run_projector(weighted=True,  masked=True),
    "paper":              run_paper(paper_x),
}
for name, r in runs.items():
    print(f"  {name:20s} RMSE={r['RMSE'][-1]:8.4f}  roughness={rmse_roughness(r['RMSE'], loop//2):.4f}  "
          f"vort={r['NUM'][-1]:5d}", flush=True)

#%% ---------- Plots ----------
# 1) weight & mask maps
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(F1, cmap="gray"); axes[0].set_title("target (floored)"); axes[0].axis("off")
axes[1].imshow(cp.asnumpy(poisson_weight), cmap="viridis"); axes[1].set_title("w(x,y) — solve weight"); axes[1].axis("off")
axes[2].imshow(cp.asnumpy(application_mask), cmap="viridis"); axes[2].set_title("mask(x,y) — alpha mask"); axes[2].axis("off")
plt.tight_layout()
fig.savefig(os.path.join(output_dir, f"{target_name}_maps.png"), dpi=150, bbox_inches="tight")
plt.show()

# 2) RMSE vs iteration
fig = plt.figure()
for name in ("off", "plain", "weighted_solve", "masked_alpha", "weighted_and_masked"):
    plt.plot(runs[name]["RMSE"][1:], label=name)
plt.plot(runs["paper"]["RMSE"][1:], "k--", label="paper")
plt.xlabel("Iteration"); plt.ylabel("RMSE (SR)")
plt.title(f"RMSE vs iteration — {target_name} (floor={target_floor_rel})")
plt.legend(fontsize=8)
fig.savefig(os.path.join(output_dir, f"{target_name}_rmse.png"), dpi=150, bbox_inches="tight")
plt.show()

# 3) bars: settled RMSE + roughness
names = list(runs.keys())
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7))
ax1.bar(range(len(names)), [runs[nm]["RMSE"][-1] for nm in names]); ax1.set_ylabel("Final RMSE")
ax1.set_xticks(range(len(names))); ax1.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
ax1.set_title(f"Final RMSE — {target_name}")
ax2.bar(range(len(names)), [rmse_roughness(runs[nm]["RMSE"], loop // 2) for nm in names], color="C1")
ax2.set_ylabel("RMSE roughness"); ax2.set_yscale("log")
ax2.set_xticks(range(len(names))); ax2.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
ax2.set_title("RMSE roughness")
plt.tight_layout()
fig.savefig(os.path.join(output_dir, f"{target_name}_bars.png"), dpi=150, bbox_inches="tight")
plt.show()

# 4) reconstructions
panels = [("target", F1)] + [(nm, runs[nm]["I_final"]) for nm in
                             ("plain", "weighted_solve", "masked_alpha", "weighted_and_masked", "paper")]
fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 4))
for ax, (title, img) in zip(axes, panels):
    ax.imshow(img, cmap="gray"); ax.set_title(title, fontsize=8); ax.axis("off")
plt.suptitle(f"Reconstructions — {target_name}")
plt.tight_layout()
fig.savefig(os.path.join(output_dir, f"{target_name}_reconstructions.png"), dpi=150, bbox_inches="tight")
plt.show()

#%% ---------- CSV ----------
csv_path = os.path.join(output_dir, "weighted_summary.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image", "method", "final_rmse", "roughness", "final_vortices"])
    for name, r in runs.items():
        writer.writerow([target_name, name, r["RMSE"][-1], rmse_roughness(r["RMSE"], loop // 2), int(r["NUM"][-1])])

print(f"\nFigures + summary saved to: {output_dir}")

# %%
