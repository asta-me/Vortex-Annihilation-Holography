#%%
"""
Floor sweep for the irrotational PROJECTOR (clean rewrite).

Same pipeline as vah_ph_projector_test.py (relaxed irrotational projector vs paper periodic
VAH, on a single target), but the ONLY swept variable is `target_floor_rel`. Nothing else
(no weighted solve / masking / alpha schedules — those live in the regularization script).

For each floor value we run three methods on the SAME target (re-floored):
    - "off"       : projector disabled (baseline GS, alpha = 0),
    - "projector" : relaxed irrotational projector every iteration (fixed PROJECTOR_ALPHA),
    - "paper"     : periodic arctan2 vortex annihilation (reference),
then compare them per iteration and summarize settled RMSE / roughness / vortex count vs floor.

Why: on dark-heavy targets (e.g. Cat_black.tif) the projector's single GLOBAL FFT Poisson
solve is polluted by the UNDEFINED phase in near-zero-intensity regions (the wrapped gradient
there is noise, and the global solve smears it everywhere). Raising the target floor defines
the phase everywhere and is the strongest single lever; this script isolates that effect.

NOTE on "roughness": it is mean |RMSE[i] - RMSE[i-1]| over the tail iterations, i.e. how
JITTERY the RMSE convergence curve is (a TEMPORAL noise metric of the optimization). It is
NOT the spatial contrast (sigma/mu) of the reconstructed image.

RESULTS (Cat_black.tif, pure projector alpha=1.0, 300 iters)
------------------------------------------------------------
    floor   proj RMSE  roughness  vortices  | paper RMSE
    0.001   20.7       0.95       9680      | 9.29
    0.003   18.5       1.00       8925      | 7.89
    0.01    20.0       0.94       7486      | 6.66
    0.03     5.70      0.055        18      | 5.75   <- projector ~matches/slightly beats paper
    0.1      4.96      0.013         0      | 6.56   <- sweet spot (clearly beats paper)
    0.3      6.16      0.014         0      | 10.18  <- rising again: uniform-background collapse
Takeaway: from floor ~0.03 upward the projector already >= paper (only slightly at 0.03, clearly
at 0.1); below 0.03 it is catastrophic (undefined dark-region phase pollutes the single global
solve); at 0.3 RMSE rises again (a large uniform floor is a diffuse background that a vortex-free
phase cannot reconstruct). Practical window for this target: floor ~0.03-0.1, sweet spot ~0.1.
With alpha=1 the transition is a sharp cliff between 0.01 and 0.03; with alpha=0.5 it is smoother.

Env: vortex (conda), GPU (CuPy). PIL + cv2 (no skimage; see repo note on cupy+skimage crash).
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

# Shared vortex functions and input images live in the sibling folder 1_Alternative_projection.
ALT_PROJ_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "1_Alternative_projection")
sys.path.insert(0, ALT_PROJ_DIR)
from function_vortex_detection_accegpu import function_vortex_detection_accegpu
from function_vortex_elimination_accegpu import function_vortex_elimination_accegpu

#%% ---------- Config ----------
lamda = 532e-6                            # [mm] Wavelength
dh = 0.00374                              # [mm] Pixel pitch
loop = 300                                # Iterations per run
seed = 42                                 # Reproducibility

input_tiff = "Cat_black.tif"              # <-- CHANGE TARGET HERE (single target)
floor_list = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]  # <-- the ONLY swept variable
PROJECTOR_ALPHA = 1.0                     # Projector strength (1 = pure irrotational, 0.5 = relaxed)
paper_x = 100                             # Paper annihilation period (reference method)

#%% ---------- Fixed computational grid (independent of the floor) ----------
n = m = 512                                                    # Signal-region size after resize
nn = n + 2 * (n // 4)                                          # 768: padded SLM-plane grid
mm = m + 2 * (m // 4)

bandlim_spe = cp.zeros((nn, mm), dtype=cp.float32)            # SLM aperture (central 384)
bandlim_spe[nn // 4:3 * nn // 4, mm // 4:3 * mm // 4] = 1.0
bandlim_in = cp.zeros((nn, mm), dtype=cp.float32)             # Signal region SR (central 512)
bandlim_in[(nn - n) // 2:(nn + n) // 2, (mm - m) // 2:(mm + m) // 2] = 1.0
bandlim_ou = 1.0 - bandlim_in                                 # Noise region NR

sr_r0, sr_r1 = (nn - n) // 2, (nn + n) // 2                    # SR crop indices
sr_c0, sr_c1 = (mm - m) // 2, (mm + m) // 2

w = 0.26                                                      # [mm] Incident Gaussian beam waist
ox, oy = cp.meshgrid(cp.linspace(-dh * mm / 2, dh * mm / 2, mm), cp.linspace(-dh * nn / 2, dh * nn / 2, nn))
Gaussian = cp.exp(-((ox ** 2) + (oy ** 2)) / w)
incident = Gaussian * bandlim_spe

# Poisson-solver denominator (eigenvalues of the 5-point Laplacian, periodic BC), for the SR crop.
_ii = cp.arange(n).reshape(n, 1)
_jj = cp.arange(m).reshape(1, m)
_denom = 2 * cp.cos(2 * cp.pi * _ii / n) + 2 * cp.cos(2 * cp.pi * _jj / m) - 4
_denom[0, 0] = 1.0                                            # Avoid divide-by-zero (DC term)

script_dir = os.path.dirname(os.path.abspath(__file__))
input_tiff_path = os.path.join(ALT_PROJ_DIR, input_tiff)
if not os.path.isfile(input_tiff_path):
    raise FileNotFoundError(f"TIFF file not found: {input_tiff_path}")

output_dir = os.path.join(script_dir, "output_test_floor_sweep")
os.makedirs(output_dir, exist_ok=True)

# Raw target loaded once; the floor is applied per-sweep-value in load_target().
_F1_raw = np.array(Image.open(input_tiff_path))
if _F1_raw.ndim == 3:
    _F1_raw = _F1_raw[..., 0]
_F1_raw = cv2.resize(_F1_raw.astype(np.float32), (m, n), interpolation=cv2.INTER_AREA)


#%% ---------- Per-floor target + core method (identical projector to vah_ph_projector_test.py) ----------
def load_target(floor_rel):
    """Build the GPU target arrays for a given relative floor."""
    F1 = np.maximum(_F1_raw, floor_rel * (np.max(_F1_raw) + 1e-12))
    E = float(np.sum(F1))
    El = 0.5 * E
    F = np.abs(np.sqrt(F1))
    F = np.pad(F, ((n // 4, n // 4), (m // 4, m // 4)), mode="constant")  # 512 -> 768
    return dict(
        F1=F1,
        F_gpu=cp.asarray(F),
        E_gpu=cp.asarray(E),
        El_gpu=cp.asarray(El),
        F1_gpu=cp.asarray(F1),
    )


def irrotational_phase(pha):
    """Least-squares irrotational (vortex-free) phase via a single-FFT Poisson solve (SR size)."""
    dx = cp.zeros((n, m))
    dy = cp.zeros((n, m))
    dx[:, :-1] = cp.mod(pha[:, 1:] - pha[:, :-1] + cp.pi, 2 * cp.pi) - cp.pi
    dy[:-1, :] = cp.mod(pha[1:, :] - pha[:-1, :] + cp.pi, 2 * cp.pi) - cp.pi
    rho = cp.zeros((n, m))
    rho[:, 0] = dx[:, 0]
    rho[:, 1:] = dx[:, 1:] - dx[:, :-1]
    rho[0, :] += dy[0, :]
    rho[1:, :] += dy[1:, :] - dy[:-1, :]
    rho_hat = cp.fft.fft2(rho)
    rho_hat[0, 0] = 0.0
    return cp.real(cp.fft.ifft2(rho_hat / _denom))


def final_reconstruction(E2_k, E_gpu):
    """Final reconstructed SR intensity and phase from the last SLM-plane field."""
    An = cp.angle(E2_k)
    hologram = incident * cp.exp(1j * An)
    Rec = cp.fft.fftshift(cp.fft.ifft2(cp.fft.fftshift(hologram)))
    Rec_sr = Rec[sr_r0:sr_r1, sr_c0:sr_c1]
    I_final = cp.abs(Rec_sr) ** 2
    I_final = E_gpu * I_final / (cp.sum(I_final) + 1e-12)
    P_final = cp.mod(cp.angle(Rec_sr), 2 * cp.pi)
    return cp.asnumpy(I_final), cp.asnumpy(P_final)


def run_projector(ctx, alpha):
    """Alternative projection with the relaxed irrotational projector (alpha=0 -> baseline off)."""
    F_gpu, E_gpu, El_gpu, F1_gpu = ctx["F_gpu"], ctx["E_gpu"], ctx["El_gpu"], ctx["F1_gpu"]
    cp.random.seed(seed)
    phi = cp.exp(1j * 2 * cp.pi * cp.random.rand(nn, mm))
    amp = cp.random.rand(nn, mm)
    RMSE = np.zeros(loop)
    NUM = np.zeros(loop, dtype=int)

    for i in range(1, loop):
        amp = bandlim_in * F_gpu + bandlim_ou * amp
        E1 = amp * phi
        E2 = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(E1)))
        E2_ave = cp.sqrt((E_gpu + El_gpu) * incident ** 2 / cp.sum(incident ** 2))
        E2_k = E2_ave * cp.exp(1j * cp.angle(E2))
        es = cp.fft.fftshift(cp.fft.ifft2(cp.fft.fftshift(E2_k)))

        amp = cp.abs(es)
        amp_in = bandlim_in * amp
        amp_ou = bandlim_ou * amp
        amp = cp.sqrt(E_gpu) * (amp_in / (cp.sqrt(cp.sum(amp_in ** 2)) + 1e-12)) \
            + cp.sqrt(El_gpu) * (amp_ou / (cp.sqrt(cp.sum(amp_ou ** 2)) + 1e-12))

        I = amp[sr_r0:sr_r1, sr_c0:sr_c1] ** 2
        I = E_gpu * I / (cp.sum(I) + 1e-12)
        RMSE[i] = float(cp.sqrt(cp.mean((I - F1_gpu) ** 2)).get())

        pha = cp.angle(es)
        pha_crop = pha[sr_r0:sr_r1, sr_c0:sr_c1]
        po, ne = function_vortex_detection_accegpu(pha_crop, dh, use_cupy=True)
        NUM[i] = po + ne

        if alpha > 0.0:
            psi = irrotational_phase(pha_crop)
            psi = psi + cp.angle(cp.sum(cp.exp(1j * (pha_crop - psi))))
            field = (1 - alpha) * cp.exp(1j * pha_crop) + alpha * cp.exp(1j * psi)
            pha_new = pha.copy()
            pha_new[sr_r0:sr_r1, sr_c0:sr_c1] = cp.angle(field)
            phi = cp.exp(1j * pha_new)
        else:
            phi = cp.exp(1j * pha)

    I_final, P_final = final_reconstruction(E2_k, E_gpu)
    return RMSE, NUM, I_final, P_final


def run_paper(ctx, x):
    """Paper method: periodic arctan2 vortex annihilation every x iterations (reference)."""
    F_gpu, E_gpu, El_gpu, F1_gpu = ctx["F_gpu"], ctx["E_gpu"], ctx["El_gpu"], ctx["F1_gpu"]
    n_cycles = max(2, round(loop / x))
    total = n_cycles * x
    cp.random.seed(seed)
    phi = cp.exp(1j * 2 * cp.pi * cp.random.rand(nn, mm))
    amp = cp.random.rand(nn, mm)
    RMSE = np.zeros(total)
    NUM = np.zeros(total, dtype=int)

    for i in range(1, total):
        amp = bandlim_in * F_gpu + bandlim_ou * amp
        E1 = amp * phi
        E2 = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(E1)))
        E2_ave = cp.sqrt((E_gpu + El_gpu) * incident ** 2 / cp.sum(incident ** 2))
        E2_k = E2_ave * cp.exp(1j * cp.angle(E2))
        es = cp.fft.fftshift(cp.fft.ifft2(cp.fft.fftshift(E2_k)))

        amp = cp.abs(es)
        amp_in = bandlim_in * amp
        amp_ou = bandlim_ou * amp
        amp = cp.sqrt(E_gpu) * (amp_in / (cp.sqrt(cp.sum(amp_in ** 2)) + 1e-12)) \
            + cp.sqrt(El_gpu) * (amp_ou / (cp.sqrt(cp.sum(amp_ou ** 2)) + 1e-12))

        I = amp[sr_r0:sr_r1, sr_c0:sr_c1] ** 2
        I = E_gpu * I / (cp.sum(I) + 1e-12)
        RMSE[i] = float(cp.sqrt(cp.mean((I - F1_gpu) ** 2)).get())

        pha = cp.angle(es)
        pha_crop = pha[sr_r0:sr_r1, sr_c0:sr_c1]
        po, ne = function_vortex_detection_accegpu(pha_crop, dh, use_cupy=True)
        NUM[i] = po + ne

        if i % x == 0:
            pha_vfree_crop = function_vortex_elimination_accegpu(pha_crop, dh, use_cupy=True, gather_output=False)
            pha_new = pha.copy()
            pha_new[sr_r0:sr_r1, sr_c0:sr_c1] = pha_vfree_crop
            phi = cp.exp(1j * pha_new)
        else:
            phi = cp.exp(1j * pha)

    I_final, P_final = final_reconstruction(E2_k, E_gpu)
    return RMSE, NUM, I_final, P_final


def rmse_roughness(rmse_array, warmup):
    """Mean |delta RMSE| over the tail = temporal jitter of the RMSE curve (NOT spatial contrast)."""
    tail = rmse_array[warmup:]
    if len(tail) < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(tail))))


#%% ---------- Run the floor sweep ----------
target_name = os.path.splitext(input_tiff)[0]
print(f"Floor sweep on {input_tiff}  (projector alpha={PROJECTOR_ALPHA}, paper x={paper_x})", flush=True)

results = {}   # floor -> dict(off=..., projector=..., paper=..., F1=...)
for floor in floor_list:
    ctx = load_target(floor)
    t0 = time.perf_counter()
    off = run_projector(ctx, 0.0)
    proj = run_projector(ctx, PROJECTOR_ALPHA)
    paper = run_paper(ctx, paper_x)
    results[floor] = dict(off=off, projector=proj, paper=paper, F1=ctx["F1"])
    print(f"  floor={floor:<6g}  "
          f"off RMSE={off[0][-1]:.4f}  "
          f"proj RMSE={proj[0][-1]:.4f} (rough {rmse_roughness(proj[0], loop//2):.4f}, vort {proj[1][-1]})  "
          f"paper RMSE={paper[0][-1]:.4f} (rough {rmse_roughness(paper[0], len(paper[0])//2):.4f})  "
          f"({time.perf_counter()-t0:.1f}s)", flush=True)


#%% ---------- Plots ----------
floors = list(floor_list)
cmap = plt.cm.viridis(np.linspace(0, 1, len(floors)))

# 1) RMSE vs iteration (projector), one curve per floor
fig = plt.figure()
for c, f in zip(cmap, floors):
    plt.plot(results[f]["projector"][0][1:], color=c, label=f"floor={f:g}")
plt.xlabel("Iteration"); plt.ylabel("RMSE (SR)")
plt.title(f"Projector RMSE vs iteration — {target_name} (alpha={PROJECTOR_ALPHA})")
plt.legend(fontsize=7)
fig.savefig(os.path.join(output_dir, f"{target_name}_rmse_vs_iter.png"), dpi=150, bbox_inches="tight")
plt.show()

# 2) Vortex count vs iteration (projector), one curve per floor
fig = plt.figure()
for c, f in zip(cmap, floors):
    plt.plot(results[f]["projector"][1][1:], color=c, label=f"floor={f:g}")
plt.xlabel("Iteration"); plt.ylabel("Vortex count (SR)")
plt.title(f"Projector vortex count vs iteration — {target_name} (alpha={PROJECTOR_ALPHA})")
plt.legend(fontsize=7)
fig.savefig(os.path.join(output_dir, f"{target_name}_vortex_vs_iter.png"), dpi=150, bbox_inches="tight")
plt.show()

# 3) Settled RMSE vs floor: off / projector / paper
fig = plt.figure()
plt.plot(floors, [results[f]["off"][0][-1] for f in floors], "o-", label="off (baseline)")
plt.plot(floors, [results[f]["projector"][0][-1] for f in floors], "o-", label=f"projector a={PROJECTOR_ALPHA}")
plt.plot(floors, [results[f]["paper"][0][-1] for f in floors], "x--", color="k", label=f"paper x={paper_x}")
plt.xscale("log"); plt.xlabel("target_floor_rel"); plt.ylabel("Settled RMSE (SR)")
plt.title(f"Settled RMSE vs floor — {target_name}")
plt.legend()
fig.savefig(os.path.join(output_dir, f"{target_name}_rmse_vs_floor.png"), dpi=150, bbox_inches="tight")
plt.show()

# 4) Roughness vs floor: off / projector / paper
fig = plt.figure()
plt.plot(floors, [rmse_roughness(results[f]["off"][0], loop//2) for f in floors], "o-", label="off (baseline)")
plt.plot(floors, [rmse_roughness(results[f]["projector"][0], loop//2) for f in floors], "o-", label=f"projector a={PROJECTOR_ALPHA}")
plt.plot(floors, [rmse_roughness(results[f]["paper"][0], len(results[f]["paper"][0])//2) for f in floors], "x--", color="k", label=f"paper x={paper_x}")
plt.xscale("log"); plt.yscale("log"); plt.xlabel("target_floor_rel"); plt.ylabel("RMSE roughness (mean |dRMSE|)")
plt.title(f"RMSE roughness vs floor — {target_name}")
plt.legend()
fig.savefig(os.path.join(output_dir, f"{target_name}_roughness_vs_floor.png"), dpi=150, bbox_inches="tight")
plt.show()

# 5) Settled vortex count vs floor
fig = plt.figure()
plt.plot(floors, [results[f]["off"][1][-1] for f in floors], "o-", label="off (baseline)")
plt.plot(floors, [results[f]["projector"][1][-1] for f in floors], "o-", label=f"projector a={PROJECTOR_ALPHA}")
plt.plot(floors, [results[f]["paper"][1][-1] for f in floors], "x--", color="k", label=f"paper x={paper_x}")
plt.xscale("log"); plt.xlabel("target_floor_rel"); plt.ylabel("Settled vortex count (SR)")
plt.title(f"Settled vortex count vs floor — {target_name}")
plt.legend()
fig.savefig(os.path.join(output_dir, f"{target_name}_vortex_vs_floor.png"), dpi=150, bbox_inches="tight")
plt.show()

# 6) Reconstruction panels: top row = floored target, bottom row = projector reconstruction
ncols = len(floors)
fig, axes = plt.subplots(2, ncols, figsize=(3.2 * ncols, 6.4))
for j, f in enumerate(floors):
    axes[0, j].imshow(results[f]["F1"], cmap="gray"); axes[0, j].set_title(f"target floor={f:g}", fontsize=8); axes[0, j].axis("off")
    axes[1, j].imshow(results[f]["projector"][2], cmap="gray"); axes[1, j].set_title(f"proj recon", fontsize=8); axes[1, j].axis("off")
plt.suptitle(f"Target (floored) vs projector reconstruction — {target_name} (alpha={PROJECTOR_ALPHA})")
plt.tight_layout()
fig.savefig(os.path.join(output_dir, f"{target_name}_reconstruction_vs_floor.png"), dpi=150, bbox_inches="tight")
plt.show()

#%% ---------- CSV summary ----------
csv_path = os.path.join(output_dir, "floor_sweep_summary.csv")
with open(csv_path, "w", newline="") as fcsv:
    writer = csv.writer(fcsv)
    writer.writerow(["image", "floor_rel", "method", "settled_rmse", "roughness", "settled_vortices"])
    for f in floors:
        for method in ("off", "projector", "paper"):
            RMSE, NUM = results[f][method][0], results[f][method][1]
            writer.writerow([target_name, f, method, RMSE[-1], rmse_roughness(RMSE, len(RMSE) // 2), int(NUM[-1])])

print(f"\nFigures + summary saved to: {output_dir}")

# %%
