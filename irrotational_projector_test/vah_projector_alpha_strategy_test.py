#%%
"""
Alpha-strategy study for the irrotational PROJECTOR.

Question this script answers
----------------------------
For the relaxed irrotational projector
    phi = (1-alpha)*exp(i*phase) + alpha*exp(i*psi_irrotational)
applied every GS iteration on the target-plane SR, what is the best way to choose alpha?

    1. Is there an OPTIMAL CONSTANT alpha?           -> constant sweep {0.25, 0.5, 0.75, 1.0}
    2. Should alpha be SCHEDULED, and how?
         - decreasing (strong early, ~0 late):       linear_down, cosine_down
         - increasing (weak early, strong late):      linear_up,  cosine_up
         - init-then-off (strong for a while, then 0): init_off_30
         - plateau-triggered pulse (paper-style):      plateau_pulse
    3. Reference baselines:                           off (alpha=0), paper periodic arctan2 VAH.

All strategies run on the SAME single target (change `input_tiff`), same RNG seed, same floor,
so the comparison is apples-to-apples. This consolidates the previous asinit / regularization
experiments into one clean script.

Metrics per iteration: SR RMSE, natural vortex count, and the applied alpha.
"roughness" = mean |RMSE[i]-RMSE[i-1]| over the tail = temporal jitter of the RMSE curve
(NOT spatial contrast).

RESULT / CONCLUSION
-------------------
Once a GOOD FLOOR is fixed, all alpha strategies give EXTREMELY SIMILAR results -> just keep a
CONSTANT alpha (the simplest choice). On marmo.tif (full, floor 5e-3, 400 iters) every projector
strategy lands RMSE ~7.5-8.5 (vs off 35.7, paper 12.5); a constant ~0.5-0.75 is near-best AND
vortex-free. Scheduling adds nothing useful:
  - decreasing-to-0 / init-off : RMSE fine, but vortices REAPPEAR once the projector is off late;
  - increasing                 : strictly worse (misses the crucial early cleanup), noisier;
  - plateau_pulse (intermittent): the noisiest, worst of the projector family.
Even on the hard dark-heavy Cat_black (with a good floor ~0.05-0.1) the differences between
strategies are MINIMAL. => Recommendation: fix a good floor, use a CONSTANT alpha ~0.5-0.75,
kept on to the end. (plateau_pulse needs a RELATIVE slope trigger, mean|dRMSE|/mean(RMSE) < tol,
else it never fires when RMSE is on a large absolute scale.)

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
lamda = 532e-6                            # [mm] Wavelength
dh = 0.00374                              # [mm] Pixel pitch
loop = 400                                # Iterations per run
seed = 42                                 # Reproducibility

input_tiff = "Cat_black.tif"                  # <-- CHANGE TARGET HERE
target_floor_rel = 0.05                   # Fixed floor (use ~0.03-0.1 for dark-heavy targets)
paper_x = 100                             # Paper annihilation period (reference)

# The alpha strategies to compare.
STRATEGIES = [
    {"name": "off",           "type": "off"},
    {"name": "const_0.25",    "type": "const", "alpha": 0.25},
    {"name": "const_0.5",     "type": "const", "alpha": 0.50},
    {"name": "const_0.75",    "type": "const", "alpha": 0.75},
    {"name": "const_1.0",     "type": "const", "alpha": 1.00},
    {"name": "linear_down",   "type": "schedule", "kind": "linear_down", "amin": 0.0, "amax": 1.0},
    {"name": "cosine_down",   "type": "schedule", "kind": "cosine_down", "amin": 0.0, "amax": 1.0},
    {"name": "linear_up",     "type": "schedule", "kind": "linear_up",   "amin": 0.0, "amax": 1.0},
    {"name": "cosine_up",     "type": "schedule", "kind": "cosine_up",   "amin": 0.0, "amax": 1.0},
    {"name": "init_off_30",   "type": "init_off", "alpha": 1.0, "switch_frac": 0.30},
    {"name": "plateau_pulse", "type": "plateau_pulse", "baseline": 0.0, "pulse": 1.0,
     "pulse_len": 5, "window": 15, "tol_rel": 1e-3, "warmup": 30},
]
CONST_NAMES = ["const_0.25", "const_0.5", "const_0.75", "const_1.0"]
SCHED_NAMES = ["linear_down", "cosine_down", "linear_up", "cosine_up", "init_off_30", "plateau_pulse"]

#%% ---------- Fixed computational grid ----------
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
output_dir = os.path.join(script_dir, "output_test_alpha_strategy")
os.makedirs(output_dir, exist_ok=True)

#%% ---------- Target ----------
input_tiff_path = os.path.join(ALT_PROJ_DIR, input_tiff)
if not os.path.isfile(input_tiff_path):
    raise FileNotFoundError(f"TIFF file not found: {input_tiff_path}")
_F1 = np.array(Image.open(input_tiff_path))
if _F1.ndim == 3:
    _F1 = _F1[..., 0]
_F1 = cv2.resize(_F1.astype(np.float32), (m, n), interpolation=cv2.INTER_AREA)
F1 = np.maximum(_F1, target_floor_rel * (np.max(_F1) + 1e-12))
E = float(np.sum(F1))
El = 0.5 * E
F = np.pad(np.abs(np.sqrt(F1)), ((n // 4, n // 4), (m // 4, m // 4)), mode="constant")

F_gpu = cp.asarray(F)
E_gpu = cp.asarray(E)
El_gpu = cp.asarray(El)
F1_gpu = cp.asarray(F1)
target_name = os.path.splitext(input_tiff)[0]

#%% ---------- Core pieces ----------
def irrotational_phase(pha):
    dx = cp.zeros((n, m)); dy = cp.zeros((n, m))
    dx[:, :-1] = cp.mod(pha[:, 1:] - pha[:, :-1] + cp.pi, 2 * cp.pi) - cp.pi
    dy[:-1, :] = cp.mod(pha[1:, :] - pha[:-1, :] + cp.pi, 2 * cp.pi) - cp.pi
    rho = cp.zeros((n, m))
    rho[:, 0] = dx[:, 0]; rho[:, 1:] = dx[:, 1:] - dx[:, :-1]
    rho[0, :] += dy[0, :]; rho[1:, :] += dy[1:, :] - dy[:-1, :]
    rho_hat = cp.fft.fft2(rho); rho_hat[0, 0] = 0.0
    return cp.real(cp.fft.ifft2(rho_hat / _denom))


def alpha_from_schedule(i, kind, amin, amax):
    t = i / max(loop - 1, 1)
    if kind == "linear_up":
        return amin + (amax - amin) * t
    if kind == "linear_down":
        return amax - (amax - amin) * t
    if kind == "cosine_up":
        return amin + (amax - amin) * (1 - np.cos(np.pi * t)) / 2
    if kind == "cosine_down":
        return amax - (amax - amin) * (1 - np.cos(np.pi * t)) / 2
    raise ValueError(kind)


def final_reconstruction(E2_k):
    An = cp.angle(E2_k)
    hologram = incident * cp.exp(1j * An)
    Rec = cp.fft.fftshift(cp.fft.ifft2(cp.fft.fftshift(hologram)))
    Rec_sr = Rec[sr_r0:sr_r1, sr_c0:sr_c1]
    I_final = cp.abs(Rec_sr) ** 2
    I_final = E_gpu * I_final / (cp.sum(I_final) + 1e-12)
    P_final = cp.mod(cp.angle(Rec_sr), 2 * cp.pi)
    return cp.asnumpy(I_final), cp.asnumpy(P_final)


def rmse_roughness(rmse_array, warmup):
    tail = rmse_array[warmup:]
    if len(tail) < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(tail))))


def run_strategy(strat):
    """Run GS with the projector, resolving alpha each iteration per the given strategy."""
    cp.random.seed(seed)
    phi = cp.exp(1j * 2 * cp.pi * cp.random.rand(nn, mm))
    amp = cp.random.rand(nn, mm)
    RMSE = np.zeros(loop)
    NUM = np.zeros(loop, dtype=int)
    ALPHA = np.zeros(loop)

    switch_iter = int(strat.get("switch_frac", 0.0) * loop)
    pulse_countdown = 0

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

        # ---- resolve alpha for this iteration ----
        t = strat["type"]
        if t == "off":
            alpha = 0.0
        elif t == "const":
            alpha = strat["alpha"]
        elif t == "schedule":
            alpha = alpha_from_schedule(i, strat["kind"], strat["amin"], strat["amax"])
        elif t == "init_off":
            alpha = strat["alpha"] if i < switch_iter else 0.0
        elif t == "plateau_pulse":
            if pulse_countdown > 0:
                alpha = strat["pulse"]; pulse_countdown -= 1
            elif i > strat["warmup"] and i > strat["window"] + 1:
                tail = RMSE[i - strat["window"] + 1:i + 1]
                rel_slope = float(np.mean(np.abs(np.diff(tail))) / (np.mean(tail) + 1e-12))
                if rel_slope < strat["tol_rel"]:
                    pulse_countdown = strat["pulse_len"] - 1
                    alpha = strat["pulse"]
                else:
                    alpha = strat["baseline"]
            else:
                alpha = strat["baseline"]
        else:
            raise ValueError(t)
        ALPHA[i] = alpha

        if alpha > 0.0:
            psi = irrotational_phase(pha_crop)
            psi = psi + cp.angle(cp.sum(cp.exp(1j * (pha_crop - psi))))
            field = (1 - alpha) * cp.exp(1j * pha_crop) + alpha * cp.exp(1j * psi)
            pha_new = pha.copy()
            pha_new[sr_r0:sr_r1, sr_c0:sr_c1] = cp.angle(field)
            phi = cp.exp(1j * pha_new)
        else:
            phi = cp.exp(1j * pha)

    I_final, P_final = final_reconstruction(E2_k)
    return dict(RMSE=RMSE, NUM=NUM, ALPHA=ALPHA, I_final=I_final, P_final=P_final)


def run_paper(x):
    n_cycles = max(2, round(loop / x))
    total = n_cycles * x
    cp.random.seed(seed)
    phi = cp.exp(1j * 2 * cp.pi * cp.random.rand(nn, mm))
    amp = cp.random.rand(nn, mm)
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
            pha_vfree_crop = function_vortex_elimination_accegpu(pha_crop, dh, use_cupy=True, gather_output=False)
            pha_new = pha.copy(); pha_new[sr_r0:sr_r1, sr_c0:sr_c1] = pha_vfree_crop
            phi = cp.exp(1j * pha_new)
        else:
            phi = cp.exp(1j * pha)
    I_final, P_final = final_reconstruction(E2_k)
    return dict(RMSE=RMSE, NUM=NUM, ALPHA=np.zeros(total), I_final=I_final, P_final=P_final)


#%% ---------- Run all strategies ----------
print(f"Alpha-strategy study on {input_tiff} (floor={target_floor_rel}, loop={loop})", flush=True)
results = {}
for strat in STRATEGIES:
    t0 = time.perf_counter()
    r = run_strategy(strat)
    results[strat["name"]] = r
    print(f"  {strat['name']:14s} final RMSE={r['RMSE'][-1]:.5f}  "
          f"roughness={rmse_roughness(r['RMSE'], loop//2):.5f}  vortices={r['NUM'][-1]:5d}  "
          f"({time.perf_counter()-t0:.1f}s)", flush=True)

paper = run_paper(paper_x)
print(f"  {'paper x'+str(paper_x):14s} final RMSE={paper['RMSE'][-1]:.5f}  "
      f"roughness={rmse_roughness(paper['RMSE'], len(paper['RMSE'])//2):.5f}  vortices={paper['NUM'][-1]:5d}")

best_const = min(CONST_NAMES, key=lambda nm: results[nm]["RMSE"][-1])

#%% ---------- Plots ----------
# 1) Constant-alpha sweep: is there an optimal constant alpha?
fig = plt.figure()
plt.plot(results["off"]["RMSE"][1:], label="off")
for nm in CONST_NAMES:
    plt.plot(results[nm]["RMSE"][1:], label=nm)
plt.plot(paper["RMSE"][1:], "k--", label=f"paper x={paper_x}")
plt.xlabel("Iteration"); plt.ylabel("RMSE (SR)")
plt.title(f"Constant-alpha sweep — {target_name}")
plt.legend(fontsize=8)
fig.savefig(os.path.join(output_dir, f"{target_name}_const_rmse.png"), dpi=150, bbox_inches="tight")
plt.show()

# 2) Schedules vs best constant: should alpha be scheduled, and how?
fig = plt.figure()
plt.plot(results[best_const]["RMSE"][1:], label=f"best const ({best_const})", linewidth=2)
for nm in SCHED_NAMES:
    plt.plot(results[nm]["RMSE"][1:], label=nm)
plt.plot(paper["RMSE"][1:], "k--", label=f"paper x={paper_x}")
plt.xlabel("Iteration"); plt.ylabel("RMSE (SR)")
plt.title(f"Scheduling strategies vs best constant — {target_name}")
plt.legend(fontsize=8)
fig.savefig(os.path.join(output_dir, f"{target_name}_sched_rmse.png"), dpi=150, bbox_inches="tight")
plt.show()

# 3) Applied alpha(i) for each strategy
fig = plt.figure()
for nm in CONST_NAMES + SCHED_NAMES:
    plt.plot(results[nm]["ALPHA"][1:], label=nm)
plt.xlabel("Iteration"); plt.ylabel("Applied alpha")
plt.title(f"Alpha schedules — {target_name}")
plt.legend(fontsize=7, ncol=2)
fig.savefig(os.path.join(output_dir, f"{target_name}_alpha_curves.png"), dpi=150, bbox_inches="tight")
plt.show()

# 4) Vortex count vs iteration (schedules + best const + paper)
fig = plt.figure()
plt.plot(results[best_const]["NUM"][1:], label=f"best const ({best_const})", linewidth=2)
for nm in SCHED_NAMES:
    plt.plot(results[nm]["NUM"][1:], label=nm)
plt.plot(paper["NUM"][1:], "k--", label=f"paper x={paper_x}")
plt.xlabel("Iteration"); plt.ylabel("Vortex count (SR)")
plt.title(f"Vortex count vs iteration — {target_name}")
plt.legend(fontsize=8)
fig.savefig(os.path.join(output_dir, f"{target_name}_vortex.png"), dpi=150, bbox_inches="tight")
plt.show()

# 5) Final settled RMSE + roughness bars
names = [s["name"] for s in STRATEGIES] + [f"paper_x{paper_x}"]
rmses = [results[s["name"]]["RMSE"][-1] for s in STRATEGIES] + [paper["RMSE"][-1]]
roughs = [rmse_roughness(results[s["name"]]["RMSE"], loop // 2) for s in STRATEGIES] \
    + [rmse_roughness(paper["RMSE"], len(paper["RMSE"]) // 2)]
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8))
ax1.bar(range(len(names)), rmses); ax1.set_ylabel("Final settled RMSE")
ax1.set_xticks(range(len(names))); ax1.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
ax1.set_title(f"Final RMSE by strategy — {target_name}")
ax2.bar(range(len(names)), roughs, color="C1"); ax2.set_ylabel("RMSE roughness"); ax2.set_yscale("log")
ax2.set_xticks(range(len(names))); ax2.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
ax2.set_title("RMSE roughness by strategy")
plt.tight_layout()
fig.savefig(os.path.join(output_dir, f"{target_name}_bars.png"), dpi=150, bbox_inches="tight")
plt.show()

# 6) Trade-off: final RMSE vs final vortex count
fig = plt.figure()
for s in STRATEGIES:
    nm = s["name"]
    plt.scatter(results[nm]["RMSE"][-1], results[nm]["NUM"][-1], c="C0")
    plt.annotate(nm, (results[nm]["RMSE"][-1], results[nm]["NUM"][-1]), fontsize=7)
plt.scatter(paper["RMSE"][-1], paper["NUM"][-1], marker="x", c="k")
plt.annotate(f"paper x={paper_x}", (paper["RMSE"][-1], paper["NUM"][-1]), fontsize=7)
plt.xlabel("Final settled RMSE (SR)"); plt.ylabel("Final settled vortex count (SR)")
plt.title(f"Trade-off: singularities vs RMSE — {target_name}")
fig.savefig(os.path.join(output_dir, f"{target_name}_tradeoff.png"), dpi=150, bbox_inches="tight")
plt.show()

#%% ---------- CSV summary ----------
csv_path = os.path.join(output_dir, "alpha_strategy_summary.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image", "strategy", "final_rmse", "roughness", "final_vortices"])
    for s in STRATEGIES:
        nm = s["name"]; R = results[nm]
        writer.writerow([target_name, nm, R["RMSE"][-1], rmse_roughness(R["RMSE"], loop // 2), int(R["NUM"][-1])])
    writer.writerow([target_name, f"paper_x{paper_x}", paper["RMSE"][-1],
                     rmse_roughness(paper["RMSE"], len(paper["RMSE"]) // 2), int(paper["NUM"][-1])])

print(f"\nBest constant alpha: {best_const}  (RMSE={results[best_const]['RMSE'][-1]:.5f})")
print(f"Figures + summary saved to: {output_dir}")

# %%
