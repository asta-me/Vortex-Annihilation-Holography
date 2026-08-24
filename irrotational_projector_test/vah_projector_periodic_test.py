#%%
"""
Periodic / stagnation-triggered irrotational PROJECTOR vs the paper's periodic VAH.

Idea
----
Instead of applying the irrotational projector at EVERY iteration (vah_projector_fft_test.py),
apply it only NOW AND THEN -- exactly where the paper applies its arctan2 vortex annihilation:
    - "periodic"    : every `trigger_x` iterations, or
    - "stagnation"  : when the RMSE curve flattens (relative-slope criterion).
At each trigger we do a FULL projection (alpha_trigger, default 1.0), analogous to the paper's
full elimination. This lets us compare, head to head and on the SAME schedule:
    projector-at-trigger  vs  paper-elimination-at-trigger
in RMSE, vortex count, and -- above all -- TIME.

Why time matters
----------------
The paper's elimination is O(K * N^2) (one arctan2 over the whole grid per detected vortex K),
so it is SLOW when there are many vortices. The Poisson solve is O(N log N), INDEPENDENT of K.
This script times each annihilation event (GPU-synchronized) to quantify that gap.

Reference runs: projector every iteration (alpha_every), and baseline off.

RESULT (marmo.tif, x=50, 300 iters) -- KEPT: big speed win
-----------------------------------------------------------
Same schedule, projector-at-trigger vs paper-at-trigger:
    proj_periodic  : RMSE 10.5, 432 vort, t/event ~1.8 ms
    paper_periodic : RMSE 14.5, 782 vort, t/event ~415 ms   => ~232x slower per event
The projector is BETTER (RMSE + vortices) AND ~232x FASTER per annihilation (O(N log N) vs the
paper's O(K*N^2)). Even applied EVERY iteration (299 projections, 2.68 s total) it is faster than
the paper applied just 5 times (4.26 s). On the stagnation trigger the gap is larger (K is bigger
at the trigger): projector ~1.9 ms vs paper ~786 ms per event. => SPEED is the headline advantage.

Env: vortex (conda), GPU (CuPy). PIL + cv2 (no skimage).
"""

#%% ---------- Imports ----------
import time
import os
import sys
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

# Target: uncomment ONE line (files live in ./targets/, or give an absolute path).
input_tiff = "marmo.tif"
# input_tiff = "object_grayscale_from_mat.tif"
# input_tiff = "Lenna.tif"
# input_tiff = "Baboon.tif"
# input_tiff = "valentini.tif"
target_floor_rel = 5e-3
trigger_x = 50                            # period for the "periodic" mode
alpha_every = 0.5                         # blend for the every-iteration reference
alpha_trigger = 1.0                       # full projection applied at a trigger

# Geometry (fix the hologram size and the SR fraction; the rest is derived)
HOLOGRAM_SIZE = 384           # SLM aperture side [px] (= half the work grid; 2x oversampling)
WORK_SIZE = 2 * HOLOGRAM_SIZE  # computational grid side [px]
SR_FRACTION = 2 / 3           # signal-region side as a fraction of WORK_SIZE (image size)
SR_SIZE = int(round(SR_FRACTION * WORK_SIZE)); SR_SIZE -= SR_SIZE % 2
assert 0 < SR_SIZE <= WORK_SIZE, "SR_FRACTION out of range: signal region must fit the grid"

# Stagnation criterion (relative slope of the RMSE curve)
stag_window = 15
stag_tol_rel = 1e-3
stag_warmup = 30
stag_cooldown = 20                        # min iterations between two stagnation triggers

#%% ---------- Grid ----------
n = m = SR_SIZE
nn = mm = WORK_SIZE
pad_each = (WORK_SIZE - SR_SIZE) // 2
ap0 = (WORK_SIZE - HOLOGRAM_SIZE) // 2
bandlim_spe = cp.zeros((nn, mm), dtype=cp.float32)
bandlim_spe[ap0:ap0 + HOLOGRAM_SIZE, ap0:ap0 + HOLOGRAM_SIZE] = 1.0
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
TARGETS_DIR = os.path.join(script_dir, "targets")

#%% ---------- Target ----------
input_tiff_path = input_tiff if os.path.isabs(input_tiff) else os.path.join(TARGETS_DIR, input_tiff)
if not os.path.isfile(input_tiff_path):
    raise FileNotFoundError(f"TIFF file not found: {input_tiff_path}")
_F1 = np.array(Image.open(input_tiff_path))
if _F1.ndim == 3:
    _F1 = _F1[..., 0]
_F1 = cv2.resize(_F1.astype(np.float32), (m, n), interpolation=cv2.INTER_AREA)
F1 = np.maximum(_F1, target_floor_rel * (np.max(_F1) + 1e-12))
E = float(np.sum(F1)); El = 0.5 * E
F = np.pad(np.abs(np.sqrt(F1)), ((pad_each, pad_each), (pad_each, pad_each)), mode="constant")
F_gpu = cp.asarray(F); E_gpu = cp.asarray(E); El_gpu = cp.asarray(El); F1_gpu = cp.asarray(F1)
target_name = os.path.splitext(input_tiff)[0]
_dev = cp.cuda.Device()

#%% ---------- Core ----------
def irrotational_phase(pha):
    dx = cp.zeros((n, m)); dy = cp.zeros((n, m))
    dx[:, :-1] = cp.mod(pha[:, 1:] - pha[:, :-1] + cp.pi, 2 * cp.pi) - cp.pi
    dy[:-1, :] = cp.mod(pha[1:, :] - pha[:-1, :] + cp.pi, 2 * cp.pi) - cp.pi
    rho = cp.zeros((n, m))
    rho[:, 0] = dx[:, 0]; rho[:, 1:] = dx[:, 1:] - dx[:, :-1]
    rho[0, :] += dy[0, :]; rho[1:, :] += dy[1:, :] - dy[:-1, :]
    rho_hat = cp.fft.fft2(rho); rho_hat[0, 0] = 0.0
    return cp.real(cp.fft.ifft2(rho_hat / _denom))


def projector_op(pha_crop, alpha):
    """Full/relaxed irrotational projection of a phase crop -> vortex-free phase crop."""
    psi = irrotational_phase(pha_crop)
    psi = psi + cp.angle(cp.sum(cp.exp(1j * (pha_crop - psi))))
    field = (1 - alpha) * cp.exp(1j * pha_crop) + alpha * cp.exp(1j * psi)
    return cp.angle(field)


def paper_op(pha_crop):
    """Paper arctan2 vortex elimination -> vortex-free phase crop."""
    return function_vortex_elimination_accegpu(pha_crop, dh, use_cupy=True, gather_output=False)


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


def run(kind, mode=None, x=trigger_x):
    """kind in {'off','every','projector','paper'}; mode in {'periodic','stagnation'} for proj/paper."""
    cp.random.seed(seed)
    phi = cp.exp(1j * 2 * cp.pi * cp.random.rand(nn, mm))
    amp = cp.random.rand(nn, mm)
    RMSE = np.zeros(loop); NUM = np.zeros(loop, dtype=int)
    event_time = 0.0; n_events = 0; cooldown = 0

    _dev.synchronize(); t_total0 = time.perf_counter()
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

        pha = cp.angle(es)
        pha_crop = pha[sr_r0:sr_r1, sr_c0:sr_c1]
        po, ne = function_vortex_detection_accegpu(pha_crop, dh, use_cupy=True)
        NUM[i] = po + ne

        # decide whether to annihilate this iteration
        if kind == "off":
            apply = False
        elif kind == "every":
            apply = True
        else:  # projector / paper, periodic or stagnation
            if mode == "periodic":
                apply = (i % x == 0)
            else:  # stagnation
                apply = False
                if cooldown > 0:
                    cooldown -= 1
                elif i > stag_warmup and i > stag_window + 1:
                    tail = RMSE[i - stag_window + 1:i + 1]
                    rel = float(np.mean(np.abs(np.diff(tail))) / (np.mean(tail) + 1e-12))
                    if rel < stag_tol_rel:
                        apply = True; cooldown = stag_cooldown

        if apply:
            _dev.synchronize(); te0 = time.perf_counter()
            if kind == "paper":
                vfree = paper_op(pha_crop)
            else:  # every / projector
                a = alpha_every if kind == "every" else alpha_trigger
                vfree = projector_op(pha_crop, a)
            _dev.synchronize(); event_time += time.perf_counter() - te0; n_events += 1
            pha_new = pha.copy(); pha_new[sr_r0:sr_r1, sr_c0:sr_c1] = vfree
            phi = cp.exp(1j * pha_new)
        else:
            phi = cp.exp(1j * pha)

    _dev.synchronize(); total_time = time.perf_counter() - t_total0
    I_final = final_reconstruction(E2_k)
    per_event = (event_time / n_events) if n_events else 0.0
    return dict(RMSE=RMSE, NUM=NUM, I_final=I_final, total_time=total_time,
                event_time=event_time, n_events=n_events, per_event=per_event)


#%% ---------- Run ----------
print(f"Periodic Poisson-projector vs paper on {input_tiff} (x={trigger_x}, loop={loop})", flush=True)
runs = {
    "off":             run("off"),
    "proj_every":      run("every"),
    "proj_periodic":   run("projector", "periodic", trigger_x),
    "paper_periodic":  run("paper", "periodic", trigger_x),
    "proj_stagnation": run("projector", "stagnation"),
    "paper_stagnation": run("paper", "stagnation"),
}
for name, r in runs.items():
    print(f"  {name:16s} RMSE={r['RMSE'][-1]:8.4f}  vort={r['NUM'][-1]:5d}  "
          f"events={r['n_events']:3d}  t_total={r['total_time']:6.2f}s  "
          f"t/event={r['per_event']*1e3:8.2f}ms", flush=True)

#%% ---------- Results table (simple markdown, regenerated each run) ----------
tbl = [f"# Periodic vs every projector — {target_name}",
       f"geometry {HOLOGRAM_SIZE}/{WORK_SIZE}/{SR_SIZE}, loop={loop}, x={trigger_x}",
       "", "| method | RMSE | vortices | events | t_total [s] | t/event [ms] |",
       "|---|---|---|---|---|---|"]
for name, r in runs.items():
    tbl.append(f"| {name} | {r['RMSE'][-1]:.4f} | {r['NUM'][-1]} | {r['n_events']} | "
               f"{r['total_time']:.2f} | {r['per_event']*1e3:.2f} |")
table_md = "\n".join(tbl) + "\n"
table_path = os.path.join(script_dir, f"results_periodic_{target_name}.md")
with open(table_path, "w", encoding="utf-8") as fh:
    fh.write(table_md)
print(f"\nResults table -> {table_path}")

#%% ---------- Plots (all figures built, then a single plt.show() at the end) ----------
# 1) RMSE vs iteration
fig1 = plt.figure()
for name in ("off", "proj_every", "proj_periodic", "paper_periodic"):
    plt.plot(runs[name]["RMSE"][1:], label=name)
plt.xlabel("Iteration"); plt.ylabel("RMSE (SR)")
plt.title(f"RMSE vs iteration — {target_name} (period x={trigger_x})")
plt.legend(fontsize=8)

# 2) Vortex count vs iteration
fig2 = plt.figure()
for name in ("off", "proj_every", "proj_periodic", "paper_periodic"):
    plt.plot(runs[name]["NUM"][1:], label=name)
plt.xlabel("Iteration"); plt.ylabel("Vortex count (SR)")
plt.title(f"Vortex count vs iteration — {target_name} (period x={trigger_x})")
plt.legend(fontsize=8)

# 3) Time per annihilation event: projector vs paper (the key comparison)
fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
names_t = ["proj_periodic", "paper_periodic", "proj_stagnation", "paper_stagnation", "proj_every"]
ax1.bar(range(len(names_t)), [runs[nm]["per_event"] * 1e3 for nm in names_t], color="C0")
ax1.set_xticks(range(len(names_t))); ax1.set_xticklabels(names_t, rotation=45, ha="right", fontsize=7)
ax1.set_ylabel("Time per annihilation event [ms]"); ax1.set_title("Cost of ONE annihilation")
ax2.bar(range(len(names_t)), [runs[nm]["total_time"] for nm in names_t], color="C1")
ax2.set_xticks(range(len(names_t))); ax2.set_xticklabels(names_t, rotation=45, ha="right", fontsize=7)
ax2.set_ylabel("Total run time [s]"); ax2.set_title("Total wall-clock")
plt.tight_layout()

# speedup print
if runs["paper_periodic"]["per_event"] > 0:
    sp = runs["paper_periodic"]["per_event"] / max(runs["proj_periodic"]["per_event"], 1e-12)
    print(f"Per-event speedup (paper / projector, periodic): {sp:.1f}x")

plt.show()

# %%
