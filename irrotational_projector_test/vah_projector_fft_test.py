#%%
"""
Direction 1 test: relaxed irrotational PROJECTOR for vortex annihilation.

Idea
----
Instead of applying vortex annihilation only on an RMSE-plateau trigger (as in the
paper / vah_ph_commented.py), apply a RELAXED irrotational projector at EVERY iteration,
in the image plane, on the SR crop.

The irrotational (vortex-free) phase psi is obtained as the LEAST-SQUARES phase via an
FFT Poisson solve (Ghiglia-Romero): the wrapped phase gradient is differentiated to a
density rho, and nabla^2 psi = rho is solved with a single FFT. psi is curl-free by
construction (it is a genuine scalar potential -> no vortices), and the solve is
O(N^2 log N), INDEPENDENT of the number of vortices. This is the fast, principled
realization of the paper's "scalar potential / irrotational part", replacing the
O(K*N^2) per-vortex arctan2 sum.

The relaxation is a CONVEX BLEND in the complex field domain (topologically sound,
unlike scaling the quantized winding):

    phi = normalize[ (1-alpha) * exp(i*phi) + alpha * exp(i*psi) ]

alpha = 1 gives the pure irrotational phase; alpha = 0 is the baseline (no annihilation);
0 < alpha < 1 interpolates the two unit-modulus fields, which handles the vortex cores
smoothly (the field magnitude naturally dips there).

For each alpha we track:
    - the NATURAL vortex count on angle(es) per iteration (before the projector),
    - the SR-only RMSE per iteration (convergence check).

Finally we plot the trade-off between final singularity count and final RMSE.

Env: vortex (conda), GPU (CuPy).
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

# Shared vortex functions and input images live in the sibling folder 1_Alternative_projection.
ALT_PROJ_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "1_Alternative_projection")
sys.path.insert(0, ALT_PROJ_DIR)
from function_vortex_detection_accegpu import function_vortex_detection_accegpu
from function_vortex_elimination_accegpu import function_vortex_elimination_accegpu

#%% ---------- Config ----------
lamda = 532e-6                            # [mm] Wavelength
dh = 0.00374                              # [mm] Pixel pitch
loop = 500                                # Iterations per run
alpha_list = [0.0, 0.25, 0.5, 0.75, 1.0]  # Relaxation knob sweep (0 = off, 1 = full irrotational)
paper_x_list = [25, 50, 75, 100]          # Paper annihilation-period sweep (fair settled endpoint)
seed = 42                                 # Reproducibility
target_floor_rel = 5e-3                   # Relative floor wrt max(F1): prevents fully dark pixels

#%% ---------- Geometry (fix the hologram size and the SR fraction; the rest is derived) ----------
HOLOGRAM_SIZE = 384           # SLM aperture side [px] (= half the work grid; 2x oversampling)
WORK_SIZE = 2 * HOLOGRAM_SIZE  # computational grid side [px]
SR_FRACTION = 2 / 3           # signal-region side as a fraction of WORK_SIZE (image size)
SR_SIZE = int(round(SR_FRACTION * WORK_SIZE)); SR_SIZE -= SR_SIZE % 2
assert 0 < SR_SIZE <= WORK_SIZE, "SR_FRACTION out of range: signal region must fit the grid"
pad_each = (WORK_SIZE - SR_SIZE) // 2      # center the SR inside the work grid
ap0 = (WORK_SIZE - HOLOGRAM_SIZE) // 2     # SLM aperture offset

#%% ---------- Import target (TIFF) ----------
# Target: uncomment ONE line (files live in ./targets/, or give an absolute path).
input_tiff = "marmo.tif"
# input_tiff = "object_grayscale_from_mat.tif"
# input_tiff = "Lenna.tif"
# input_tiff = "Baboon.tif"
# input_tiff = "valentini.tif"

script_dir = os.path.dirname(os.path.abspath(__file__))
TARGETS_DIR = os.path.join(script_dir, "targets")
input_tiff_path = input_tiff if os.path.isabs(input_tiff) else os.path.join(TARGETS_DIR, input_tiff)
if not os.path.isfile(input_tiff_path):
    raise FileNotFoundError(f"TIFF file not found: {input_tiff_path}")

F1 = np.array(Image.open(input_tiff_path))
if F1.ndim == 3:
    # Keep behavior deterministic for RGB/RGBA TIFFs: use the first channel.
    F1 = F1[..., 0]
F1 = F1.astype(np.float32)
F1 = cv2.resize(F1, (SR_SIZE, SR_SIZE), interpolation=cv2.INTER_AREA)  # Resize to the signal-region size
# Reuse Marco's floor idea without changing this script's global intensity scale.
F1 = np.maximum(F1, target_floor_rel * (np.max(F1) + 1e-12))
n, m = F1.shape
E = np.sum(F1)                            # Target energy
El = 0.5 * E                              # Noise-region energy
F = np.abs(np.sqrt(F1))                   # Target amplitude
F = np.pad(F, ((pad_each, pad_each), (pad_each, pad_each)), mode="constant")  # Center SR in the work grid
nn, mm = F.shape

#%% ---------- Band-limitation masks ----------
bandlim_spe = cp.zeros((nn, mm), dtype=cp.float32)             # SLM aperture (central HOLOGRAM_SIZE)
bandlim_spe[ap0:ap0+HOLOGRAM_SIZE, ap0:ap0+HOLOGRAM_SIZE] = 1.0
bandlim_in = cp.zeros((nn, mm), dtype=cp.float32)              # Signal region SR (central 512)
bandlim_in[(nn-n)//2:(nn+n)//2, (mm-m)//2:(mm+m)//2] = 1.0
bandlim_ou = 1.0 - bandlim_in                                  # Noise region NR

# SR crop indices
sr_r0, sr_r1 = (nn-n)//2, (nn+n)//2
sr_c0, sr_c1 = (mm-m)//2, (mm+m)//2

# Incident Gaussian
w = 0.26                                                       # [mm] Beam waist
ox, oy = cp.meshgrid(cp.linspace(-dh*mm/2, dh*mm/2, mm), cp.linspace(-dh*nn/2, dh*nn/2, nn))
Gaussian = cp.exp(-((ox**2)+(oy**2))/w)
incident = Gaussian * bandlim_spe

F_gpu = cp.asarray(F)
E_gpu = cp.asarray(E)
El_gpu = cp.asarray(El)
F1_gpu = cp.asarray(F1)

# Poisson-solver denominator (eigenvalues of the discrete Laplacian, periodic BC),
# precomputed once for the SR size (n x m).
_ii = cp.arange(n).reshape(n, 1)
_jj = cp.arange(m).reshape(1, m)
_denom = 2 * cp.cos(2 * cp.pi * _ii / n) + 2 * cp.cos(2 * cp.pi * _jj / m) - 4
_denom[0, 0] = 1.0                                             # Avoid divide-by-zero (DC term)

def irrotational_phase(pha):
    """Least-squares irrotational (vortex-free) phase via an FFT Poisson solve.

    Returns the scalar-potential phase psi whose gradient best matches the wrapped
    gradient of `pha`. psi is curl-free by construction, so it contains no vortices.
    O(N^2 log N), independent of the vortex count.
    """
    # Wrapped forward differences (zero on the last row/column)
    dx = cp.zeros((n, m))
    dy = cp.zeros((n, m))
    dx[:, :-1] = cp.mod(pha[:, 1:] - pha[:, :-1] + cp.pi, 2 * cp.pi) - cp.pi
    dy[:-1, :] = cp.mod(pha[1:, :] - pha[:-1, :] + cp.pi, 2 * cp.pi) - cp.pi
    # Divergence of the wrapped gradient (Poisson right-hand side)
    rho = cp.zeros((n, m))
    rho[:, 0] = dx[:, 0]
    rho[:, 1:] = dx[:, 1:] - dx[:, :-1]
    rho[0, :] += dy[0, :]
    rho[1:, :] += dy[1:, :] - dy[:-1, :]
    # Solve nabla^2 psi = rho with a single FFT
    rho_hat = cp.fft.fft2(rho)
    rho_hat[0, 0] = 0.0                                        # Zero-mean solution
    psi = cp.real(cp.fft.ifft2(rho_hat / _denom))
    return psi

def final_reconstruction(E2_k):
    """Final reconstructed SR intensity and phase from the last SLM-plane field (phase-only hologram)."""
    An = cp.angle(E2_k)
    hologram = incident * cp.exp(1j * An)
    Rec = cp.fft.fftshift(cp.fft.ifft2(cp.fft.fftshift(hologram)))
    Rec_sr = Rec[sr_r0:sr_r1, sr_c0:sr_c1]
    I_final = cp.abs(Rec_sr) ** 2
    I_final = E_gpu * I_final / (cp.sum(I_final) + 1e-12)
    P_final = cp.mod(cp.angle(Rec_sr), 2 * cp.pi)             # Reconstructed phase at focal plane
    return cp.asnumpy(I_final), cp.asnumpy(P_final)

#%% ---------- Single run with the relaxed projector ----------
def run_alpha(alpha):
    """Alternative projection with the relaxed irrotational projector applied every iteration."""
    cp.random.seed(seed)
    phi = cp.exp(1j * 2 * cp.pi * cp.random.rand(nn, mm))      # Initial random phase
    amp = cp.random.rand(nn, mm)                              # Initial noise-region amplitude

    RMSE = np.zeros(loop)                                     # SR-only RMSE per iteration
    NUM = np.zeros(loop, dtype=int)                          # Natural vortex count per iteration

    print(f"--- Running alpha={alpha:.2f} ---", flush=True)
    t_run = time.perf_counter()
    for i in range(1, loop):
        amp = bandlim_in * F_gpu + bandlim_ou * amp           # Amplitude constraint in target plane
        E1 = amp * phi
        E2 = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(E1)))  # Forward FFT to SLM plane
        E2_ave = cp.sqrt((E_gpu + El_gpu) * incident ** 2 / cp.sum(incident ** 2))
        E2_k = E2_ave * cp.exp(1j * cp.angle(E2))             # SLM amplitude constraint, keep phase
        es = cp.fft.fftshift(cp.fft.ifft2(cp.fft.fftshift(E2_k)))  # Inverse FFT to target plane

        amp = cp.abs(es)
        amp_in = bandlim_in * amp
        amp_ou = bandlim_ou * amp
        amp = cp.sqrt(E_gpu) * (amp_in / (cp.sqrt(cp.sum(amp_in ** 2)) + 1e-12)) \
            + cp.sqrt(El_gpu) * (amp_ou / (cp.sqrt(cp.sum(amp_ou ** 2)) + 1e-12))

        # SR-only RMSE (crop to SR, renormalize to target energy E)
        I = amp[sr_r0:sr_r1, sr_c0:sr_c1] ** 2
        I = E_gpu * I / (cp.sum(I) + 1e-12)
        Diff = I - F1_gpu
        RMSE[i] = float(cp.sqrt(cp.mean(Diff ** 2)).get())

        # Natural vortex count on the propagated object phase angle(es) in the SR
        pha = cp.angle(es)
        pha_crop = pha[sr_r0:sr_r1, sr_c0:sr_c1]
        po, ne = function_vortex_detection_accegpu(pha_crop, dh, use_cupy=True)
        NUM[i] = po + ne

        # Relaxed irrotational projector on the SR crop, applied every iteration
        if alpha > 0.0:
            psi = irrotational_phase(pha_crop)                                      # Curl-free (vortex-free) phase
            psi = psi + cp.angle(cp.sum(cp.exp(1j * (pha_crop - psi))))             # Align global offset
            field = (1 - alpha) * cp.exp(1j * pha_crop) + alpha * cp.exp(1j * psi)  # Complex blend
            pha_new = pha.copy()
            pha_new[sr_r0:sr_r1, sr_c0:sr_c1] = cp.angle(field)
            phi = cp.exp(1j * pha_new)
        else:
            phi = cp.exp(1j * pha)                            # Baseline: no annihilation

        # Progress feedback
        if i % 25 == 0 or i == 1:
            elapsed = time.perf_counter() - t_run
            eta = elapsed / i * (loop - i)
            print(f"  alpha={alpha:.2f}  iter {i:4d}/{loop-1}  RMSE={RMSE[i]:.5f}  "
                  f"vortices={NUM[i]:5d}  elapsed={elapsed:.1f}s  ETA={eta:.0f}s", flush=True)

    I_final, P_final = final_reconstruction(E2_k)
    return RMSE, NUM, I_final, P_final

#%% ---------- Paper method (periodic arctan2 VAH), fair settled comparison ----------
def run_paper_periodic(x):
    """Paper method with vortex annihilation applied every x iterations.

    Fairness: the run ends (x-1) iterations after the LAST annihilation
    (total = n_cycles * x, measured at index total-1), so GS has re-converged and we
    never measure right after an annihilation. The vortex count is measured consistently
    on angle(es) (the natural / shipped-reconstruction phase), like the projector, so both
    methods are evaluated at a settled GS state.
    """
    n_cycles = max(2, round(loop / x))
    total = n_cycles * x                                      # end index total-1 is settled
    cp.random.seed(seed)
    phi = cp.exp(1j * 2 * cp.pi * cp.random.rand(nn, mm))
    amp = cp.random.rand(nn, mm)
    RMSE = np.zeros(total)
    NUM = np.zeros(total, dtype=int)
    print(f"--- Paper periodic x={x} (total={total}, settle window={x-1}) ---", flush=True)
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
        Diff = I - F1_gpu
        RMSE[i] = float(cp.sqrt(cp.mean(Diff ** 2)).get())

        pha = cp.angle(es)
        pha_crop = pha[sr_r0:sr_r1, sr_c0:sr_c1]
        po, ne = function_vortex_detection_accegpu(pha_crop, dh, use_cupy=True)  # natural count (consistent)
        NUM[i] = po + ne

        if i % x == 0:                                        # periodic annihilation
            pha_vfree_crop = function_vortex_elimination_accegpu(pha_crop, dh, use_cupy=True, gather_output=False)
            pha_new = pha.copy()
            pha_new[sr_r0:sr_r1, sr_c0:sr_c1] = pha_vfree_crop
            phi = cp.exp(1j * pha_new)
        else:
            phi = cp.exp(1j * pha)

    I_final, P_final = final_reconstruction(E2_k)
    return RMSE, NUM, I_final, P_final

#%% ---------- Sweep over alpha + paper period sweep ----------
results = {}
for alpha in alpha_list:
    t0 = time.perf_counter()
    RMSE, NUM, I_final, P_final = run_alpha(alpha)
    results[alpha] = (RMSE, NUM, I_final, P_final)
    print(f"alpha={alpha:.2f}  final RMSE={RMSE[-1]:.5f}  final vortices={NUM[-1]:5d}  ({time.perf_counter()-t0:.1f}s)")

paper_results = {}
for x in paper_x_list:
    t0 = time.perf_counter()
    RMSE, NUM, I_final, P_final = run_paper_periodic(x)
    paper_results[x] = (RMSE, NUM, I_final, P_final)
    print(f"paper x={x:3d}  settled RMSE={RMSE[-1]:.5f}  settled vortices={NUM[-1]:5d}  ({time.perf_counter()-t0:.1f}s)")

# Best paper config = lowest settled RMSE
best_x = min(paper_x_list, key=lambda xx: paper_results[xx][0][-1])
print(f"Best paper period x={best_x}  (settled RMSE={paper_results[best_x][0][-1]:.5f}, "
      f"vortices={paper_results[best_x][1][-1]})")

#%% ---------- Results table (simple markdown, regenerated each run) ----------
target_name = os.path.splitext(os.path.basename(input_tiff))[0]
tbl = [f"# Alpha-blend sweep — {target_name}",
       f"geometry {HOLOGRAM_SIZE}/{WORK_SIZE}/{SR_SIZE}, loop={loop}, floor={target_floor_rel}",
       "", "| method | final RMSE (SR) | final vortices |", "|---|---|---|"]
for a in alpha_list:
    R, N, _, _ = results[a]
    tbl.append(f"| proj alpha={a} | {R[-1]:.4f} | {N[-1]} |")
for x in paper_x_list:
    R, N, _, _ = paper_results[x]
    tbl.append(f"| paper x={x} | {R[-1]:.4f} | {N[-1]} |")
table_md = "\n".join(tbl) + "\n"
table_path = os.path.join(script_dir, f"results_alpha_{target_name}.md")
with open(table_path, "w", encoding="utf-8") as fh:
    fh.write(table_md)
print(f"\nResults table -> {table_path}")

#%% ---------- Plots (all figures built, then a single plt.show() at the end) ----------
best_paper_RMSE, best_paper_NUM, best_paper_I, best_paper_P = paper_results[best_x]
alpha_show = 0.5 if 0.5 in results else alpha_list[-1]
alpha_show = 1
baseline_key = 0.0 if 0.0 in results else alpha_list[0]

# Vortex count per iteration
fig = plt.figure()
for alpha in alpha_list:
    plt.plot(results[alpha][1][1:], label=f"proj alpha={alpha}")
plt.plot(best_paper_NUM[1:], "k--", label=f"paper x={best_x}")
plt.xlabel("Iteration")
plt.ylabel("Natural vortex count (SR)")
plt.title("Vortex count vs iteration: Poisson projector vs paper (fair)")
plt.legend()

# RMSE per iteration
fig = plt.figure()
for alpha in alpha_list:
    plt.plot(results[alpha][0][1:], label=f"proj alpha={alpha}")
plt.plot(best_paper_RMSE[1:], "k--", label=f"paper x={best_x}")
plt.xlabel("Iteration")
plt.ylabel("RMSE (SR)")
plt.title("RMSE vs iteration: Poisson projector vs paper (fair)")
plt.legend()

# Trade-off: final vortices vs final RMSE (settled)
fig = plt.figure()
for alpha in alpha_list:
    RMSE, NUM, _, _ = results[alpha]
    plt.scatter(RMSE[-1], NUM[-1], c="C0")
    plt.annotate(f"a={alpha}", (RMSE[-1], NUM[-1]))
for x in paper_x_list:
    RMSE, NUM, _, _ = paper_results[x]
    plt.scatter(RMSE[-1], NUM[-1], marker="x", c="k")
    plt.annotate(f"x={x}", (RMSE[-1], NUM[-1]))
plt.xlabel("Final settled RMSE (SR)")
plt.ylabel("Final settled vortex count (SR)")
plt.title("Trade-off: singularities vs RMSE (settled)")

# Reconstructed intensity: target vs baseline vs best paper vs projector
int_panels = [
    ("Target", F1),
    ("Baseline (alpha=0)", results[baseline_key][2]),
    (f"Paper (x={best_x})", best_paper_I),
    (f"Poisson projector (alpha={alpha_show})", results[alpha_show][2]),
]
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, (title, img) in zip(axes, int_panels):
    ax.imshow(img, cmap="gray")
    ax.set_title(title)
    ax.axis("off")
plt.suptitle("Reconstructed intensity (SR)")
plt.tight_layout()

# Reconstructed phase at focal plane: baseline vs best paper vs projector
pha_panels = [
    ("Baseline (alpha=0)", results[baseline_key][3]),
    (f"Paper (x={best_x})", best_paper_P),
    (f"Poisson projector (alpha={alpha_show})", results[alpha_show][3]),
]
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
im = None
for ax, (title, img) in zip(axes, pha_panels):
    im = ax.imshow(img, cmap="hsv")
    ax.set_title(title)
    ax.axis("off")
fig.colorbar(im, ax=axes, fraction=0.025, label="Phase (rad)")
plt.suptitle("Reconstructed phase at focal plane (SR)")

plt.show()

# %%
