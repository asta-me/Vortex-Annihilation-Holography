#%%
"""
Neumann-vs-periodic test: relaxed irrotational PROJECTOR for vortex annihilation.

This is EXACTLY vah_projector_fft_test.py (relaxed irrotational projector applied every
iteration on the SR crop, swept over alpha, compared against the paper's periodic
arctan2 elimination), with ONE addition: a second irrotational solver that uses
**Neumann (zero-flux) boundary conditions via a DCT Poisson solve** instead of the
**periodic boundary conditions via an FFT Poisson solve**.

Why
---
The variational problem behind the least-squares irrotational phase produces the NATURAL
Neumann boundary condition d(psi)/dn = g.n (see TEORIA_PROIETTORE_IRROTAZIONALE.md,
Appendix A/G). We originally solved it with a periodic FFT (+ guard padding) purely for
FFT speed. But the DCT-II diagonalizes the Laplacian with Neumann BCs at the SAME
O(N log N) cost (it is an FFT on a symmetrized signal) and needs NO guard padding, so we
can have the CORRECT boundary condition for free. This script measures whether it
actually changes the result on our targets.

The two solvers are otherwise identical (same wrapped-gradient RHS, same DC gauge, same
complex-field blend). Everything else (geometry, band-limit, GS loop, RMSE, vortex
detection, paper comparison) is copied verbatim from vah_projector_fft_test.py.

    solver = "fft"  -> periodic BC  (eigvals 2cos(2pi i/n)+2cos(2pi j/m)-4)
    solver = "dct"  -> Neumann  BC  (eigvals (2cos(pi i/n)-2)+(2cos(pi j/m)-2))

Env: vortex (conda), GPU (CuPy).
"""

#%% ---------- Imports ----------
import time
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cupy as cp
from cupyx.scipy.fft import dctn, idctn
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
alpha_list = [0.0, 1.0]                    # Baseline vs full projector (enough to expose the BC effect)
paper_x_list = [100]                       # Paper annihilation period (settled reference)
seed = 42                                 # Reproducibility
target_floor_rel = 5e-3                   # Relative floor wrt max(F1): prevents fully dark pixels

#%% ---------- Geometry (fix the hologram size and the SR fraction; the rest is derived) ----------
# Chen band-limit / 2x oversampling: the SLM aperture (the "hologram") is HALF the
# computational grid. So fixing the hologram size fixes the grid. Then you choose what
# FRACTION of the grid the image (signal region) occupies. NB: the 2x oversampling here
# comes only from aperture = grid/2 (implicit spectral zero-padding), NOT from an explicit
# sinc-interpolation of the SLM pattern.
HOLOGRAM_SIZE = 384                        # SLM aperture side [px] (= half the work grid; 2x oversampling)
WORK_SIZE = 2 * HOLOGRAM_SIZE             # computational grid side [px] (= 2x aperture)
SR_FRACTION = 2 / 3                       # signal-region side as a fraction of WORK_SIZE (image size)
SR_SIZE = int(round(SR_FRACTION * WORK_SIZE))
SR_SIZE -= SR_SIZE % 2                    # keep even so the SR is centered exactly in the grid
assert 0 < SR_SIZE <= WORK_SIZE, "SR_FRACTION out of range: signal region must fit the grid"
# Defaults above (384 / 768 / 512) reproduce the previous fixed geometry.

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

_img = np.array(Image.open(input_tiff_path))
F1 = (_img[..., 0] if _img.ndim == 3 else _img).astype(np.float32)  # first channel for RGB/RGBA
F1 = cv2.resize(F1, (SR_SIZE, SR_SIZE), interpolation=cv2.INTER_AREA)  # image -> signal-region size
# Reuse Marco's floor idea without changing this script's global intensity scale.
F1 = np.maximum(F1, target_floor_rel * (np.max(F1) + 1e-12))
n, m = F1.shape                          # = (SR_SIZE, SR_SIZE); the Poisson solve runs on this crop
E = np.sum(F1)                            # Target energy
El = 0.5 * E                              # Noise-region energy
F = np.abs(np.sqrt(F1))                   # Target amplitude
pad_each = (WORK_SIZE - SR_SIZE) // 2     # center the SR inside the work grid
F = np.pad(F, ((pad_each, pad_each), (pad_each, pad_each)), mode="constant")
nn, mm = F.shape                          # = (WORK_SIZE, WORK_SIZE)

#%% ---------- Band-limitation masks ----------
ap0 = (WORK_SIZE - HOLOGRAM_SIZE) // 2
bandlim_spe = cp.zeros((nn, mm), dtype=cp.float32)             # SLM aperture (central HOLOGRAM_SIZE)
bandlim_spe[ap0:ap0 + HOLOGRAM_SIZE, ap0:ap0 + HOLOGRAM_SIZE] = 1.0
bandlim_in = cp.zeros((nn, mm), dtype=cp.float32)             # Signal region SR (central SR_SIZE)
bandlim_in[pad_each:pad_each + SR_SIZE, pad_each:pad_each + SR_SIZE] = 1.0
bandlim_ou = 1.0 - bandlim_in                                  # Noise region NR

# SR crop indices
sr_r0, sr_r1 = pad_each, pad_each + SR_SIZE
sr_c0, sr_c1 = pad_each, pad_each + SR_SIZE

# Incident Gaussian
w = 0.26                                                       # [mm] Beam waist
ox, oy = cp.meshgrid(cp.linspace(-dh*mm/2, dh*mm/2, mm), cp.linspace(-dh*nn/2, dh*nn/2, nn))
Gaussian = cp.exp(-((ox**2)+(oy**2))/w)
incident = Gaussian * bandlim_spe

F_gpu = cp.asarray(F)
E_gpu = cp.asarray(E)
El_gpu = cp.asarray(El)
F1_gpu = cp.asarray(F1)

# ---------- Poisson-solver denominators (Laplacian eigenvalues) ----------
_ii = cp.arange(n).reshape(n, 1)
_jj = cp.arange(m).reshape(1, m)

# (a) PERIODIC BC (FFT): eigenvalues of the 5-point Laplacian on a torus.
_denom_fft = 2 * cp.cos(2 * cp.pi * _ii / n) + 2 * cp.cos(2 * cp.pi * _jj / m) - 4
_denom_fft[0, 0] = 1.0                                         # Avoid divide-by-zero (DC term)

# (b) NEUMANN BC (DCT-II): eigenvalues of the zero-flux Laplacian (Ghiglia-Romero).
_denom_dct = (2 * cp.cos(cp.pi * _ii / n) - 2) + (2 * cp.cos(cp.pi * _jj / m) - 2)
_denom_dct[0, 0] = 1.0                                         # Avoid divide-by-zero (DC term)


def _wrapped_divergence(pha):
    """Divergence rho = div( wrap(grad pha) ), the shared Poisson right-hand side.

    Wrapped forward differences (zero on the last row/column), then the discrete
    divergence with out-of-domain edges contributing 0. Identical to the RHS used by
    both the periodic and the Neumann solve.
    """
    dx = cp.zeros((n, m))
    dy = cp.zeros((n, m))
    dx[:, :-1] = cp.mod(pha[:, 1:] - pha[:, :-1] + cp.pi, 2 * cp.pi) - cp.pi
    dy[:-1, :] = cp.mod(pha[1:, :] - pha[:-1, :] + cp.pi, 2 * cp.pi) - cp.pi
    rho = cp.zeros((n, m))
    rho[:, 0] = dx[:, 0]
    rho[:, 1:] = dx[:, 1:] - dx[:, :-1]
    rho[0, :] += dy[0, :]
    rho[1:, :] += dy[1:, :] - dy[:-1, :]
    return rho


def irrotational_phase_fft(pha):
    """Least-squares irrotational (vortex-free) phase, PERIODIC BC via a single FFT.

    Returns the scalar-potential phase psi whose gradient best matches the wrapped
    gradient of `pha`. psi is curl-free by construction, so it contains no vortices.
    O(N^2 log N), independent of the vortex count. (== vah_projector_fft_test.py)
    """
    rho = _wrapped_divergence(pha)
    rho_hat = cp.fft.fft2(rho)
    rho_hat[0, 0] = 0.0                                        # Zero-mean solution
    psi = cp.real(cp.fft.ifft2(rho_hat / _denom_fft))
    return psi


def irrotational_phase_dct(pha):
    """Least-squares irrotational (vortex-free) phase, NEUMANN BC via a DCT-II Poisson
    solve (zero-flux boundaries; the natural BC of the variational problem).

    Same wrapped-gradient RHS and DC gauge as the FFT solver, but the Laplacian is
    diagonalized in the DCT-II basis (Neumann) instead of the FFT basis (periodic).
    Also O(N^2 log N), and needs NO guard padding. (new1 in ../../Vortex_patching)
    """
    rho = _wrapped_divergence(pha)
    rho_hat = dctn(rho, type=2, norm="ortho")
    phi_hat = rho_hat / _denom_dct
    phi_hat[0, 0] = 0.0                                        # Gauge: fix the global offset
    psi = idctn(phi_hat, type=2, norm="ortho")
    return psi


_SOLVERS = {"fft": irrotational_phase_fft, "dct": irrotational_phase_dct}
_SOLVER_LABEL = {"fft": "periodic/FFT", "dct": "Neumann/DCT"}


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

#%% ---------- Single run with the relaxed projector (solver-parametrized) ----------
def run_alpha(alpha, solver="fft"):
    """Alternative projection with the relaxed irrotational projector applied every
    iteration. `solver` selects the Poisson boundary condition:
        "fft" -> periodic BC, "dct" -> Neumann BC.
    """
    solve_fn = _SOLVERS[solver]
    cp.random.seed(seed)
    phi = cp.exp(1j * 2 * cp.pi * cp.random.rand(nn, mm))      # Initial random phase
    amp = cp.random.rand(nn, mm)                              # Initial noise-region amplitude

    RMSE = np.zeros(loop)                                     # SR-only RMSE per iteration
    NUM = np.zeros(loop, dtype=int)                          # Natural vortex count per iteration

    print(f"--- Running solver={solver} ({_SOLVER_LABEL[solver]})  alpha={alpha:.2f} ---", flush=True)
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
            psi = solve_fn(pha_crop)                                                # Curl-free (vortex-free) phase
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
            print(f"  [{solver}] alpha={alpha:.2f}  iter {i:4d}/{loop-1}  RMSE={RMSE[i]:.5f}  "
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

#%% ---------- Sweep over alpha for BOTH solvers + paper period sweep ----------
# results_fft : periodic BC over the full alpha_list (alpha=0 is the shared baseline).
# results_dct : Neumann BC over alpha > 0 (alpha=0 would be identical to the FFT baseline).
results_fft = {}
for alpha in alpha_list:
    t0 = time.perf_counter()
    RMSE, NUM, I_final, P_final = run_alpha(alpha, solver="fft")
    results_fft[alpha] = (RMSE, NUM, I_final, P_final)
    print(f"[fft] alpha={alpha:.2f}  final RMSE={RMSE[-1]:.5f}  final vortices={NUM[-1]:5d}  ({time.perf_counter()-t0:.1f}s)")

results_dct = {}
for alpha in alpha_list:
    if alpha == 0.0:
        continue                                             # baseline is solver-independent (reuse fft)
    t0 = time.perf_counter()
    RMSE, NUM, I_final, P_final = run_alpha(alpha, solver="dct")
    results_dct[alpha] = (RMSE, NUM, I_final, P_final)
    print(f"[dct] alpha={alpha:.2f}  final RMSE={RMSE[-1]:.5f}  final vortices={NUM[-1]:5d}  ({time.perf_counter()-t0:.1f}s)")

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

#%% ---------- Console summary: FFT vs DCT head-to-head ----------
print("\n" + "=" * 64)
print(f"{'alpha':>6}{'RMSE fft':>12}{'RMSE dct':>12}{'vort fft':>10}{'vort dct':>10}")
print("-" * 64)
for alpha in alpha_list:
    if alpha == 0.0:
        r_f = results_fft[alpha]
        print(f"{alpha:>6.2f}{r_f[0][-1]:>12.5f}{'(base)':>12}{r_f[1][-1]:>10d}{'(base)':>10}")
        continue
    r_f = results_fft[alpha]
    r_d = results_dct[alpha]
    print(f"{alpha:>6.2f}{r_f[0][-1]:>12.5f}{r_d[0][-1]:>12.5f}{r_f[1][-1]:>10d}{r_d[1][-1]:>10d}")
print("=" * 64)

#%% ---------- Results table (simple markdown, regenerated each run) ----------
target_name = os.path.splitext(os.path.basename(input_tiff))[0]
tbl = [f"# FFT vs DCT (periodic vs Neumann BC) — {target_name}",
       f"geometry {HOLOGRAM_SIZE}/{WORK_SIZE}/{SR_SIZE}, loop={loop}",
       "", "| alpha | RMSE fft | RMSE dct | vort fft | vort dct |", "|---|---|---|---|---|"]
for a in alpha_list:
    rf = results_fft[a]
    if a == 0.0:
        tbl.append(f"| {a} (base) | {rf[0][-1]:.4f} | - | {rf[1][-1]} | - |")
    else:
        rd = results_dct[a]
        tbl.append(f"| {a} | {rf[0][-1]:.4f} | {rd[0][-1]:.4f} | {rf[1][-1]} | {rd[1][-1]} |")
for x in paper_x_list:
    rp = paper_results[x]
    tbl.append(f"| paper x={x} | {rp[0][-1]:.4f} | - | {rp[1][-1]} | - |")
table_md = "\n".join(tbl) + "\n"
table_path = os.path.join(script_dir, f"results_neumann_{target_name}.md")
with open(table_path, "w", encoding="utf-8") as fh:
    fh.write(table_md)
print(f"\nResults table -> {table_path}")

#%% ---------- Plots (all figures built, then a single plt.show() at the end) ----------
best_paper_RMSE, best_paper_NUM, best_paper_I, best_paper_P = paper_results[best_x]
alpha_show = 1.0 if 1.0 in results_fft else alpha_list[-1]   # strongest BC contrast (pure projector)
baseline_key = 0.0 if 0.0 in results_fft else alpha_list[0]

# Vortex count per iteration (fft solid, dct dashed, paper black)
fig = plt.figure(figsize=(9, 5))
for k, alpha in enumerate(alpha_list):
    plt.plot(results_fft[alpha][1][1:], color=f"C{k}", label=f"fft a={alpha}")
    if alpha in results_dct:
        plt.plot(results_dct[alpha][1][1:], color=f"C{k}", ls="--", label=f"dct a={alpha}")
plt.plot(best_paper_NUM[1:], "k:", label=f"paper x={best_x}")
plt.xlabel("Iteration")
plt.ylabel("Natural vortex count (SR)")
plt.title("Vortex count: periodic (solid) vs Neumann (dashed) vs paper")
plt.legend(fontsize=8, ncol=2)

# RMSE per iteration (fft solid, dct dashed, paper black)
fig = plt.figure(figsize=(9, 5))
for k, alpha in enumerate(alpha_list):
    plt.plot(results_fft[alpha][0][1:], color=f"C{k}", label=f"fft a={alpha}")
    if alpha in results_dct:
        plt.plot(results_dct[alpha][0][1:], color=f"C{k}", ls="--", label=f"dct a={alpha}")
plt.plot(best_paper_RMSE[1:], "k:", label=f"paper x={best_x}")
plt.xlabel("Iteration")
plt.ylabel("RMSE (SR)")
plt.title("RMSE: periodic (solid) vs Neumann (dashed) vs paper")
plt.legend(fontsize=8, ncol=2)

# Trade-off: final vortices vs final RMSE (settled)
fig = plt.figure(figsize=(7, 5))
for alpha in alpha_list:
    RMSE, NUM, _, _ = results_fft[alpha]
    plt.scatter(RMSE[-1], NUM[-1], c="C0", marker="o")
    plt.annotate(f"fft {alpha}", (RMSE[-1], NUM[-1]), fontsize=8)
for alpha in results_dct:
    RMSE, NUM, _, _ = results_dct[alpha]
    plt.scatter(RMSE[-1], NUM[-1], c="C1", marker="s")
    plt.annotate(f"dct {alpha}", (RMSE[-1], NUM[-1]), fontsize=8)
for x in paper_x_list:
    RMSE, NUM, _, _ = paper_results[x]
    plt.scatter(RMSE[-1], NUM[-1], marker="x", c="k")
    plt.annotate(f"x={x}", (RMSE[-1], NUM[-1]), fontsize=8)
plt.xlabel("Final settled RMSE (SR)")
plt.ylabel("Final settled vortex count (SR)")
plt.title("Trade-off: singularities vs RMSE  (o=fft, s=dct, x=paper)")

# FFT vs DCT summary bars: final RMSE and final vortices per alpha (alpha > 0)
alphas_pos = [a for a in alpha_list if a > 0.0]
xpos = np.arange(len(alphas_pos))
width = 0.38
fig, (axr, axv) = plt.subplots(1, 2, figsize=(12, 4.5))
axr.bar(xpos - width/2, [results_fft[a][0][-1] for a in alphas_pos], width, label="fft (periodic)")
axr.bar(xpos + width/2, [results_dct[a][0][-1] for a in alphas_pos], width, label="dct (Neumann)")
axr.axhline(results_fft[baseline_key][0][-1], color="grey", ls=":", label="baseline")
axr.axhline(best_paper_RMSE[-1], color="k", ls="--", label=f"paper x={best_x}")
axr.set_xticks(xpos); axr.set_xticklabels([f"a={a}" for a in alphas_pos])
axr.set_ylabel("Final RMSE (SR)"); axr.set_title("Final RMSE: periodic vs Neumann"); axr.legend(fontsize=8)
axv.bar(xpos - width/2, [results_fft[a][1][-1] for a in alphas_pos], width, label="fft (periodic)")
axv.bar(xpos + width/2, [results_dct[a][1][-1] for a in alphas_pos], width, label="dct (Neumann)")
axv.set_xticks(xpos); axv.set_xticklabels([f"a={a}" for a in alphas_pos])
axv.set_ylabel("Final vortex count (SR)"); axv.set_title("Final vortices: periodic vs Neumann"); axv.legend(fontsize=8)
plt.tight_layout()

# Reconstructed intensity: target vs baseline vs paper vs projector(fft) vs projector(dct)
int_panels = [
    ("Target", F1),
    ("Baseline (alpha=0)", results_fft[baseline_key][2]),
    (f"Paper (x={best_x})", best_paper_I),
    (f"Projector FFT (a={alpha_show})", results_fft[alpha_show][2]),
    (f"Projector DCT (a={alpha_show})", results_dct[alpha_show][2]),
]
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for ax, (title, img) in zip(axes, int_panels):
    ax.imshow(img, cmap="gray")
    ax.set_title(title, fontsize=9)
    ax.axis("off")
plt.suptitle("Reconstructed intensity (SR)")
plt.tight_layout()

# Reconstructed phase at focal plane: baseline vs paper vs projector(fft) vs projector(dct)
pha_panels = [
    ("Baseline (alpha=0)", results_fft[baseline_key][3]),
    (f"Paper (x={best_x})", best_paper_P),
    (f"Projector FFT (a={alpha_show})", results_fft[alpha_show][3]),
    (f"Projector DCT (a={alpha_show})", results_dct[alpha_show][3]),
]
fig, axes = plt.subplots(1, 4, figsize=(17, 4))
im = None
for ax, (title, img) in zip(axes, pha_panels):
    im = ax.imshow(img, cmap="hsv")
    ax.set_title(title, fontsize=9)
    ax.axis("off")
fig.colorbar(im, ax=axes, fraction=0.025, label="Phase (rad)")
plt.suptitle("Reconstructed phase at focal plane (SR)")

plt.show()

# %%
