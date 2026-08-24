#%%
"""
SLM-plane vs target-plane irrotational PROJECTOR — comparison test.  [FAILED EXPERIMENT]

STATUS
------
FAILED. Applying the irrotational projector on the SLM-plane phase WRECKS the reconstruction
(RMSE explodes, monotonically worse with alpha). Kept for the record. The projector must be
applied ONLY on the target (image) plane. See RESULT and "Why it fails" below.

What we were testing
--------------------
`vah_ph_projector_test.py` applies the relaxed irrotational (vortex-free) projector on the
TARGET-plane phase (the SR crop of `es`). Hypothesis under test: since phase vortices also live
on the SLM-plane phase and propagate through the FFT, maybe for vortex conservation through
propagation we should project on the SLM plane too. This script runs BOTH variants under
identical initial conditions so they are directly comparable:

- plane = "target": projector on the target-plane phase angle(es), SR crop (central 512).
                    Identical to vah_ph_projector_test.py.
- plane = "slm":    projector on the SLM-plane phase angle(E2), APERTURE crop (central 384,
                    = bandlim_spe support). Since E2_k = E2_ave*exp(i*phase) and E2_ave is
                    nonzero only inside the aperture (incident*bandlim_spe), the aperture
                    crop is exactly the physically meaningful region on the SLM plane.

The projector itself is identical in both cases: least-squares irrotational phase psi via a
single-FFT Poisson solve, then a convex blend in the complex field domain
    field = (1-alpha)*exp(i*phase) + alpha*exp(i*psi).

Metrics tracked per iteration (both comparable across planes):
- RMSE on the SR (target plane).
- NUM      : natural vortex count on the target-plane SR phase angle(es).
- NUM_SLM  : natural vortex count on the SLM-plane aperture phase angle(E2).

Paper periodic arctan2 VAH (target-plane elimination) is included as a common reference.

RESULT (marmo.tif, 300 iters)
-----------------------------
    baseline (a=0):     RMSE   35.9
    target-proj a=0.5:  RMSE    8.9   vSR ~0        <- works (vortices are real defects here)
    slm-proj  a=0.25:   RMSE  104
    slm-proj  a=0.5:    RMSE  334
    slm-proj  a=0.75:   RMSE 1026
    slm-proj  a=1.0:    RMSE 1271                   <- all alphas hurt; more alpha = worse

Why it fails (physics)
----------------------
The SLM plane is the FOURIER plane of the hologram. Its phase angle(E2) is speckle-like and
carries a HUGE number of vortices (vSLM ~36000 vs vSR ~7600 at baseline), mostly in +/- pairs.
Those vortices are INTRINSIC to how the image is encoded in the Fourier phase, not defects.
What is (approximately) conserved under FFT propagation is the total ORBITAL ANGULAR MOMENTUM
(a weighted integral), NOT the vortex COUNT: +/- vortex pairs nucleate/annihilate freely
without changing the OAM. Forcing the SLM phase to be irrotational (curl-free) therefore
destroys the encoding and the reconstruction collapses. (The GS amplitude constraints in both
planes are also non-unitary, so even OAM is not strictly conserved in this loop.)

CONCLUSION: apply the irrotational projector ONLY on the target (image) plane. See the theory
doc TEORIA_PROIETTORE_IRROTAZIONALE.md, "Appendice H", for vortex-count vs OAM.

Env: vortex (conda), GPU (CuPy). Uses PIL + cv2 (no skimage; see repo note on cupy+skimage crash).
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
loop = 300                                # Iterations per run (2 planes x len(alpha_list) runs + paper)
alpha_list = [0.0, 0.25, 0.5, 0.75, 1.0]  # Relaxation knob sweep (0 = off, 1 = full irrotational)
paper_x_list = [50, 100]                  # Paper annihilation-period sweep (fair settled endpoint)
alpha_show = 1.0                          # Alpha used for the per-iteration comparison figures
seed = 42                                 # Reproducibility
target_floor_rel = 5e-3                   # Relative floor wrt max(F1): prevents fully dark pixels

#%% ---------- Import target (TIFF) ----------
input_tiff = "marmo.tif"  # Change this filename to run the same pipeline on another TIFF.

script_dir = os.path.dirname(os.path.abspath(__file__))
input_tiff_path = os.path.join(ALT_PROJ_DIR, input_tiff)
if not os.path.isfile(input_tiff_path):
    raise FileNotFoundError(f"TIFF file not found: {input_tiff_path}")

F1 = np.array(Image.open(input_tiff_path))
if F1.ndim == 3:
    F1 = F1[..., 0]                        # Deterministic for RGB/RGBA: first channel
F1 = F1.astype(np.float32)
F1 = cv2.resize(F1, (512, 512), interpolation=cv2.INTER_AREA)
F1 = np.maximum(F1, target_floor_rel * (np.max(F1) + 1e-12))
n, m = F1.shape
E = np.sum(F1)                            # Target energy
El = 0.5 * E                              # Noise-region energy
F = np.abs(np.sqrt(F1))                   # Target amplitude
F = np.pad(F, ((n//4, n//4), (m//4, m//4)), mode="constant")  # Pad 512 -> 768
nn, mm = F.shape

#%% ---------- Band-limitation masks ----------
bandlim_spe = cp.zeros((nn, mm), dtype=cp.float32)             # SLM aperture (central 384)
bandlim_spe[nn//4:3*nn//4, mm//4:3*mm//4] = 1.0
bandlim_in = cp.zeros((nn, mm), dtype=cp.float32)              # Signal region SR (central 512)
bandlim_in[(nn-n)//2:(nn+n)//2, (mm-m)//2:(mm+m)//2] = 1.0
bandlim_ou = 1.0 - bandlim_in                                  # Noise region NR

# Target-plane SR crop indices (central 512)
sr_r0, sr_r1 = (nn-n)//2, (nn+n)//2
sr_c0, sr_c1 = (mm-m)//2, (mm+m)//2

# SLM-plane aperture crop indices (central 384, = bandlim_spe support)
sp_r0, sp_r1 = nn//4, 3*nn//4
sp_c0, sp_c1 = mm//4, 3*mm//4
spn, spm = sp_r1 - sp_r0, sp_c1 - sp_c0

# Incident Gaussian
w = 0.26                                                       # [mm] Beam waist
ox, oy = cp.meshgrid(cp.linspace(-dh*mm/2, dh*mm/2, mm), cp.linspace(-dh*nn/2, dh*nn/2, nn))
Gaussian = cp.exp(-((ox**2)+(oy**2))/w)
incident = Gaussian * bandlim_spe

F_gpu = cp.asarray(F)
E_gpu = cp.asarray(E)
El_gpu = cp.asarray(El)
F1_gpu = cp.asarray(F1)

# Poisson-solver denominators (eigenvalues of the discrete Laplacian, periodic BC).
def _make_denom(nr, nc):
    ii = cp.arange(nr).reshape(nr, 1)
    jj = cp.arange(nc).reshape(1, nc)
    d = 2 * cp.cos(2 * cp.pi * ii / nr) + 2 * cp.cos(2 * cp.pi * jj / nc) - 4
    d[0, 0] = 1.0                                              # Avoid divide-by-zero (DC term)
    return d

_denom_sr = _make_denom(n, m)             # for the target-plane SR crop (512)
_denom_slm = _make_denom(spn, spm)        # for the SLM-plane aperture crop (384)


def irrotational_phase(pha, denom):
    """Least-squares irrotational (vortex-free) phase via an FFT Poisson solve.

    `denom` are the precomputed 5-point Laplacian eigenvalues for pha.shape (periodic BC).
    Returns the scalar-potential phase psi whose gradient best matches the wrapped gradient
    of `pha`; psi is curl-free by construction (no vortices). O(N^2 log N).
    """
    nr, nc = pha.shape
    dx = cp.zeros((nr, nc))
    dy = cp.zeros((nr, nc))
    dx[:, :-1] = cp.mod(pha[:, 1:] - pha[:, :-1] + cp.pi, 2 * cp.pi) - cp.pi
    dy[:-1, :] = cp.mod(pha[1:, :] - pha[:-1, :] + cp.pi, 2 * cp.pi) - cp.pi
    rho = cp.zeros((nr, nc))
    rho[:, 0] = dx[:, 0]
    rho[:, 1:] = dx[:, 1:] - dx[:, :-1]
    rho[0, :] += dy[0, :]
    rho[1:, :] += dy[1:, :] - dy[:-1, :]
    rho_hat = cp.fft.fft2(rho)
    rho_hat[0, 0] = 0.0                                        # Zero-mean solution
    return cp.real(cp.fft.ifft2(rho_hat / denom))


def _blend(crop, denom, alpha):
    """Relaxed irrotational projector on a phase crop: returns the blended phase crop."""
    psi = irrotational_phase(crop, denom)
    psi = psi + cp.angle(cp.sum(cp.exp(1j * (crop - psi))))     # Align global offset
    field = (1 - alpha) * cp.exp(1j * crop) + alpha * cp.exp(1j * psi)
    return cp.angle(field)


def final_reconstruction(E2_k):
    """Final reconstructed SR intensity and phase from the last SLM-plane field."""
    An = cp.angle(E2_k)
    hologram = incident * cp.exp(1j * An)
    Rec = cp.fft.fftshift(cp.fft.ifft2(cp.fft.fftshift(hologram)))
    Rec_sr = Rec[sr_r0:sr_r1, sr_c0:sr_c1]
    I_final = cp.abs(Rec_sr) ** 2
    I_final = E_gpu * I_final / (cp.sum(I_final) + 1e-12)
    P_final = cp.mod(cp.angle(Rec_sr), 2 * cp.pi)
    return cp.asnumpy(I_final), cp.asnumpy(P_final)


#%% ---------- Unified run: projector on "target" or "slm" plane ----------
def run_alpha(alpha, plane):
    """Alternative projection with the relaxed irrotational projector applied every iteration,
    either on the target-plane SR phase (plane='target') or the SLM-plane aperture phase
    (plane='slm'). All else identical (same RNG seed) so the two planes are comparable.
    """
    assert plane in ("target", "slm")
    cp.random.seed(seed)
    phi = cp.exp(1j * 2 * cp.pi * cp.random.rand(nn, mm))
    amp = cp.random.rand(nn, mm)

    RMSE = np.zeros(loop)
    NUM = np.zeros(loop, dtype=int)                            # target-plane SR vortex count
    NUM_SLM = np.zeros(loop, dtype=int)                        # SLM-plane aperture vortex count

    print(f"--- Running plane={plane}  alpha={alpha:.2f} ---", flush=True)
    t_run = time.perf_counter()
    for i in range(1, loop):
        amp = bandlim_in * F_gpu + bandlim_ou * amp           # Amplitude constraint in target plane
        E1 = amp * phi
        E2 = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(E1)))  # Forward FFT to SLM plane

        # SLM-plane natural vortex count (aperture crop), measured before any SLM projection
        slm_pha = cp.angle(E2)
        slm_crop = slm_pha[sp_r0:sp_r1, sp_c0:sp_c1]
        po_s, ne_s = function_vortex_detection_accegpu(slm_crop, dh, use_cupy=True)
        NUM_SLM[i] = po_s + ne_s

        # SLM-plane projector
        if plane == "slm" and alpha > 0.0:
            slm_pha_new = slm_pha.copy()
            slm_pha_new[sp_r0:sp_r1, sp_c0:sp_c1] = _blend(slm_crop, _denom_slm, alpha)
            slm_phase = slm_pha_new
        else:
            slm_phase = slm_pha

        E2_ave = cp.sqrt((E_gpu + El_gpu) * incident ** 2 / cp.sum(incident ** 2))
        E2_k = E2_ave * cp.exp(1j * slm_phase)                 # SLM amplitude constraint, chosen phase
        es = cp.fft.fftshift(cp.fft.ifft2(cp.fft.fftshift(E2_k)))  # Inverse FFT to target plane

        amp = cp.abs(es)
        amp_in = bandlim_in * amp
        amp_ou = bandlim_ou * amp
        amp = cp.sqrt(E_gpu) * (amp_in / (cp.sqrt(cp.sum(amp_in ** 2)) + 1e-12)) \
            + cp.sqrt(El_gpu) * (amp_ou / (cp.sqrt(cp.sum(amp_ou ** 2)) + 1e-12))

        # SR-only RMSE
        I = amp[sr_r0:sr_r1, sr_c0:sr_c1] ** 2
        I = E_gpu * I / (cp.sum(I) + 1e-12)
        RMSE[i] = float(cp.sqrt(cp.mean((I - F1_gpu) ** 2)).get())

        # Target-plane natural vortex count (SR crop)
        pha = cp.angle(es)
        pha_crop = pha[sr_r0:sr_r1, sr_c0:sr_c1]
        po, ne = function_vortex_detection_accegpu(pha_crop, dh, use_cupy=True)
        NUM[i] = po + ne

        # Target-plane projector
        if plane == "target" and alpha > 0.0:
            pha_new = pha.copy()
            pha_new[sr_r0:sr_r1, sr_c0:sr_c1] = _blend(pha_crop, _denom_sr, alpha)
            phi = cp.exp(1j * pha_new)
        else:
            phi = cp.exp(1j * pha)

        if i % 25 == 0 or i == 1:
            elapsed = time.perf_counter() - t_run
            eta = elapsed / i * (loop - i)
            print(f"  [{plane}] alpha={alpha:.2f}  iter {i:4d}/{loop-1}  RMSE={RMSE[i]:.5f}  "
                  f"vSR={NUM[i]:5d}  vSLM={NUM_SLM[i]:5d}  elapsed={elapsed:.1f}s  ETA={eta:.0f}s",
                  flush=True)

    I_final, P_final = final_reconstruction(E2_k)
    return RMSE, NUM, NUM_SLM, I_final, P_final


#%% ---------- Paper method (periodic arctan2 VAH, target-plane), reference ----------
def run_paper_periodic(x):
    """Paper method with vortex annihilation applied every x iterations on the target-plane SR."""
    n_cycles = max(2, round(loop / x))
    total = n_cycles * x
    cp.random.seed(seed)
    phi = cp.exp(1j * 2 * cp.pi * cp.random.rand(nn, mm))
    amp = cp.random.rand(nn, mm)
    RMSE = np.zeros(total)
    NUM = np.zeros(total, dtype=int)
    NUM_SLM = np.zeros(total, dtype=int)
    print(f"--- Paper periodic x={x} (total={total}) ---", flush=True)
    for i in range(1, total):
        amp = bandlim_in * F_gpu + bandlim_ou * amp
        E1 = amp * phi
        E2 = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(E1)))
        slm_crop = cp.angle(E2)[sp_r0:sp_r1, sp_c0:sp_c1]
        po_s, ne_s = function_vortex_detection_accegpu(slm_crop, dh, use_cupy=True)
        NUM_SLM[i] = po_s + ne_s

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

    I_final, P_final = final_reconstruction(E2_k)
    return RMSE, NUM, NUM_SLM, I_final, P_final


#%% ---------- Run sweeps ----------
results = {}   # (plane, alpha) -> (RMSE, NUM, NUM_SLM, I_final, P_final)
for plane in ("target", "slm"):
    for alpha in alpha_list:
        t0 = time.perf_counter()
        results[(plane, alpha)] = run_alpha(alpha, plane)
        RMSE, NUM, NUM_SLM, _, _ = results[(plane, alpha)]
        print(f"[{plane}] alpha={alpha:.2f}  final RMSE={RMSE[-1]:.5f}  "
              f"final vSR={NUM[-1]:5d}  final vSLM={NUM_SLM[-1]:5d}  ({time.perf_counter()-t0:.1f}s)")

paper_results = {}
for x in paper_x_list:
    t0 = time.perf_counter()
    paper_results[x] = run_paper_periodic(x)
    RMSE, NUM, NUM_SLM, _, _ = paper_results[x]
    print(f"paper x={x:3d}  settled RMSE={RMSE[-1]:.5f}  vSR={NUM[-1]:5d}  vSLM={NUM_SLM[-1]:5d}  "
          f"({time.perf_counter()-t0:.1f}s)")

best_x = min(paper_x_list, key=lambda xx: paper_results[xx][0][-1])
print(f"Best paper period x={best_x} (settled RMSE={paper_results[best_x][0][-1]:.5f})")

#%% ---------- Plots ----------
output_dir = os.path.join(script_dir, "output_test_onslm")
os.makedirs(output_dir, exist_ok=True)

best_paper = paper_results[best_x]
baseline = results[("target", 0.0)]  # alpha=0 is identical for both planes

# 1) RMSE vs iteration: target-proj vs slm-proj vs paper (at alpha_show)
fig = plt.figure()
plt.plot(baseline[0][1:], label="baseline (alpha=0)")
plt.plot(results[("target", alpha_show)][0][1:], label=f"target-proj (a={alpha_show})")
plt.plot(results[("slm", alpha_show)][0][1:], label=f"SLM-proj (a={alpha_show})")
plt.plot(best_paper[0][1:], "k--", label=f"paper x={best_x}")
plt.xlabel("Iteration"); plt.ylabel("RMSE (SR)")
plt.title(f"RMSE vs iteration — projector plane comparison (alpha={alpha_show})")
plt.legend(fontsize=8)
fig.savefig(os.path.join(output_dir, "rmse_compare.png"), dpi=150, bbox_inches="tight")
plt.show()

# 2) Target-plane SR vortex count vs iteration
fig = plt.figure()
plt.plot(baseline[1][1:], label="baseline (alpha=0)")
plt.plot(results[("target", alpha_show)][1][1:], label=f"target-proj (a={alpha_show})")
plt.plot(results[("slm", alpha_show)][1][1:], label=f"SLM-proj (a={alpha_show})")
plt.plot(best_paper[1][1:], "k--", label=f"paper x={best_x}")
plt.xlabel("Iteration"); plt.ylabel("Vortex count — target SR")
plt.title(f"Target-plane SR vortices vs iteration (alpha={alpha_show})")
plt.legend(fontsize=8)
fig.savefig(os.path.join(output_dir, "vortex_target_compare.png"), dpi=150, bbox_inches="tight")
plt.show()

# 3) SLM-plane aperture vortex count vs iteration
fig = plt.figure()
plt.plot(baseline[2][1:], label="baseline (alpha=0)")
plt.plot(results[("target", alpha_show)][2][1:], label=f"target-proj (a={alpha_show})")
plt.plot(results[("slm", alpha_show)][2][1:], label=f"SLM-proj (a={alpha_show})")
plt.plot(best_paper[2][1:], "k--", label=f"paper x={best_x}")
plt.xlabel("Iteration"); plt.ylabel("Vortex count — SLM aperture")
plt.title(f"SLM-plane aperture vortices vs iteration (alpha={alpha_show})")
plt.legend(fontsize=8)
fig.savefig(os.path.join(output_dir, "vortex_slm_compare.png"), dpi=150, bbox_inches="tight")
plt.show()

# 4) Trade-off: final RMSE vs final target-SR vortices, all (plane, alpha) + paper
fig = plt.figure()
colors = {"target": "C0", "slm": "C1"}
for (plane, alpha), (RMSE, NUM, NUM_SLM, _, _) in results.items():
    plt.scatter(RMSE[-1], NUM[-1], c=colors[plane])
    plt.annotate(f"{plane[:3]} a={alpha}", (RMSE[-1], NUM[-1]), fontsize=7)
for x in paper_x_list:
    RMSE, NUM, _, _, _ = paper_results[x]
    plt.scatter(RMSE[-1], NUM[-1], marker="x", c="k")
    plt.annotate(f"paper x={x}", (RMSE[-1], NUM[-1]), fontsize=7)
plt.xlabel("Final settled RMSE (SR)"); plt.ylabel("Final settled vortex count (target SR)")
plt.title("Trade-off: singularities vs RMSE (settled)")
fig.savefig(os.path.join(output_dir, "tradeoff.png"), dpi=150, bbox_inches="tight")
plt.show()

# 5) Reconstructed intensity panels
int_panels = [
    ("Target", F1),
    ("Baseline (a=0)", baseline[3]),
    (f"Paper (x={best_x})", best_paper[3]),
    (f"Target-proj (a={alpha_show})", results[("target", alpha_show)][3]),
    (f"SLM-proj (a={alpha_show})", results[("slm", alpha_show)][3]),
]
fig, axes = plt.subplots(1, len(int_panels), figsize=(4 * len(int_panels), 4))
for ax, (title, img) in zip(axes, int_panels):
    ax.imshow(img, cmap="gray"); ax.set_title(title, fontsize=9); ax.axis("off")
plt.suptitle("Reconstructed intensity (SR)")
plt.tight_layout()
fig.savefig(os.path.join(output_dir, "reconstruction_intensity.png"), dpi=150, bbox_inches="tight")
plt.show()

# 6) Numeric summary
print("\n==== Final settled metrics ====")
print(f"{'method':28s} {'RMSE':>10s} {'vSR':>7s} {'vSLM':>7s}")
print(f"{'baseline (a=0)':28s} {baseline[0][-1]:10.5f} {baseline[1][-1]:7d} {baseline[2][-1]:7d}")
for plane in ("target", "slm"):
    for alpha in alpha_list:
        if alpha == 0.0:
            continue
        RMSE, NUM, NUM_SLM, _, _ = results[(plane, alpha)]
        print(f"{f'{plane}-proj a={alpha}':28s} {RMSE[-1]:10.5f} {NUM[-1]:7d} {NUM_SLM[-1]:7d}")
for x in paper_x_list:
    RMSE, NUM, NUM_SLM, _, _ = paper_results[x]
    print(f"{f'paper x={x}':28s} {RMSE[-1]:10.5f} {NUM[-1]:7d} {NUM_SLM[-1]:7d}")

print(f"\nFigures saved to: {output_dir}")

# %%
