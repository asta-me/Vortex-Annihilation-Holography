#%%
"""
FINAL COMPARISON: paper vortex annihilation vs the irrotational (DCT/Neumann) projector.

Goal
----
Show, head to head and on the SAME geometry / GS loop, why the irrotational projector beats
the paper's arctan2 vortex annihilation. All methods evaluated on the SR crop:

  0. baseline         -- the classic Gerchberg-Saxton loop with NO vortex handling at all
                         (reference). Same GS iteration, annihilation never applied. Random init.
  1. paper            -- the paper's periodic arctan2 vortex elimination (period x). O(K*N^2)
                         per event (K = #vortices), so slow when vortices are many. Random init.
  2. proj_periodic    -- MY projector applied on the SAME schedule as the paper (period x, full
                         projection alpha=1). O(N log N) per event, independent of K. Should reach
                         the paper's quality but in far less time. Random init.
  3. proj_every       -- MY projector applied at EVERY iteration (alpha=1). Cheap enough to run
                         each iteration; compare time AND -- above all -- quality vs the paper.
                         Random init.
  4. proj_every_chen  -- same as (3) but started from a CHEN quadratic (lens) phase whose fringes
                         fill the pad margin (c* = 4*pi*pad = pi*SR in the canonical 3/4 geometry).
                         The optimized init drops RMSE further at no extra per-iteration cost.
  5. proj_every_filt  -- same as (3) but started from a FILTERED-random phase: a low-pass smooth
                         random field whose per-pixel gradient is Nyquist-capped (max|grad|<grad_cap*pi),
                         so it is single-valued AND alias-free -> ZERO vortices by construction, while
                         still filling the SLM aperture. Outperforms the random-init projector.

The projector solves the least-squares irrotational (vortex-free) phase with a DCT-II Poisson
solve (NEUMANN / zero-flux BC -- the natural BC of the variational problem, no guard padding).

Geometry is fully parametrized (HOLOGRAM_SIZE -> WORK grid -> SR fraction), inheriting the size
freedom of dct_vs_fft.py, so any image size / oversampling can be swept.

Env: vortex (conda), GPU (CuPy). PIL + cv2 (no skimage).
"""

#%% ---------- Imports ----------
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import cupy as cp
from cupyx.scipy.fft import dctn, idctn
import cv2
from PIL import Image

ALT_PROJ_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "1_Alternative_projection")
sys.path.insert(0, ALT_PROJ_DIR)
from function_vortex_detection_accegpu import function_vortex_detection_accegpu
from function_vortex_elimination_accegpu import function_vortex_elimination_accegpu

#%% ---------- Config ----------
dh = 0.00374                             # [mm] Pixel pitch
loop = 500                               # Iterations per run
trigger_x = 100                          # Annihilation period for the periodic methods (paper + proj_periodic)
alpha = 1.0                              # Full irrotational projection (alpha=1)
seed = 42                                # Reproducibility
target_floor_rel = 5e-3                  # Relative floor wrt max(F1): prevents fully dark pixels
# Parameters to be set for the filtered-random init (proj_every_filt):
corr_len_px = 20.0                       # [px] correlation length: empirical RMSE optimum (broad plateau ~16-31 px)
grad_cap = 0.9                           # Nyquist safety in (0,1): largest per-pixel phase step = grad_cap*pi < pi
COUNT_VORTICES = False                    # Diagnostic only; set False for clean timings (final count computed once, no vortex-vs-iter curve)

# Target image: uncomment ONE line (files live in ./targets/, or give an absolute path).
input_tiff = "marmo.tif"
# input_tiff = "object_grayscale_from_mat.tif"
# input_tiff = "Lenna.tif"
# input_tiff = "Baboon.tif"
input_tiff = "valentini.tif"

#%% ---------- Geometry (fix the hologram size and the SR fraction; the rest is derived) ----------
# Chen band-limit / 2x oversampling: SLM aperture = HALF the computational grid. Fixing the
# hologram size fixes the grid; SR_FRACTION sets what fraction of the grid the image occupies.
HOLOGRAM_SIZE = 384                       # SLM aperture side [px]  (default reproduces 384/768/512)
HOLOGRAM_SIZE = 1080                       # SLM aperture side [px]  (default reproduces 384/768/512)
WORK_SIZE = 2 * HOLOGRAM_SIZE            # computational grid side [px] (= 2x aperture)
SR_FRACTION = 2 / 3                      # signal-region side as a fraction of WORK_SIZE (image size)
SR_SIZE = int(round(SR_FRACTION * WORK_SIZE))
SR_SIZE -= SR_SIZE % 2                   # keep even so the SR is centered exactly in the grid
assert 0 < SR_SIZE <= WORK_SIZE, "SR_FRACTION out of range: signal region must fit the grid"

script_dir = os.path.dirname(os.path.abspath(__file__))
TARGETS_DIR = os.path.join(script_dir, "targets")

#%% ---------- Target ----------
input_tiff_path = input_tiff if os.path.isabs(input_tiff) else os.path.join(TARGETS_DIR, input_tiff)
if not os.path.isfile(input_tiff_path):
    raise FileNotFoundError(f"TIFF file not found: {input_tiff_path}")
target_name = os.path.splitext(os.path.basename(input_tiff))[0]

F1 = np.array(Image.open(input_tiff_path))
F1 = F1[..., 0] if F1.ndim == 3 else F1                       # first channel for RGB/RGBA
F1 = cv2.resize(F1.astype(np.float32), (SR_SIZE, SR_SIZE), interpolation=cv2.INTER_AREA)
F1 = np.maximum(F1, target_floor_rel * (np.max(F1) + 1e-12))
n, m = F1.shape                                               # = (SR_SIZE, SR_SIZE); Poisson solve size
E = float(np.sum(F1)); El = 0.5 * E                          # target / noise-region energies
pad_each = (WORK_SIZE - SR_SIZE) // 2                         # center the SR inside the work grid
F = np.pad(np.abs(np.sqrt(F1)), ((pad_each, pad_each), (pad_each, pad_each)), mode="constant")
nn, mm = F.shape                                              # = (WORK_SIZE, WORK_SIZE)

#%% ---------- Band-limit masks, incident field, grids ----------
ap0 = (WORK_SIZE - HOLOGRAM_SIZE) // 2
bandlim_spe = cp.zeros((nn, mm), cp.float32)                  # SLM aperture (central HOLOGRAM_SIZE)
bandlim_spe[ap0:ap0 + HOLOGRAM_SIZE, ap0:ap0 + HOLOGRAM_SIZE] = 1.0
bandlim_in = cp.zeros((nn, mm), cp.float32)                   # Signal region SR (central SR_SIZE)
bandlim_in[pad_each:pad_each + SR_SIZE, pad_each:pad_each + SR_SIZE] = 1.0
bandlim_ou = 1.0 - bandlim_in                                 # Noise region NR
sr_r0, sr_r1 = pad_each, pad_each + SR_SIZE
sr_c0, sr_c1 = pad_each, pad_each + SR_SIZE

w = 0.26                                                      # [mm] beam waist
ox, oy = cp.meshgrid(cp.linspace(-dh * mm / 2, dh * mm / 2, mm), cp.linspace(-dh * nn / 2, dh * nn / 2, nn))
incident = cp.exp(-((ox ** 2) + (oy ** 2)) / w) * bandlim_spe

# Chen quadratic (lens) init over the WORK grid; fringes fill the pad margin.
u = cp.linspace(-0.5, 0.5, mm).reshape(1, mm); v = cp.linspace(-0.5, 0.5, nn).reshape(nn, 1)
RR2 = cp.broadcast_to(u, (nn, mm)) ** 2 + cp.broadcast_to(v, (nn, mm)) ** 2
c_chen = 4.0 * np.pi * pad_each                              # = pi*SR_SIZE when pad = SR/4 (canonical 3/4 geometry)

# DCT-II (Neumann) Laplacian eigenvalues for the SR-size Poisson solve.
_ii = cp.arange(n).reshape(n, 1); _jj = cp.arange(m).reshape(1, m)
_denom = (2 * cp.cos(cp.pi * _ii / n) - 2) + (2 * cp.cos(cp.pi * _jj / m) - 2); _denom[0, 0] = 1.0

F_gpu = cp.asarray(F); E_gpu = cp.asarray(E); El_gpu = cp.asarray(El); F1_gpu = cp.asarray(F1)
_dev = cp.cuda.Device()

#%% ---------- Core operators ----------
def irrotational_phase(pha):
    """Least-squares irrotational (vortex-free) phase via a DCT-II (Neumann) Poisson solve.

    Wrapped-gradient RHS -> nabla^2 psi = rho with zero-flux BC. psi is curl-free by
    construction (no vortices), O(N^2 log N), independent of the vortex count.
    """
    dx = cp.zeros((n, m)); dy = cp.zeros((n, m))
    dx[:, :-1] = cp.mod(pha[:, 1:] - pha[:, :-1] + cp.pi, 2 * cp.pi) - cp.pi
    dy[:-1, :] = cp.mod(pha[1:, :] - pha[:-1, :] + cp.pi, 2 * cp.pi) - cp.pi
    rho = cp.zeros((n, m)); rho[:, 0] = dx[:, 0]; rho[:, 1:] = dx[:, 1:] - dx[:, :-1]
    rho[0, :] += dy[0, :]; rho[1:, :] += dy[1:, :] - dy[:-1, :]
    phi_hat = dctn(rho, type=2, norm="ortho") / _denom; phi_hat[0, 0] = 0.0
    return idctn(phi_hat, type=2, norm="ortho")


def projector_op(pha_crop, a):
    """Relaxed irrotational projection of an SR phase crop -> vortex-free phase crop."""
    psi = irrotational_phase(pha_crop)
    psi = psi + cp.angle(cp.sum(cp.exp(1j * (pha_crop - psi))))           # align global offset
    return cp.angle((1 - a) * cp.exp(1j * pha_crop) + a * cp.exp(1j * psi))


def paper_op(pha_crop):
    """Paper arctan2 vortex elimination -> vortex-free phase crop (O(K*N^2))."""
    return function_vortex_elimination_accegpu(pha_crop, dh, use_cupy=True, gather_output=False)


def final_reconstruction(E2_k):
    """Shipped phase-only reconstruction: SR intensity from the final SLM-plane field."""
    hologram = incident * cp.exp(1j * cp.angle(E2_k))
    Rec = cp.fft.fftshift(cp.fft.ifft2(cp.fft.fftshift(hologram)))[sr_r0:sr_r1, sr_c0:sr_c1]
    I = cp.abs(Rec) ** 2
    return cp.asnumpy(E_gpu * I / (cp.sum(I) + 1e-12))


def filtered_random_phase(s):
    """Vortex-free init: a smooth REAL field used AS the phase, Nyquist-capped -> ZERO vortices.

    Only two physical numbers are chosen: corr_len_px (smoothness) and grad_cap (< 1, Nyquist
    safety); everything else is derived. Unlike arg() of a band-limited COMPLEX field (whose zeros
    are speckle vortices), a single-valued real field has zero circulation -> no vortices.
    """
    # 1) White REAL noise: one independent Gaussian value per pixel (REAL, never a complex field).
    cp.random.seed(s); noise = cp.random.randn(nn, mm)
    # 2) Low-pass = impose a correlation length. ks is DERIVED from corr_len_px (no magic Fourier units):
    #    a real-space Gaussian of sigma=corr_len_px pairs with a frequency Gaussian of sigma_k=N/(2*pi*corr_len_px).
    ks = WORK_SIZE / (2 * np.pi * corr_len_px)
    ky = (cp.fft.fftfreq(nn) * nn).reshape(nn, 1); kx = (cp.fft.fftfreq(mm) * mm).reshape(1, mm)
    lp = cp.exp(-(kx ** 2 + ky ** 2) / (2 * ks ** 2))
    # 3) Back to real space -> a smooth REAL field (no mean/std normalise: only step 4 matters).
    sm = cp.real(cp.fft.ifft2(cp.fft.fft2(noise) * lp))
    # 4) Nyquist scaling: scale so the LARGEST neighbour step = grad_cap*pi < pi (no aliasing). This
    #    single peak-step rescale is the only thing that matters, so std-normalising first cancels.
    gmax = float(cp.maximum(cp.abs(cp.diff(sm, axis=1)).max(), cp.abs(cp.diff(sm, axis=0)).max()).get())
    # 5) Use the real field DIRECTLY as the phase -> single-valued -> zero circulation -> no vortices.
    return cp.exp(1j * (grad_cap * np.pi / (gmax + 1e-12)) * sm)


#%% ---------- Single run ----------
def run(kind, schedule, init, label=""):
    """kind in {'paper','projector'}; schedule in {'periodic','every'}; init in {'random','chen','filtered'}."""
    print(f"[{label}] start  (kind={kind}, schedule={schedule}, init={init})", flush=True)
    cp.random.seed(seed)
    if init == "chen":
        phi = cp.exp(1j * c_chen * RR2)
    elif init == "filtered":
        phi = filtered_random_phase(seed)
    else:
        phi = cp.exp(1j * 2 * cp.pi * cp.random.rand(nn, mm))
    cp.random.seed(seed)
    amp = cp.random.rand(nn, mm)

    RMSE = np.zeros(loop); NUM = np.zeros(loop, dtype=int)
    event_time = 0.0; n_events = 0

    _dev.synchronize(); t0 = time.perf_counter()
    for i in range(1, loop):
        amp = bandlim_in * F_gpu + bandlim_ou * amp
        E2 = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(amp * phi)))
        E2_ave = cp.sqrt((E_gpu + El_gpu) * incident ** 2 / cp.sum(incident ** 2))
        E2_k = E2_ave * cp.exp(1j * cp.angle(E2))
        es = cp.fft.fftshift(cp.fft.ifft2(cp.fft.fftshift(E2_k)))

        amp = cp.abs(es); a_in = bandlim_in * amp; a_ou = bandlim_ou * amp
        amp = cp.sqrt(E_gpu) * (a_in / (cp.sqrt(cp.sum(a_in ** 2)) + 1e-12)) \
            + cp.sqrt(El_gpu) * (a_ou / (cp.sqrt(cp.sum(a_ou ** 2)) + 1e-12))

        Isr = amp[sr_r0:sr_r1, sr_c0:sr_c1] ** 2; Isr = E_gpu * Isr / (cp.sum(Isr) + 1e-12)
        RMSE[i] = float(cp.sqrt(cp.mean((Isr - F1_gpu) ** 2)).get())

        pha = cp.angle(es); pha_crop = pha[sr_r0:sr_r1, sr_c0:sr_c1]
        if COUNT_VORTICES:
            po, ne = function_vortex_detection_accegpu(pha_crop, dh, use_cupy=True); NUM[i] = po + ne

        if schedule == "every":
            apply = True
        elif schedule == "never":                # baseline: plain GS, no annihilation ever
            apply = False
        else:                                    # periodic
            apply = (i % trigger_x == 0)
        if apply:
            _dev.synchronize(); te0 = time.perf_counter()
            vfree = paper_op(pha_crop) if kind == "paper" else projector_op(pha_crop, alpha)
            _dev.synchronize(); event_time += time.perf_counter() - te0; n_events += 1
            pha_new = pha.copy(); pha_new[sr_r0:sr_r1, sr_c0:sr_c1] = vfree
            phi = cp.exp(1j * pha_new)
        else:
            phi = cp.exp(1j * pha)

        if i % 100 == 0 or i == 1:
            _dev.synchronize()
            elapsed = time.perf_counter() - t0
            eta = elapsed / i * (loop - i)
            vtxt = f"vort={NUM[i]:6d}  " if COUNT_VORTICES else ""
            print(f"  [{label}] iter {i:4d}/{loop-1}  RMSE={RMSE[i]:7.4f}  {vtxt}"
                  f"elapsed={elapsed:5.1f}s  ETA={eta:4.0f}s", flush=True)

    _dev.synchronize(); total_time = time.perf_counter() - t0
    if not COUNT_VORTICES:                       # count once, OUTSIDE the timed region
        po, ne = function_vortex_detection_accegpu(pha_crop, dh, use_cupy=True); NUM[-1] = po + ne
    print(f"[{label}] done   RMSE={RMSE[-1]:.4f}  vort={NUM[-1]}  t_total={total_time:.2f}s", flush=True)
    return dict(RMSE=RMSE, NUM=NUM, I_final=final_reconstruction(E2_k),
                total_time=total_time, event_time=event_time, n_events=n_events,
                per_event=(event_time / n_events if n_events else 0.0))


#%% ---------- Run the methods (baseline + paper + projector variants) ----------
print(f"Comparison on {target_name}  (geometry {HOLOGRAM_SIZE}/{WORK_SIZE}/{SR_SIZE}, "
      f"loop={loop}, x={trigger_x}, alpha={alpha}, c*={c_chen:g})", flush=True)
runs = {
    "baseline":        run("projector", "never",    "random",   label="baseline"),
    "paper":           run("paper",     "periodic", "random",   label="paper"),
    "proj_periodic":   run("projector", "periodic", "random",   label="proj_periodic"),
    "proj_every":      run("projector", "every",    "random",   label="proj_every"),
    "proj_every_chen": run("projector", "every",    "chen",     label="proj_every_chen"),
    "proj_every_filt": run("projector", "every",    "filtered", label="proj_every_filt"),
}

t_paper = runs["paper"]["total_time"]
print(f"\n{'method':16s} {'RMSE':>9s} {'vort':>6s} {'events':>7s} {'t_total':>9s} "
      f"{'t/event':>10s} {'speedup':>8s}")
for name, r in runs.items():
    speed = t_paper / r["total_time"] if r["total_time"] else float("nan")
    print(f"{name:16s} {r['RMSE'][-1]:9.4f} {r['NUM'][-1]:6d} {r['n_events']:7d} "
          f"{r['total_time']:8.2f}s {r['per_event']*1e3:9.2f}ms {speed:7.2f}x", flush=True)

#%% ---------- Results table (simple markdown, regenerated each run) ----------
tbl = [f"# Final comparison — {target_name}",
       f"geometry {HOLOGRAM_SIZE}/{WORK_SIZE}/{SR_SIZE}, loop={loop}, x={trigger_x}, alpha={alpha}, c*={c_chen:g}",
       "",
       "| method | RMSE | vortices | events | t_total [s] | t/event [ms] | speedup |",
       "|---|---|---|---|---|---|---|"]
for name, r in runs.items():
    speed = t_paper / r["total_time"] if r["total_time"] else float("nan")
    tbl.append(f"| {name} | {r['RMSE'][-1]:.4f} | {r['NUM'][-1]} | {r['n_events']} | "
               f"{r['total_time']:.2f} | {r['per_event']*1e3:.2f} | {speed:.2f}x |")
table_md = "\n".join(tbl) + "\n"
table_path = os.path.join(script_dir, f"results_comparison_{target_name}.md")
with open(table_path, "w", encoding="utf-8") as fh:
    fh.write(table_md)
print(f"\nResults table -> {table_path}")

#%% ---------- Plots (all figures built, then a single plt.show() at the end) ----------
colors = {"baseline": "tab:gray", "paper": "tab:red", "proj_periodic": "tab:orange",
          "proj_every": "tab:blue", "proj_every_chen": "tab:green",
          "proj_every_filt": "tab:purple"}

fig1 = plt.figure()
for name, r in runs.items():
    plt.plot(r["RMSE"][1:], label=name, color=colors[name])
plt.xlabel("Iteration"); plt.ylabel("RMSE (SR)")
plt.title(f"RMSE vs iteration — {target_name}"); plt.legend(fontsize=8)

fig2 = None
if COUNT_VORTICES:
    fig2 = plt.figure()
    for name, r in runs.items():
        plt.plot(r["NUM"][1:], label=name, color=colors[name])
    plt.xlabel("Iteration"); plt.ylabel("Vortex count (SR)")
    plt.title(f"Vortex count vs iteration — {target_name}"); plt.legend(fontsize=8)

panels = [("target", F1)] + [(f"{name}\nRMSE={r['RMSE'][-1]:.3f}", r["I_final"]) for name, r in runs.items()]
fig3, ax = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.4))
for a, (t, im) in zip(ax, panels):
    a.imshow(im, cmap="gray"); a.set_title(t, fontsize=8); a.axis("off")
plt.tight_layout()

plt.show()
# %%
