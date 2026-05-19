#%%
"""
AlternativeProjection_VAH_PH in Python (GPU-first variant for experimental data).

Compared with the baseline script `vah_ph.py`, this version introduces:

1) Experimental input pipeline
    - Target comes from image file (`input_image_path`) instead of `object_grayscale.mat`.
    - Incident field comes from measured camera intensity (`Input_Gaussian.tiff`) and is
      converted to amplitude with sqrt + normalization.

2) Upsampling strategy
    - Input target is first resized to `res x res`.
    - Then a sinc-like 2x upsampling (`sinc_interpol`) builds a `2*res x 2*res`
      computational grid for holographic propagation.

3) Configurable signal region (SR)
    - SR is a centered square `M x M` defined by mask (`bandlim_in`).
    - RMSE is computed only on SR support (`sr_idx`), not on the full plane.

4) VAH localized to active SLM area
    - Vortex elimination/detection is run on a crop around active SLM
      (`r0_vah:r1_vah, c0_vah:c1_vah`) with optional `padding_vah`.

5) Faster GPU execution path
    - Main iterative loop uses CuPy arrays.
    - Live plotting can be disabled (`live_plotting=False`) to reduce overhead.
    - RMSE is accumulated on GPU and gathered at the end for plotting.

Usage
-----
1. Set core parameters:
    - `res`: SLM side in pixels.
    - `M`: centered SR side in pixels.
    - `iters`: number of AP iterations.
    - `mixing_parameter`: noise-region energy ratio (`El = mixing_parameter * E`).
    - `vah_application_interval` or RMSE-based VAH condition in loop.

2. Select input files:
    - `input_image_path` for target image.
    - `Input_Gaussian.tiff` for measured incident intensity.

3. Run:
    - `python vah_ph_marco.py`

4. Outputs:
    - RMSE trend plot.
    - Final reconstructed intensity on SR.
    - Vortex count trends (`NUM_PO`, `NUM_NE`, `NUM`).
    - Final SLM phase BMP (`output_phase_red.bmp`).
"""

#%% ---------- Imports ----------
import time
import numpy as np
import matplotlib.pyplot as plt
import cupy as cp
from function_vortex_detection_accegpu import function_vortex_detection_accegpu
from function_vortex_elimination_accegpu import function_vortex_elimination_accegpu
from skimage.transform import resize
from skimage.io import imread, imsave

#%% ---------- System parameters ----------
lamda = 632e-6                           # [mm] Wavelength in mm
dh = 0.008                               # [mm] Pixel pitch
res = 540                                # SLM/output hologram size (NxN)
work_size = 2 * res                      # Computational grid size (2N x 2N)
live_plotting = False                    # Whether to show live updates of the reconstruction during iterations  
padding_vah = 0                          # Extra pixels around active SLM region used for VAH crop
M = 400                                  # Signal region is central MxM pixel
mixing_parameter = 0.5                   # Mixing parameter for energy distribution

# We can either select a diff and rmse criterion (like in the original paper)
# or Just apply vah every N iterations
# Comment or uncomment the if condition in the main loop accordingly
iters = 500
vah_application_interval = 100             # Apply VAH every N iterations (if using interval-based application)
diff_threshold = 0.00023                                            # Original difference threshold for applying VAH
rmse_threshold = 0.035                                              # Original RMSE threshold for applying VAH

input_image_path = "Cat_black.tif"       # Target Image intensity path
# input_image_path = "Cat_1.tif"
# input_image_path = "Cat_2.tif"


def sinc_interpol(image: cp.ndarray) -> cp.ndarray:
    """2x sinc-like interpolation via FFT zero-padding"""
    h, w = image.shape
    # 1. FFT
    spec = cp.fft.fftshift(cp.fft.fft2(image))
    spec_up = cp.zeros((2 * h, 2 * w), dtype=cp.complex128)   
    # 2. Padding
    spec_up[h//2 : h//2 + h, w//2 : w//2 + w] = spec
    # 3. Inverse FFT
    up = cp.fft.ifft2(cp.fft.ifftshift(spec_up))
    # 4. Extract real part and clip (not abs but real>0)
    up = cp.maximum(cp.real(up), 0.0)
    # 5. Energy conservation
    in_energy = cp.sum(image)
    out_energy = cp.sum(up)
    if out_energy > 0:
        up *= (in_energy / out_energy)
    return up.astype(cp.float32)

#%% ---------- Import target ----------
F1 = imread(input_image_path).astype(np.float32)
F1 = resize(F1, (res, res), order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)
n = m = res
nn = mm = work_size
F1_work = sinc_interpol(cp.asarray(F1)) # 2N x 2N sinc upsampling
E = np.sum(F1)                          # Target Energy
El = mixing_parameter * E               # Noise Energy 
F = cp.sqrt(F1_work)                    # Amplitude on 2N x 2N grid

# --- Initial phase and amplitude ---
# Set a random seed for reproducibility
cp.random.seed(42)
phi = cp.exp(1j * 2 * cp.pi * cp.random.rand(nn, mm))               # Random initial phase (complex exponential of random values in [0,1))
amp = cp.random.rand(nn, mm)                                        # Random initial amplitude (Initial guess for target noise region)

# --- Band-limitation masks ---
# SLM Active Region: Centered square of size res x res within the 2N x 2N grid
active_rows = slice(res // 2, res // 2 + res) 
active_cols = slice(res // 2, res // 2 + res)
bandlim_spe = cp.zeros((nn, mm), dtype=cp.float32)                  # SLM aperture mask  
bandlim_spe[active_rows, active_cols] = 1.0                         # Active central SLM region (res x res)

# Target Image Signal and Noise Region
bandlim_in = cp.zeros((nn, mm), dtype=cp.float32)                   
# Define SR from M
sr_r0, sr_r1 = (nn - M) // 2, (nn + M) // 2
sr_c0, sr_c1 = (mm - M) // 2, (mm + M) // 2
bandlim_in[sr_r0:sr_r1, sr_c0:sr_c1] = 1.0                          # Signal Region SR
bandlim_ou = 1.0 - bandlim_in                                       # Noise region NR

# SR indices derived from mask.
sr_idx = cp.where(bandlim_in > 0)
target_sr = cp.asarray(F1_work[sr_idx])

# Incident intensity measured on the SLM
incident_np = imread("Input_Gaussian.tiff").astype(np.float32)
if incident_np.shape != (n, m):
    incident_np = resize(incident_np, (n, m), order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)
incident_cp = cp.asarray(incident_np)                          # Bring on GPU
incident_cp = cp.sqrt(incident_cp)                             # Convert intensity to amplitude
incident_cp = incident_cp - cp.min(incident_cp)                # Normalize
incident_cp = incident_cp / (cp.max(incident_cp) + 1e-12)
incident = cp.zeros((nn, mm), dtype=cp.float32)                # Pad to working size
incident[active_rows, active_cols] = incident_cp


# VAH is now not performd on all of the 2N x 2N grid, but only on the active SLM region + some padding
# Define VAH crop indices with padding
r0_vah = active_rows.start - padding_vah
r1_vah = active_rows.stop + padding_vah
c0_vah = active_cols.start - padding_vah
c1_vah = active_cols.stop + padding_vah

#%% --- Iterative alternative projection with VAH ---
RMSE = np.zeros(iters)                                               # Root mean square error
NUM_PO = np.zeros(iters, dtype=int)                                  # Number of positive vortices    
NUM_NE = np.zeros(iters, dtype=int)                                  # Number of negative vortices

F_gpu = cp.asarray(F)
E_gpu = cp.asarray(E)
El_gpu = cp.asarray(El)
RMSE_gpu = cp.zeros(iters, dtype=cp.float32)

if live_plotting:
    plt.ion()                                                           # For plotting with live updates
    fig, ax = plt.subplots()

t_start = time.perf_counter()
for i in range(1, iters):
    amp = bandlim_in * F_gpu + bandlim_ou * amp                     # Amplitude constraint in target plane: Target in SR, noise in NR
    E1 = amp * phi                                                  # Constrained amplitude, retain phase 
    E2 = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(E1)))          # Forward FFT to SLM plane
    # Amplitude constraint in the SLM Plane : 
    # Incident beam, Normalise energy to E+El (E_target + 0.5*E_target) 
    E2_ave = cp.sqrt((E_gpu + El_gpu) * incident ** 2 / cp.sum(incident ** 2))  
    E2_k = E2_ave * cp.exp(1j * cp.angle(E2))                       # Constrained amplitude, retain phase
    es = cp.fft.fftshift(cp.fft.ifft2(cp.fft.fftshift(E2_k)))       # Inverse FFT to target plane
    
    amp = cp.abs(es)                                                # Amplitude in target plane after inverse FFT
    amp_in = bandlim_in * amp                                       # Amplitude in signal region
    amp_ou = bandlim_ou * amp                                       # Amplitude in noise region
    
    # Normalize each region separately, then sum
    norm_in = cp.sqrt(E_gpu)  # Normalization coefficient in signal region
    norm_ou = cp.sqrt(El_gpu) # Normalization coefficient in noise region
    # Scale Energy in signal region to E, and Energy in noise region to El (0.5*E)
    amp = norm_in * (amp_in / (cp.sqrt(cp.sum(amp_in ** 2)) + 1e-12)) + norm_ou * (amp_ou / (cp.sqrt(cp.sum(amp_ou ** 2)) + 1e-12))
    
    # Reconstructed intensity on full 2N x 2N target plane.
    I_full = amp ** 2
    I_full = E_gpu * I_full / (cp.sum(I_full) + 1e-12)
    
    # Visual feedback
    if live_plotting:
        if i % 20 == 0 or i == iters - 1:
            # Small 2D display crop using the configured SR rectangle.
            I = I_full[sr_r0:sr_r1, sr_c0:sr_c1]
            ax.clear()
            ax.imshow(cp.asnumpy(I), cmap="gray")
            ax.set_title(f"Iteration {i}")
            plt.pause(0.01)
    
    # Compute RMSE directly on mask support.
    diff_sq_sr = (I_full[sr_idx] - target_sr) ** 2
    mse = cp.mean(diff_sq_sr)
    # RMSE[i] = float(cp.sqrt(mse).get())
    RMSE_gpu[i] = cp.sqrt(mse)
    diff_RMSE = RMSE_gpu[i] - RMSE_gpu[i-1]
    
    # Print progress every 10 iterations
    if i % 10 == 0:
        rmse_i = float(RMSE_gpu[i].get())
        elapsed = time.perf_counter() - t_start
        eta = elapsed / i * (iters - i)
        print(f"Iter {i:4d}/{iters-1}  RMSE={rmse_i:.5f}  elapsed={elapsed:.1f}s  Tempo mancante={eta:.0f}s")

    # My proposal to only apply vah to slm region + some borders
    # if abs(diff_RMSE) < diff_threshold and RMSE[i] > rmse_threshold:
    if i % vah_application_interval == 0:
        print(f"VAH APPLIED at iter {i}")
        pha = cp.angle(es)  # pha: cp.ndarray

        # Crop
        pha_crop = pha[r0_vah:r1_vah, c0_vah:c1_vah]

        # Vortex elimination
        pha_vfree_crop = function_vortex_elimination_accegpu(pha_crop, dh, use_cupy=True, gather_output=False)

        # Vortex detection su GPU
        NUM_PO_i, NUM_NE_i = function_vortex_detection_accegpu(pha_vfree_crop, dh, use_cupy=True)
        NUM_PO[i], NUM_NE[i] = NUM_PO_i, NUM_NE_i

        # Reconstruct full phase map with vortex-free crop
        pha_vfree = pha.copy()
        pha_vfree[r0_vah:r1_vah, c0_vah:c1_vah] = pha_vfree_crop

        phi = cp.exp(1j * pha_vfree)
    
    else:
        phi = cp.exp(1j * cp.angle(es))
        phi_in_crop = cp.angle(phi)[r0_vah:r1_vah, c0_vah:c1_vah]
        NUM_PO[i], NUM_NE[i] = function_vortex_detection_accegpu(phi_in_crop, dh, use_cupy=True)

t_total = time.perf_counter() - t_start
print(f"\nDone! Total time: {t_total:.1f}s ({t_total/60:.1f} min)")
plt.ioff()

RMSE = cp.asnumpy(RMSE_gpu)

# --- Final reconstruction ---
An = cp.angle(E2_k)
hologram = incident * cp.exp(1j * An)
Rec = cp.fft.fftshift(cp.fft.ifft2(cp.fft.fftshift(hologram)))
I_final = cp.abs(Rec) ** 2
# Crop to signal region (SR) only
I_final = I_final[sr_r0:sr_r1, sr_c0:sr_c1]
I_final = E_gpu * I_final / cp.sum(I_final)
NUM = NUM_PO + NUM_NE

# --- Plots ---
plt.figure()
plt.plot(RMSE[1:], label="RMSE")
plt.xlabel("Iteration")
plt.ylabel("RMSE")
plt.title("RMSE vs Iteration (VAH PH)")
plt.legend()
plt.show()

plt.figure(figsize=(8, 6), dpi=300)
plt.imshow(cp.asnumpy(I_final), cmap="gray")
plt.title("Final Reconstructed Intensity (VAH PH) - Signal Region")
plt.show()

plt.figure()
plt.plot(NUM_PO, label="Positive vortices")
plt.plot(NUM_NE, label="Negative vortices")
plt.plot(NUM, label="Total vortices")
plt.xlabel("Iteration")
plt.ylabel("Vortex count")
plt.title("Vortex Count Trend During Iterations")
plt.legend()
plt.show()

print(f"Final RMSE: {RMSE[-1]:.6f}")
print(f"Final vortex count: {NUM[-1]}")

# --- Save active SLM phase as 8-bit BMP (0..255) ---
phase_slm = cp.mod(An, 2 * cp.pi)
phase_active = cp.asnumpy(phase_slm[active_rows, active_cols])
phase_bmp = np.uint8(np.round((phase_active / (2 * np.pi)) * 255.0))
imsave("output_phase_red.bmp", phase_bmp, check_contrast=False)
print(f"Saved output phase BMP: output_phase.bmp with shape {phase_bmp.shape}")



# %%
