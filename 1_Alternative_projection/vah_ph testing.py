#%%
"""
Python translation of AlternativeProjection_VAH_PH.m
- Phase-only hologram with vortex annihilation (VAH)
- GPU-first (CuPy), fallback NumPy
- Uses function_vortex_detection_accegpu.py and function_vortex_elimination_accegpu.py
"""

#%% ---------- Imports ----------
import time
import numpy as np
import matplotlib.pyplot as plt
import cupy as cp
from function_vortex_detection_accegpu import function_vortex_detection_accegpu
from function_vortex_elimination_accegpu import function_vortex_elimination_accegpu
from skimage.transform import resize

#%% ---------- System parameters ----------
lamda = 532e-6                          # [mm] Wavelength in mm
k = 2 * np.pi / lamda                   # [1/mm] Wavenumber
dh = 0.00374                            # [mm] Pixel pitch

#%% ---------- Import target ----------

# --- To start from a tiff file
# from skimage.io import imread
# input_image_path = "Cat_black.tif"
# F1 = imread(input_image_path)
# --- To start from a .mat file
import scipy.io
mat = scipy.io.loadmat("object_grayscale.mat")
F1 = mat["F1"]


F1 = np.array(F1, dtype=np.float32)     # Convert to float32 
n, m = F1.shape                         # Original dimensions    
F1 = resize(F1, (512, 512), order=1, preserve_range=True, anti_aliasing=True) # Resize to 512x512
n, m = F1.shape                         # Dimensions after resizing  
E = np.sum(F1)                          # Energy of the target image
El = 0.5 * E                            # Energy for the "outer" region in the VAH algorithm
                                        # I guess 0.5 could be a parameter to make this MRAF-like


F = np.abs(np.sqrt(F1))                                             # Amplitude of the target field (sqrt of intensity) (abs(?))
F = np.pad(F, ((n//4, n//4), (m//4, m//4)), mode="constant")        # Pad to 1024x1024 with zeros (constant padding)
nn, mm = F.shape                                                    # Dimensions after padding (should be 1024x1024)

# --- Initial phase and amplitude ---
phi = cp.exp(1j * 2 * cp.pi * cp.random.rand(nn, mm))               # Random initial phase (complex exponential of random values in [0,1))
amp = cp.random.rand(nn, mm)

# --- Band-limitation masks ---
bandlim_spe = cp.zeros((nn, mm), dtype=cp.float32)  # Limits on the SLM aperture
bandlim_spe[nn//4:3*nn//4, mm//4:3*mm//4] = 1.0

bandlim_in = cp.zeros((nn, mm), dtype=cp.float32)   
bandlim_in[nn//4:3*nn//4, mm//4:3*mm//4] = 1.0
bandlim_in[nn//3:2*nn//3, mm//3:2*mm//3] = 1.0 # Inner region of the target image
bandlim_ou = 1.0 - bandlim_in

# Incident Gaussian
w = 0.26                            # [mm] Beam waist of the incident Gaussian
ox, oy = cp.meshgrid(cp.linspace(-dh*mm/2, dh*mm/2, mm), cp.linspace(-dh*nn/2, dh*nn/2, nn))
Gaussian = cp.exp(-((ox**2)+(oy**2))/w)
incident = bandlim_spe * Gaussian   # Incident field on the SLM, limited by the SLM aperture

# --- Iterative alternative projection with VAH ---
loop = 200
RMSE = np.zeros(loop)
NUM_PO = np.zeros(loop, dtype=int)
NUM_NE = np.zeros(loop, dtype=int)

F_gpu = cp.asarray(F)
F1_gpu = cp.asarray(F1)
E_gpu = cp.asarray(E)
El_gpu = cp.asarray(El)

plt.ion()
fig, ax = plt.subplots()

t_start = time.perf_counter()
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
    # Normalize each region separately, then sum
    norm_in = cp.sqrt(E_gpu * cp.sum(amp_in ** 2) / cp.sum(amp_in ** 2)) if cp.sum(amp_in ** 2) > 0 else 0
    norm_ou = cp.sqrt(El_gpu * cp.sum(amp_ou ** 2) / cp.sum(amp_ou ** 2)) if cp.sum(amp_ou ** 2) > 0 else 0
    amp = norm_in * (amp_in / (cp.sqrt(cp.sum(amp_in ** 2)) + 1e-12)) + norm_ou * (amp_ou / (cp.sqrt(cp.sum(amp_ou ** 2)) + 1e-12))
    # Crop to original size for intensity and phase
    I = amp[(nn//2-n//2):(nn//2+n//2), (mm//2-m//2):(mm//2+m//2)] ** 2
    I = E_gpu * I / cp.sum(I)
    P = cp.mod(cp.angle(es), 2 * cp.pi)
    P = P[(nn//2-n//2):(nn//2+n//2), (mm//2-m//2):(mm//2+m//2)]
    # Visual feedback
    if i % 20 == 0 or i == loop - 1:
        ax.clear()
        ax.imshow(cp.asnumpy(I), cmap="gray")
        ax.set_title(f"Iteration {i}")
        plt.pause(0.01)
    # RMSE
    Diff = cp.asnumpy(I) - F1
    MSE = np.sum(Diff ** 2) / I.size
    RMSE[i] = np.sqrt(MSE)
    diff_RMSE = RMSE[i] - RMSE[i-1]
    if i % 10 == 0:
        elapsed = time.perf_counter() - t_start
        eta = elapsed / i * (loop - i)
        print(f"Iter {i:4d}/{loop-1}  RMSE={RMSE[i]:.5f}  elapsed={elapsed:.1f}s  ETA={eta:.0f}s")
    if abs(diff_RMSE) < 0.00023 and RMSE[i] > 0.035:
    # if i% 50 == 0:
        print(f"VAH APPLIED at iter {i}")
        # MATLAB-consistent VAH branch on full padded field
        pha = cp.asnumpy(cp.angle(es))
        pha_in = pha * cp.asnumpy(bandlim_in)
        pha_vfree = function_vortex_elimination_accegpu(pha_in, dh, use_cupy=False)
        NUM_PO[i], NUM_NE[i] = function_vortex_detection_accegpu(pha_vfree, dh, use_cupy=False)
        pha_vfree = cp.asnumpy(bandlim_in) * pha_vfree + cp.asnumpy(bandlim_ou) * pha
        phi = cp.exp(1j * cp.asarray(pha_vfree))

    else:
        phi = cp.exp(1j * cp.angle(es))
        phi_in = cp.asnumpy(cp.angle(phi)) * cp.asnumpy(bandlim_in)
        NUM_PO[i], NUM_NE[i] = function_vortex_detection_accegpu(phi_in, dh, use_cupy=False)

t_total = time.perf_counter() - t_start
print(f"\nDone! Total time: {t_total:.1f}s ({t_total/60:.1f} min)")
plt.ioff()
plt.figure()
plt.plot(RMSE, label="RMSE")
plt.xlabel("Iteration")
plt.ylabel("RMSE")
plt.title("RMSE vs Iteration (VAH PH)")
plt.legend()
plt.show()

# --- Final reconstruction ---
An = cp.angle(E2_k)
hologram = incident * cp.exp(1j * An)
Rec = cp.fft.fftshift(cp.fft.ifft2(cp.fft.fftshift(hologram)))
I_final = cp.abs(Rec) ** 2
I_final = I_final[(nn//2-n//2):(nn//2+n//2), (mm//2-m//2):(mm//2+m//2)]
I_final = E_gpu * I_final / cp.sum(I_final)
NUM = NUM_PO + NUM_NE

plt.figure()
plt.imshow(cp.asnumpy(I_final), cmap="gray")
plt.title("Final reconstructed intensity (VAH PH)")
plt.show()

print(f"Final RMSE: {RMSE[-1]:.6f}")
print(f"Final vortex count: {NUM[-1]}")
