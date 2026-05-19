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
                                        # Could be a parameter to make this MRAF-like


F = np.abs(np.sqrt(F1))                                             # Amplitude of the target field (sqrt of intensity) (why abs?)
F = np.pad(F, ((n//4, n//4), (m//4, m//4)), mode="constant")        # Pad to 1024x1024 with zeros (constant padding)
nn, mm = F.shape                                                    # Dimensions after padding (should be 1024x1024)

# --- Initial phase and amplitude ---
# Set a random seed for reproducibility
cp.random.seed(42)
phi = cp.exp(1j * 2 * cp.pi * cp.random.rand(nn, mm))               # Random initial phase (complex exponential of random values in [0,1))
amp = cp.random.rand(nn, mm)                                        # Random initial amplitude (Initial guess for target noise region)

# --- Band-limitation masks ---
bandlim_spe = cp.zeros((nn, mm), dtype=cp.float32)                  # SLM aperture mask  
bandlim_spe[nn//4:3*nn//4, mm//4:3*mm//4] = 1.0                       

bandlim_in = cp.zeros((nn, mm), dtype=cp.float32)                   # Signal Region SR
# bandlim_in[nn//4:3*nn//4, mm//4:3*mm//4] = 1.0                      # Originally here 
# bandlim_in[nn//3:2*nn//3, mm//3:2*mm//3] = 1.0                    # Originally here, useless   
bandlim_in[(nn - n)//2:(nn + n)//2, (mm - m)//2:(mm + m)//2] = 1.0  # My correction, include all signal region     
bandlim_ou = 1.0 - bandlim_in                                       # Noise region NR

# Incident Gaussian
w = 0.26                                                            # [mm] Beam waist of the incident Gaussian
ox, oy = cp.meshgrid(cp.linspace(-dh*mm/2, dh*mm/2, mm), cp.linspace(-dh*nn/2, dh*nn/2, nn))
Gaussian = cp.exp(-((ox**2)+(oy**2))/w)
incident = Gaussian * bandlim_spe                                   # Incident field * SLM aperture mask

#%% --- Iterative alternative projection with VAH ---
loop = 200
diff_threshold = 0.00023                                            # Original difference threshold for applying VAH
rmse_threshold = 0.035                                              # Original RMSE threshold for applying VAH
RMSE = np.zeros(loop)                                               # Root mean square error
NUM_PO = np.zeros(loop, dtype=int)                                  # Number of positive vortices    
NUM_NE = np.zeros(loop, dtype=int)                                  # Number of negative vortices

padding_vah = 0                                                     # Extra pixels around active SLM region used for VAH crop
r0_vah = (nn - n)//2 - padding_vah
r1_vah = (nn - n)//2 + n + padding_vah
c0_vah = (mm - m)//2 - padding_vah
c1_vah = (mm - m)//2 + m + padding_vah

F_gpu = cp.asarray(F)
F1_gpu = cp.asarray(F1)
E_gpu = cp.asarray(E)
El_gpu = cp.asarray(El)

plt.ion()                                                           # For plotting with live updates
fig, ax = plt.subplots()

t_start = time.perf_counter()
for i in range(1, loop):
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
    norm_in = cp.sqrt(E_gpu * cp.sum(amp_in ** 2) / cp.sum(amp_in ** 2)) if cp.sum(amp_in ** 2) > 0 else 0  # Normalization coefficient in signal region
    norm_ou = cp.sqrt(El_gpu * cp.sum(amp_ou ** 2) / cp.sum(amp_ou ** 2)) if cp.sum(amp_ou ** 2) > 0 else 0 # Normalization coefficient in noise region
    # Scale Energy in signal region to E, and Energy in noise region to El (0.5*E)
    amp = norm_in * (amp_in / (cp.sqrt(cp.sum(amp_in ** 2)) + 1e-12)) + norm_ou * (amp_ou / (cp.sqrt(cp.sum(amp_ou ** 2)) + 1e-12))
    
    # Crop to signal region to display and compute RMSE
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
    
    # Compute RMSE
    Diff = cp.asnumpy(I) - F1
    MSE = np.sum(Diff ** 2) / I.size
    RMSE[i] = np.sqrt(MSE)
    diff_RMSE = RMSE[i] - RMSE[i-1]
    
    # Print progress every 10 iterations
    if i % 10 == 0:
        elapsed = time.perf_counter() - t_start
        eta = elapsed / i * (loop - i)
        print(f"Iter {i:4d}/{loop-1}  RMSE={RMSE[i]:.5f}  elapsed={elapsed:.1f}s  ETA={eta:.0f}s")
    
    # Condition to apply vortex annihilation (pretty arbitrary)
    # if abs(diff_RMSE) < diff_threshold and RMSE[i] > rmse_threshold:
    #     print(f"VAH APPLIED at iter {i}")
    #     pha = cp.asnumpy(cp.angle(es))                          # Phase on SLM region
    #     pha_in = pha * cp.asnumpy(bandlim_in)                   # Phase on SLM
    #     pha_vfree = function_vortex_elimination_accegpu(pha_in, dh, use_cupy=False) # Vortex elimination in the SLM plane, returns vortex-free phase
    #     NUM_PO[i], NUM_NE[i] = function_vortex_detection_accegpu(pha_vfree, dh, use_cupy=False) # Count vortices in the vortex-free phase to verify elimination
    #     # Combine vortex-free phase in signal region with original phase in noise region
    #     # (useless mixing we only propagate amplitude from the central region the intensity)
    #     pha_vfree = cp.asnumpy(bandlim_in) * pha_vfree + cp.asnumpy(bandlim_ou) * pha 
    #     phi = cp.exp(1j * cp.asarray(pha_vfree))

    # else:
    # phi = cp.exp(1j * cp.angle(es))
    # phi_in = cp.asnumpy(cp.angle(phi)) * cp.asnumpy(bandlim_in)
    # NUM_PO[i], NUM_NE[i] = function_vortex_detection_accegpu(phi_in, dh, use_cupy=False)


    # My proposal to only apply vah to slm region + some borders
    if abs(diff_RMSE) < diff_threshold and RMSE[i] > rmse_threshold:
        print(f"VAH APPLIED at iter {i}")
        pha = cp.asnumpy(cp.angle(es))  # Phase on SLM region

        # Re-define the region of interest for VAH: SLM region + some padding
        pha_crop = pha[r0_vah:r1_vah, c0_vah:c1_vah]

        # Vortex elimination solo sul crop
        pha_vfree_crop = function_vortex_elimination_accegpu(pha_crop, dh, use_cupy=False)

        # Vortex detection only onh the active region
        NUM_PO[i], NUM_NE[i] = function_vortex_detection_accegpu(pha_vfree_crop, dh, use_cupy=False)

        # Ricostruisci la matrice fase grande
        pha_vfree = pha.copy()
        pha_vfree[r0_vah:r1_vah, c0_vah:c1_vah] = pha_vfree_crop

        phi = cp.exp(1j * cp.asarray(pha_vfree))
    
    else:
        phi = cp.exp(1j * cp.angle(es))
        phi_in_crop = cp.asnumpy(cp.angle(phi))[r0_vah:r1_vah, c0_vah:c1_vah]
        NUM_PO[i], NUM_NE[i] = function_vortex_detection_accegpu(phi_in_crop, dh, use_cupy=False)

t_total = time.perf_counter() - t_start
print(f"\nDone! Total time: {t_total:.1f}s ({t_total/60:.1f} min)")
plt.ioff()
plt.figure()
plt.plot(RMSE[1:], label="RMSE")
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

# %%
