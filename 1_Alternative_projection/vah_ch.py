"""
Python translation of AlternativeProjection_VAH_CH.m
- Complex hologram with vortex annihilation (VAH)
- GPU-first (CuPy), fallback NumPy
- Uses function_vortex_detection_accegpu.py and function_vortex_elimination_accegpu.py
"""
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import cupy as cp
from function_vortex_detection_accegpu import function_vortex_detection_accegpu
from function_vortex_elimination_accegpu import function_vortex_elimination_accegpu
from skimage.transform import resize

# --- Parameters ---
mat = scipy.io.loadmat("object_grayscale.mat")
F1 = mat["F1"]
F1 = np.array(F1, dtype=np.float32)
F1 = resize(F1, (512, 512), order=1, preserve_range=True, anti_aliasing=True)
nn, mm = F1.shape
E = np.sum(F1)
lamda = 532e-6
k = 2 * np.pi / lamda
dh = 0.00374
F = np.abs(np.sqrt(F1))

# --- Initial phase ---
phi = cp.exp(1j * 2 * cp.pi * cp.random.rand(nn, mm))

# --- Band-limitation mask ---
bandlim_spe = cp.zeros((nn, mm), dtype=cp.float32)
bandlim_spe[nn//4:3*nn//4, mm//4:3*mm//4] = 1.0

# --- Iterative alternative projection with VAH ---
loop = 100
RMSE = np.zeros(loop)
NUM_PO = np.zeros(loop, dtype=int)
NUM_NE = np.zeros(loop, dtype=int)

F_gpu = cp.asarray(F)
F1_gpu = cp.asarray(F1)
E_gpu = cp.asarray(E)

plt.ion()
fig, ax = plt.subplots()

t_start = time.perf_counter()
for i in range(1, loop):
    amp = F_gpu
    E1 = amp * phi
    E2 = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(E1)))
    E2_abs = cp.abs(E2) * bandlim_spe
    E2_ave = cp.sqrt(E_gpu * cp.sum(E2_abs ** 2) / cp.sum(E2_abs ** 2))
    E2_k = E2_ave * cp.exp(1j * cp.angle(E2))
    es = cp.fft.fftshift(cp.fft.ifft2(cp.fft.fftshift(E2_k)))
    amp = cp.abs(es)
    amp = cp.sqrt(E_gpu * cp.sum(amp ** 2) / cp.sum(amp ** 2))
    I = amp ** 2
    I = E_gpu * I / cp.sum(I)
    P = cp.mod(cp.angle(es), 2 * cp.pi)
    # Visual feedback
    if i % 10 == 0 or i == loop - 1:
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
    if abs(diff_RMSE) < 0.0005 and RMSE[i] > 0.03:
        pha = cp.asnumpy(cp.angle(es))
        pha_vfree = function_vortex_elimination_accegpu(pha, dh, use_cupy=False)
        NUM_PO[i], NUM_NE[i] = function_vortex_detection_accegpu(pha_vfree, dh, use_cupy=False)
        phi = cp.exp(1j * cp.asarray(pha_vfree))
    else:
        phi = cp.exp(1j * cp.angle(es))
        phi_in = cp.asnumpy(cp.angle(es))
        NUM_PO[i], NUM_NE[i] = function_vortex_detection_accegpu(phi_in, dh, use_cupy=False)

t_total = time.perf_counter() - t_start
print(f"\nDone! Total time: {t_total:.1f}s ({t_total/60:.1f} min)")
plt.ioff()
plt.figure()
plt.plot(RMSE, label="RMSE")
plt.xlabel("Iteration")
plt.ylabel("RMSE")
plt.title("RMSE vs Iteration (VAH CH)")
plt.legend()
plt.show()

# --- Final reconstruction ---
An = cp.angle(E2_k)
Am = cp.abs(E2_k)
hologram = Am * cp.exp(1j * An)
Rec = cp.fft.fftshift(cp.fft.ifft2(cp.fft.fftshift(hologram)))
I_final = cp.abs(Rec) ** 2
I_final = E_gpu * I_final / cp.sum(I_final)
NUM = NUM_PO + NUM_NE

plt.figure()
plt.imshow(cp.asnumpy(I_final), cmap="gray")
plt.title("Final reconstructed intensity (VAH CH)")
plt.show()

print(f"Final RMSE: {RMSE[-1]:.6f}")
print(f"Final vortex count: {NUM[-1]}")
