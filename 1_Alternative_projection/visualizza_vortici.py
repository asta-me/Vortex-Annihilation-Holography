
"""
Visualizzazione della fase solenoidale φₚ (Eq. 8) dell'articolo:
"Vortex annihilation to reduce phase singularities and resultant speckles in computer-generated holography"
Xiaomeng Sui et al.

Questo script genera e visualizza la distribuzione di fase φₚ dovuta a vortici puntiformi (singolarità di carica q) nel piano,
come descritto nell'articolo per la parte solenoidale del campo di fase.
"""

import numpy as np
import matplotlib.pyplot as plt


def gaussian_vortex_focus(size, waist, charge, pad_factor=4):
    """
    Genera un fascio gaussiano con fase vorticosa exp(i m phi)
    e calcola il pattern nel piano di fuoco (Fraunhofer) via FFT 2D
    con zero-padding per aumentare la risoluzione di campionamento.

    Args:
        size (int): dimensione griglia quadrata
        waist (float): raggio gaussiano in coordinate normalizzate [-1, 1]
        charge (int): carica topologica m del vortice
        pad_factor (int): fattore di zero-padding per la FFT

    Returns:
        tuple: X, Y, amp0, phase0, Ifocus_norm
    """
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)

    R = np.sqrt(X**2 + Y**2)
    Phi = np.arctan2(Y, X)

    amp0 = np.exp(-(R / waist) ** 2)
    field0 = amp0 * np.exp(1j * charge * Phi)
    phase0 = np.angle(field0)

    # Zero-padding centrato prima della FFT per aumentare la risoluzione nel piano di fuoco
    padded_size = pad_factor * size
    field0_padded = np.zeros((padded_size, padded_size), dtype=complex)
    start = (padded_size - size) // 2
    end = start + size
    field0_padded[start:end, start:end] = field0

    # Trasformata di Fourier -> campo nel piano di fuoco di una lente ideale
    ufocus = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field0_padded)))
    ifocus = np.abs(ufocus) ** 2
    ifocus_norm = ifocus / np.max(ifocus)

    return X, Y, amp0, phase0, ifocus_norm

def vortex_phase(size, singularities):
    """
    Calcola la fase solenoidale φₚ generata da singolarità (x, y, q) su una griglia quadrata.
    Args:
        size (int): dimensione della griglia (size x size)
        singularities (list): lista di tuple (x, y, q) con posizione normalizzata [-1,1] e carica q
    Returns:
        np.ndarray: matrice 2D della fase φₚ in [-π, π]
    """
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    phi_p = np.zeros((size, size))
    for sx, sy, q in singularities:
        phi_p += q * np.arctan2(Y - sy, X - sx)
    # Wrapping fase tra -π e π
    return (phi_p + np.pi) % (2 * np.pi) - np.pi


# -----------------------------------------------------------------------------
# Passo iniziale: singolo vortice gaussiano exp(i m phi)
# Mostriamo che nel piano di fuoco il centro resta buio per m != 0
# -----------------------------------------------------------------------------
N0 = 512
w0 = 0.35
m = 1

X0, Y0, A0, P0, I_focus = gaussian_vortex_focus(N0, w0, m, pad_factor=4)
center_idx = I_focus.shape[0] // 2
center_intensity = I_focus[center_idx, center_idx]

# Mostriamo solo la regione super-centrale del piano di Fourier
zoom_half_width = 28
I_focus_zoom = I_focus[
    center_idx - zoom_half_width:center_idx + zoom_half_width + 1,
    center_idx - zoom_half_width:center_idx + zoom_half_width + 1,
]

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

im0 = axes[0].imshow(A0, extent=[-1, 1, -1, 1], origin='lower', cmap='magma')
axes[0].set_title('Ampiezza iniziale gaussiana')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

im1 = axes[1].imshow(P0, extent=[-1, 1, -1, 1], origin='lower', cmap='bwr', vmin=-np.pi, vmax=np.pi)
axes[1].set_title(rf'Fase iniziale $m\phi$ con $m={m}$')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

im2 = axes[2].imshow(I_focus_zoom, origin='lower', cmap='inferno')
axes[2].set_title('Intensita al fuoco (zoom super-centrale)')
axes[2].set_xlabel('kx (zoom)')
axes[2].set_ylabel('ky (zoom)')
fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

fig.suptitle(
    f'Singolo vortice gaussiano: I(0) / Imax = {center_intensity:.2e} (atteso ~ 0 per m != 0)',
    y=1.03,
)
plt.tight_layout()
plt.show()

# Parametri: dimensione griglia e vortici (x, y, carica)
N = 512
vortices = [
    (0.4, 0.5, 1),    # Vortice +1
    (-0.3, -0.2, 1),  # Vortice +1
    (0.2, -0.6, -1)   # Vortice -1
]

# Calcolo fase solenoidale
phi_p = vortex_phase(N, vortices)

# Visualizzazione
plt.figure(figsize=(7, 6))
plt.imshow(phi_p, extent=[-1, 1, -1, 1], cmap='bwr', origin='lower')
plt.colorbar(label='Fase [rad]')
for sx, sy, q in vortices:
    color = 'black' if q > 0 else 'white'
    marker = 'o' if q > 0 else 'x'
    plt.scatter(sx, sy, color=color, marker=marker, s=80, edgecolors='white', zorder=5)
plt.title(r'Distribuzione di fase $\varphi_P$ (Eq. 8)')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()