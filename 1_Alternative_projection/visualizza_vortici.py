
"""
Visualizzazione della fase solenoidale φₚ (Eq. 8) dell'articolo:
"Vortex annihilation to reduce phase singularities and resultant speckles in computer-generated holography"
Xiaomeng Sui et al.

Questo script genera e visualizza la distribuzione di fase φₚ dovuta a vortici puntiformi (singolarità di carica q) nel piano,
come descritto nell'articolo per la parte solenoidale del campo di fase.
"""

import numpy as np
import matplotlib.pyplot as plt

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
plt.imshow(phi_p, extent=[-1, 1, -1, 1], cmap='hsv', origin='lower')
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