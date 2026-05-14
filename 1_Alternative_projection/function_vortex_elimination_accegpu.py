"""Python translation of MATLAB function_vortex_elimination_accegpu.m.

GPU-first implementation using CuPy when available, with NumPy fallback.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import cupy as cp
except Exception:  # pragma: no cover - CuPy may be unavailable on some systems
    cp = None


ArrayLike = Any


def _get_array_module(pha: ArrayLike, use_cupy: bool | None):
    """Return the numeric backend module (cupy or numpy)."""
    if use_cupy is True and cp is None:
        raise ImportError("CuPy requested but not available.")

    if cp is not None:
        if use_cupy is True:
            return cp
        if use_cupy is None and isinstance(pha, cp.ndarray):
            return cp

    return np


def _to_numpy(arr: Any) -> np.ndarray:
    """Gather to CPU if array is on CuPy, otherwise return NumPy view."""
    if cp is not None and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)


def function_vortex_elimination_accegpu(
    pha: ArrayLike,
    dh: float,
    *,
    use_cupy: bool | None = None,
    gather_output: bool = True,
):
    """Eliminate detected phase vortexes from a wrapped phase map.

    Args:
        pha: 2D phase array.
        dh: Sampling pitch (same meaning as MATLAB code).
        use_cupy: True to force CuPy, False to force NumPy, None to auto.
        gather_output: True to return NumPy array (MATLAB gather behavior).

    Returns:
        Phase map without detected vortexes, wrapped in [0, 2*pi).
    """
    xp = _get_array_module(pha, use_cupy)
    pha_xp = xp.asarray(pha)

    if pha_xp.ndim != 2:
        raise ValueError("pha must be a 2D array")

    n, m = pha_xp.shape

    x = xp.linspace(-m * dh / 2, m * dh / 2, m)
    y = xp.linspace(-n * dh / 2, n * dh / 2, n)
    xx, yy = xp.meshgrid(x, y)

    # Same curl stencil used by the original MATLAB implementation.
    pha_gy = xp.exp(1j * xp.diff(pha_xp, axis=0))
    pha_gx = xp.exp(1j * xp.diff(pha_xp, axis=1))

    gy = xp.vstack([xp.angle(pha_gy), xp.zeros((1, m), dtype=pha_xp.real.dtype)])
    gx = xp.hstack([xp.angle(pha_gx), xp.zeros((n, 1), dtype=pha_xp.real.dtype)])

    gy_m1 = xp.hstack([gy[:, 1:m], xp.zeros((n, 1), dtype=pha_xp.real.dtype)])
    gx_n1 = xp.vstack([gx[1:n, :], xp.zeros((1, m), dtype=pha_xp.real.dtype)])

    g_curl = gx + gy_m1 - gx_n1 - gy

    threshold = 2 * xp.pi - 0.1
    vor_po = (g_curl > threshold).astype(pha_xp.real.dtype)
    vor_ne = ((-g_curl) > threshold).astype(pha_xp.real.dtype)

    x_po = vor_po * xx
    y_po = vor_po * yy
    xv_po = x_po[x_po != 0] + dh / 2
    yv_po = y_po[y_po != 0] + dh / 2

    x_ne = vor_ne * xx
    y_ne = vor_ne * yy
    xv_ne = x_ne[x_ne != 0] + dh / 2
    yv_ne = y_ne[y_ne != 0] + dh / 2

    vor_po_phase = xp.zeros((n, m), dtype=pha_xp.real.dtype)
    for i in range(int(xv_po.size)):
        vor_single = xp.arctan2((yy - yv_po[i]), (xx - xv_po[i]))
        vor_po_phase = vor_po_phase + vor_single

    vor_ne_phase = xp.zeros((n, m), dtype=pha_xp.real.dtype)
    for i in range(int(xv_ne.size)):
        vor_single = xp.arctan2((yy - yv_ne[i]), (xx - xv_ne[i]))
        vor_ne_phase = vor_ne_phase + vor_single

    vor = vor_po_phase - vor_ne_phase
    pha_vfree = xp.mod(pha_xp - vor, 2 * xp.pi)

    if gather_output:
        return _to_numpy(pha_vfree)
    return pha_vfree


if __name__ == "__main__":
    # Quick synthetic sanity check: remove a center vortex from helical phase.
    nn = 128
    mm = 128
    dh = 3.74e-3

    yy, xx = np.mgrid[-1:1:complex(0, nn), -1:1:complex(0, mm)]
    pha = np.mod(np.arctan2(yy, xx), 2 * np.pi)

    pha_vfree = function_vortex_elimination_accegpu(pha, dh, use_cupy=False, gather_output=True)
    print(f"output shape: {pha_vfree.shape}")
    print(f"range: min={pha_vfree.min():.6f}, max={pha_vfree.max():.6f}")
