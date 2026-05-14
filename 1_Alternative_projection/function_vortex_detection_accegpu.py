"""Python translation of MATLAB function_vortex_detection_accegpu.m.

GPU-first implementation using CuPy when available, with NumPy fallback.
"""

from __future__ import annotations

from typing import Any, Tuple

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


def _to_python_int(value: Any) -> int:
    """Convert scalar from NumPy/CuPy to a Python int."""
    if hasattr(value, "item"):
        return int(value.item())
    return int(value)


def function_vortex_detection_accegpu(
    pha: ArrayLike,
    dh: float,
    *,
    use_cupy: bool | None = None,
) -> Tuple[int, int]:
    """Detect positive/negative phase vortex counts from a wrapped phase map.

    Args:
        pha: 2D phase array.
        dh: Sampling pitch (same meaning as MATLAB code).
        use_cupy: True to force CuPy, False to force NumPy, None to auto.

    Returns:
        (num_po, num_ne): positive and negative vortex counts.
    """
    xp = _get_array_module(pha, use_cupy)
    pha_xp = xp.asarray(pha)

    if pha_xp.ndim != 2:
        raise ValueError("pha must be a 2D array")

    n, m = pha_xp.shape

    x = xp.linspace(-m * dh / 2, m * dh / 2, m)
    y = xp.linspace(-n * dh / 2, n * dh / 2, n)
    xx, yy = xp.meshgrid(x, y)

    # Discrete phase-gradient curl (same stencil and padding as MATLAB script).
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
    _ = yv_po  # Kept for one-to-one readability with MATLAB variable flow.
    num_po = xv_po.size

    x_ne = vor_ne * xx
    y_ne = vor_ne * yy
    xv_ne = x_ne[x_ne != 0] + dh / 2
    yv_ne = y_ne[y_ne != 0] + dh / 2
    _ = yv_ne
    num_ne = xv_ne.size

    return _to_python_int(num_po), _to_python_int(num_ne)


if __name__ == "__main__":
    # Quick synthetic sanity check: one helical phase singularity at center.
    nn = 128
    mm = 128
    dh = 3.74e-3

    yy, xx = np.mgrid[-1:1:complex(0, nn), -1:1:complex(0, mm)]
    pha = np.mod(np.arctan2(yy, xx), 2 * np.pi)

    po, ne = function_vortex_detection_accegpu(pha, dh, use_cupy=False)
    print(f"positive vortexes: {po}")
    print(f"negative vortexes: {ne}")
