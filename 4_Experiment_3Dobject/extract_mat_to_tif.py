from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

try:
    from scipy.io import loadmat
except Exception:  # pragma: no cover
    loadmat = None

try:
    import h5py
except Exception:  # pragma: no cover
    h5py = None


def normalize01(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float64, copy=False)
    mn = float(np.min(arr))
    mx = float(np.max(arr))
    den = mx - mn
    if den <= np.finfo(np.float64).eps:
        return np.zeros_like(arr, dtype=np.float64)
    return (arr - mn) / den


def to_uint8_image(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return (normalize01(img) * 255.0).round().astype(np.uint8)

    if img.ndim == 3:
        if img.shape[2] not in (1, 3, 4) and img.shape[0] in (1, 3, 4):
            img = np.transpose(img, (1, 2, 0))
        out = np.empty_like(img, dtype=np.uint8)
        for c in range(img.shape[2]):
            out[:, :, c] = (normalize01(img[:, :, c]) * 255.0).round().astype(np.uint8)
        return out

    raise ValueError(f"Unsupported F1 dimensions: {img.shape}")


def to_uint16_depth(depth: np.ndarray) -> np.ndarray:
    if depth.ndim > 2:
        depth = np.squeeze(depth)
    if depth.ndim != 2:
        raise ValueError(f"Unsupported D1 dimensions: {depth.shape}")
    return (normalize01(depth) * 65535.0).round().astype(np.uint16)


def load_mat_robust(mat_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if loadmat is not None:
        try:
            mat = loadmat(mat_path)
            if "F1" in mat and "D1" in mat:
                return np.array(mat["F1"]), np.array(mat["D1"])
        except NotImplementedError:
            pass
        except ValueError:
            pass

    if h5py is None:
        raise RuntimeError(
            "Could not read MAT file with scipy.io.loadmat, and h5py is not available."
        )

    with h5py.File(mat_path, "r") as f:
        if "F1" not in f or "D1" not in f:
            raise KeyError(f"F1 or D1 not found in {mat_path}")
        f1 = np.array(f["F1"])
        d1 = np.array(f["D1"])

    # MATLAB v7.3 HDF5 often stores dimensions in reversed order.
    if f1.ndim >= 2:
        f1 = np.transpose(f1, axes=tuple(reversed(range(f1.ndim))))
    if d1.ndim >= 2:
        d1 = np.transpose(d1, axes=tuple(reversed(range(d1.ndim))))

    return f1, d1


def save_tiff(arr: np.ndarray, out_path: Path) -> None:
    if arr.ndim == 2:
        Image.fromarray(arr).save(out_path, format="TIFF")
        return
    if arr.ndim == 3:
        if arr.shape[2] == 1:
            Image.fromarray(arr[:, :, 0]).save(out_path, format="TIFF")
            return
        Image.fromarray(arr).save(out_path, format="TIFF")
        return
    raise ValueError(f"Unsupported array dimensions for TIFF: {arr.shape}")


def process_folder(folder: Path) -> None:
    mat_path = folder / "object.mat"
    if not mat_path.is_file():
        print(f"[SKIP] object.mat not found in {folder}")
        return

    f1, d1 = load_mat_robust(mat_path)
    img8 = to_uint8_image(np.asarray(f1))
    dep16 = to_uint16_depth(np.asarray(d1))

    img_out = folder / "image_2d.tif"
    dep_out = folder / "depth_map.tif"
    save_tiff(img8, img_out)
    save_tiff(dep16, dep_out)

    print(f"[OK] {img_out}")
    print(f"[OK] {dep_out}")


def main() -> None:
    root = Path(__file__).resolve().parent
    targets = [root / "complex_amplitude", root / "phase_only"]
    for folder in targets:
        process_folder(folder)


if __name__ == "__main__":
    main()
