"""Shared helpers for the noise analysis Skills.

Handles loading traces from `.npy`/`.csv`/`.npz` files or accepting in-memory
`NoiseResult` / ndarray inputs, and saving figures + value arrays.
"""
from __future__ import annotations

import os
from typing import Tuple, Union

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL = True
except ImportError:  # pragma: no cover
    _MPL = False


def load_trace(path_or_obj, fs=None):
    """Load a trace from a file path or pass through an ndarray/NoiseResult.

    Returns ``(array, fs, spec)`` where `array` has shape
    `(n_traj, [n_sites,] n_time)` and `spec` is a dict (empty for raw arrays).
    """
    from openspin.noise import NoiseResult
    if isinstance(path_or_obj, NoiseResult):
        return path_or_obj.traj, path_or_obj.fs, path_or_obj.spec
    if isinstance(path_or_obj, np.ndarray):
        return path_or_obj, (fs if fs is not None else 1.0), {}
    if isinstance(path_or_obj, str):
        p = path_or_obj
        if p.endswith(".npz"):
            with np.load(p, allow_pickle=True) as z:
                if "traj" in z:
                    arr = z["traj"]
                    f = float(z["fs"]) if "fs" in z else (fs or 1.0)
                    spec = z["spec"].item() if "spec" in z and z["spec"].dtype == object else {}
                    return arr, f, spec
                # otherwise treat as a plain value archive
                keys = list(z.keys())
                arr = z[keys[0]]
                return arr, (fs or 1.0), {}
        if p.endswith(".csv"):
            arr = np.loadtxt(p, delimiter=",")
            if arr.ndim == 1:
                arr = arr[np.newaxis, :]
            return arr, (fs or 1.0), {}
        if p.endswith(".npy"):
            arr = np.load(p)
            return arr, (fs or 1.0), {}
    raise ValueError(f"cannot load trace from {path_or_obj!r}")


def save_figure(fig, output_dir, name):
    """Save a matplotlib figure to `output_dir/name.png` (creates the dir)."""
    if not _MPL:
        raise ImportError("matplotlib is required for figures: pip install openspin-noise[viz]")
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, name)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def save_values(output_dir, name, **arrays):
    """Save named arrays to `output_dir/name.npz`."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, name)
    np.savez(path, **arrays)
    return path


def setup_plot():
    """Return a fresh matplotlib figure (or raise if matplotlib missing)."""
    if not _MPL:
        raise ImportError("matplotlib is required for figures: pip install openspin-noise[viz]")
    return plt.figure(figsize=(7, 5))