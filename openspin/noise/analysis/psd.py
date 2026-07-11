"""One-sided power spectral density estimators (NumPy reimplementation).

Conventions (spec §5): one-sided ``S(f)`` for ``f >= 0`` in ``units^2 / Hz``
satisfying ``var(x) = integral_0^{fs/2} S(f) df``. All estimators are vectorized
over the leading trajectory axis and ensemble-averaged.
"""
from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

__all__ = ["welch", "periodogram", "multitaper", "psd"]


def _as_2d(x: np.ndarray) -> np.ndarray:
    """Coerce to shape ``(n_traj, n_time)``; 1D input -> single trajectory."""
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[np.newaxis, :]
    elif x.ndim == 2:
        pass
    elif x.ndim == 3:
        # spatial: flatten (n_traj*n_sites) for per-channel PSD
        x = x.reshape(-1, x.shape[-1])
    else:
        raise ValueError(f"unsupported input ndim {x.ndim}")
    return x


def _windows(n: int, name: str = "hann") -> np.ndarray:
    if name == "hann":
        return np.hanning(n)
    if name == "hamming":
        return np.hamming(n)
    if name == "blackman":
        return np.blackman(n)
    if name in ("rect", "rectangular", "boxcar"):
        return np.ones(n, dtype=float)
    raise ValueError(f"unknown window {name!r}")


def periodogram(x: np.ndarray, fs: float = 1.0,
                window: str = "rect", detrend: str = "mean"
                ) -> Tuple[np.ndarray, np.ndarray]:
    """One-sided periodogram, ensemble-averaged over the leading axis.

    Returns ``(f, S)`` with ``f`` shape ``(n_freq,)`` and ``S`` shape
    ``(n_freq,)`` (already averaged over trajectories).
    """
    x = _as_2d(np.asarray(x, dtype=float))
    n_traj, n_time = x.shape
    if detrend == "mean":
        x = x - x.mean(axis=-1, keepdims=True)
    elif detrend == "linear":
        t = np.arange(n_time, dtype=float)
        A = np.vstack([np.ones_like(t), t - t.mean()]).T
        coef, *_ = np.linalg.lstsq(A, x.T, rcond=None)
        x = x - (A @ coef).T
    win = _windows(n_time, window)
    w_norm = np.sum(win**2)
    X = np.fft.rfft(x * win, axis=-1)
    # one-sided PSD: |X|^2 / (fs * sum(w^2)); multiply by 2 except DC & Nyquist
    psd_raw = (np.abs(X) ** 2) / (fs * w_norm)
    n_freq = psd_raw.shape[-1]
    psd_raw[..., 1:-1 if n_time % 2 == 0 else n_freq] *= 2.0
    if n_time % 2 == 0 and n_freq > 1:
        psd_raw[..., -1] *= 1.0  # Nyquist stays single-sided (no factor 2)
    f = np.fft.rfftfreq(n_time, d=1.0 / fs)
    return f, psd_raw.mean(axis=0)


def welch(x: np.ndarray, fs: float = 1.0,
          nperseg: Optional[int] = None, noverlap: Optional[int] = None,
          window: str = "hann", detrend: str = "mean"
          ) -> Tuple[np.ndarray, np.ndarray]:
    """One-sided Welch PSD (segment, window, average), ensemble-averaged.

    Vectorized over the leading trajectory axis; segments are averaged within
    each trajectory, then across trajectories.
    """
    x = _as_2d(np.asarray(x, dtype=float))
    n_traj, n_time = x.shape
    if nperseg is None:
        # default: largest power of two <= n_time, but at least 256
        nperseg = int(2 ** np.floor(np.log2(max(n_time, 256))))
        nperseg = min(nperseg, n_time)
    nperseg = int(min(nperseg, n_time))
    if nperseg < 2:
        raise ValueError("nperseg must be >= 2")
    if noverlap is None:
        noverlap = nperseg // 2
    noverlap = int(min(noverlap, nperseg - 1))
    step = nperseg - noverlap
    win = _windows(nperseg, window)
    w_norm = np.sum(win**2)

    n_seg = 1 + (n_time - nperseg) // step
    if n_seg < 1:
        n_seg = 1
        nperseg = n_time
        step = 1
        win = _windows(nperseg, window)
        w_norm = np.sum(win**2)

    # Process segments with a loop over the (small) segment axis to avoid the
    # large contiguous copy that `as_strided + ascontiguousarray` would create.
    # n_seg is small (e.g. ~7 for 16384/4096/2), so the loop overhead is minimal
    # and we keep memory to one segment-batch at a time.
    f = np.fft.rfftfreq(nperseg, d=1.0 / fs)
    n_freq = f.size
    psd_sum = np.zeros(n_freq, dtype=float)
    n_count = 0
    t_lin = None
    if detrend == "linear":
        t_lin = np.arange(nperseg, dtype=float)
        t_lin = t_lin - t_lin.mean()
    for s in range(n_seg):
        i0 = s * step
        seg = x[:, i0:i0 + nperseg]  # view (n_traj, nperseg)
        if detrend == "mean":
            seg = seg - seg.mean(axis=-1, keepdims=True)
        elif detrend == "linear":
            denom = np.dot(t_lin, t_lin)
            slope = (seg @ t_lin) / denom
            seg = seg - slope[..., None] * t_lin[None, :]
            seg = seg - seg.mean(axis=-1, keepdims=True)
        seg = seg * win[None, :]
        X = np.fft.rfft(seg, axis=-1)
        # power = |X|^2 = X.real^2 + X.imag^2 (avoids abs temporary)
        psd_seg = (X.real ** 2 + X.imag ** 2) / (fs * w_norm)
        psd_seg[..., 1:-1 if nperseg % 2 == 0 else n_freq] *= 2.0
        psd_sum += psd_seg.sum(axis=0)
        n_count += n_traj
    psd_avg = psd_sum / n_count
    return f, psd_avg


def _dpss(n: int, nw: float = 4.0, k: int = None) -> np.ndarray:
    """Discrete prolate spheroidal (Slepian) sequences via the tridiagonal eigenproblem.

    Lightweight NumPy implementation (no SciPy dependency). Returns ``k`` tapers
    of length ``n``.
    """
    if k is None:
        k = int(2 * nw - 1)
    k = max(1, min(k, n))
    # tridiagonal covariance matrix of the DPSS eigenproblem
    idx = np.arange(n)
    diag = ((n - 1 - 2 * idx) ** 2) / 4.0
    off = idx[1:] * (n - idx[1:]) / 2.0
    # build full symmetric tridiagonal (n small for taper counts we use)
    M = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)
    M = M + nw * (nw + 1) * np.eye(n) * 0.0  # constant shift cancels in eigvecs
    vals, vecs = np.linalg.eigh(M)
    # eigenvalues sorted ascending; DPSS are the top-k (largest concentration)
    order = np.argsort(vals)[::-1][:k]
    tapers = vecs[:, order].T  # (k, n)
    # normalize each taper to unit energy
    tapers = tapers / np.sqrt(np.sum(tapers**2, axis=1, keepdims=True))
    return tapers


def multitaper(x: np.ndarray, fs: float = 1.0, nw: float = 4.0,
               k: Optional[int] = None, detrend: str = "mean"
               ) -> Tuple[np.ndarray, np.ndarray]:
    """One-sided multitaper PSD using DPSS (Slepian) tapers, ensemble-averaged."""
    x = _as_2d(np.asarray(x, dtype=float))
    n_traj, n_time = x.shape
    if detrend == "mean":
        x = x - x.mean(axis=-1, keepdims=True)
    tapers = _dpss(n_time, nw=nw, k=k)
    k = tapers.shape[0]
    # apply each taper to each trajectory: (n_traj, k, n_time)
    xt = x[:, None, :] * tapers[None, :, :]
    X = np.fft.rfft(xt, axis=-1)
    psd_k = (np.abs(X) ** 2) / (fs * np.sum(tapers**2, axis=1)[None, :, None])
    n_freq = psd_k.shape[-1]
    psd_k[..., 1:-1 if n_time % 2 == 0 else n_freq] *= 2.0
    psd_avg = psd_k.mean(axis=(0, 1))
    f = np.fft.rfftfreq(n_time, d=1.0 / fs)
    return f, psd_avg


def psd(x: np.ndarray, fs: float = 1.0, method: str = "welch", **kw
        ) -> Tuple[np.ndarray, np.ndarray]:
    """Dispatch to ``welch`` / ``periodogram`` / ``multitaper``."""
    if method == "welch":
        return welch(x, fs=fs, **kw)
    if method == "periodogram":
        return periodogram(x, fs=fs, **kw)
    if method == "multitaper":
        return multitaper(x, fs=fs, **kw)
    raise ValueError(f"unknown method {method!r}")