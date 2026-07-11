"""Cross-spectral density, magnitude-squared coherence, and phase."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

__all__ = ["cross_spectrum", "coherence", "cross_spectral_density"]


def _as_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[np.newaxis, :]
    elif x.ndim == 2:
        pass
    else:
        raise ValueError(f"unsupported input ndim {x.ndim}")
    return x


def _welch_segments(x: np.ndarray, nperseg: int, noverlap: int, window: str
                    ) -> np.ndarray:
    """Return segments of shape ``(n_traj, n_seg, nperseg)`` (mean-detrended)."""
    n_traj, n_time = x.shape
    step = nperseg - noverlap
    n_seg = 1 + (n_time - nperseg) // step
    shape = (n_traj, n_seg, nperseg)
    strides = (x.strides[0], step * x.strides[1], x.strides[1])
    segs = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    segs = np.ascontiguousarray(segs)
    segs = segs - segs.mean(axis=-1, keepdims=True)
    if window == "hann":
        w = np.hanning(nperseg)
    elif window == "hamming":
        w = np.hamming(nperseg)
    elif window in ("rect", "rectangular", "boxcar"):
        w = np.ones(nperseg, dtype=float)
    else:
        raise ValueError(f"unknown window {window!r}")
    return segs * w[None, None, :], w


def cross_spectrum(x: np.ndarray, y: np.ndarray, fs: float = 1.0,
                   nperseg: Optional[int] = None, noverlap: Optional[int] = None,
                   window: str = "hann"
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """One-sided cross-spectral density ``S_xy(f)``, ensemble-averaged.

    Returns ``(f, S_xy)`` where ``S_xy`` is complex. The magnitude-squared
    coherence and phase are derived from this plus the auto-PSDs (see
    :func:`coherence`).
    """
    x = _as_2d(np.asarray(x, dtype=float))
    y = _as_2d(np.asarray(y, dtype=float))
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape, got {x.shape} vs {y.shape}")
    n_traj, n_time = x.shape
    if nperseg is None:
        nperseg = int(2 ** np.floor(np.log2(max(n_time, 256))))
        nperseg = min(nperseg, n_time)
    nperseg = int(min(nperseg, n_time))
    if noverlap is None:
        noverlap = nperseg // 2
    noverlap = int(min(noverlap, nperseg - 1))
    segs_x, w = _welch_segments(x, nperseg, noverlap, window)
    segs_y, _ = _welch_segments(y, nperseg, noverlap, window)
    w_norm = np.sum(w**2)
    X = np.fft.rfft(segs_x, axis=-1)
    Y = np.fft.rfft(segs_y, axis=-1)
    Pxy = (X * np.conj(Y)) / (fs * w_norm)
    n_freq = Pxy.shape[-1]
    Pxy[..., 1:-1 if nperseg % 2 == 0 else n_freq] *= 2.0
    Pxy_avg = Pxy.mean(axis=(0, 1))
    f = np.fft.rfftfreq(nperseg, d=1.0 / fs)
    return f, Pxy_avg


def coherence(x: np.ndarray, y: np.ndarray, fs: float = 1.0, **kw
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Magnitude-squared coherence ``gamma^2(f)`` and cross-spectral phase.

    Returns ``(f, gamma2, phase, S_xy)`` where:
    * ``gamma2 = |S_xy|^2 / (S_xx * S_yy)`` in ``[0, 1]``
    * ``phase = angle(S_xy)`` in radians
    * ``S_xy`` is the (complex) cross-spectral density
    """
    from .psd import welch
    f, Pxy = cross_spectrum(x, y, fs=fs, **kw)
    _, Pxx = welch(x, fs=fs, **kw)
    _, Pyy = welch(y, fs=fs, **kw)
    denom = Pxx * Pyy
    denom = np.where(denom == 0, 1.0, denom)
    gamma2 = (np.abs(Pxy) ** 2) / denom
    phase = np.angle(Pxy)
    return f, gamma2, phase, Pxy


def cross_spectral_density(x: np.ndarray, y: np.ndarray, fs: float = 1.0, **kw
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """Alias for :func:`cross_spectrum`."""
    return cross_spectrum(x, y, fs=fs, **kw)