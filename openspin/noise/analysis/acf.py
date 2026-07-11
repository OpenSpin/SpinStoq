"""Autocorrelation function via FFT (Wiener-Khinchin), ensemble-averaged."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

__all__ = ["autocorrelation", "acf"]


def _as_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[np.newaxis, :]
    elif x.ndim == 2:
        pass
    elif x.ndim == 3:
        x = x.reshape(-1, x.shape[-1])
    else:
        raise ValueError(f"unsupported input ndim {x.ndim}")
    return x


def autocorrelation(x: np.ndarray, fs: float = 1.0, biased: bool = True,
                    max_lag: Optional[int] = None, detrend: bool = True
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """One-sided autocorrelation ``R(tau)`` via FFT (Wiener-Khinchin).

    Parameters
    ----------
    x : ndarray, shape ``(n_traj, [n_sites,] n_time)`` or ``(n_time,)``
    fs : float
        Sample rate [Hz] for the lag axis.
    biased : bool
        If True, normalize by ``N`` (biased, decays to 0); if False by
        ``N - k`` (unbiased). Biased is the standard Wiener-Khinchin choice and
        is what the convention test expects.
    max_lag : int, optional
        Return only lags ``0..max_lag``. Default: all positive lags.
    detrend : bool
        Subtract the per-trajectory mean before estimating.

    Returns
    -------
    tau : ndarray, shape ``(n_lag,)``
        Lags [s].
    R : ndarray, shape ``(n_lag,)``
        Ensemble-averaged autocorrelation.
    """
    x = _as_2d(np.asarray(x, dtype=float))
    n_traj, n_time = x.shape
    if detrend:
        x = x - x.mean(axis=-1, keepdims=True)
    # zero-pad to >= 2N to avoid circular correlation
    n_fft = 1
    while n_fft < 2 * n_time:
        n_fft *= 2
    X = np.fft.rfft(x, n=n_fft, axis=-1)
    acf_full = np.fft.irfft(np.abs(X) ** 2, n=n_fft, axis=-1)[..., :n_time]
    if biased:
        acf_full = acf_full / n_time
    else:
        denom = np.arange(n_time, 0, -1, dtype=float)
        acf_full = acf_full / denom
    R = acf_full.mean(axis=0)
    if max_lag is not None:
        max_lag = int(min(max_lag, n_time - 1))
        R = R[: max_lag + 1]
    else:
        R = R[:n_time]
    tau = np.arange(R.size, dtype=float) / fs
    return tau, R


def acf(x: np.ndarray, fs: float = 1.0, **kw) -> Tuple[np.ndarray, np.ndarray]:
    """Alias for :func:`autocorrelation`."""
    return autocorrelation(x, fs=fs, **kw)