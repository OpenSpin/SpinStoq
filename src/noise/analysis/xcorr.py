"""Cross-correlation between two traces/sites, FFT-based, with lag axis."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

__all__ = ["cross_correlation", "xcorr"]


def _as_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[np.newaxis, :]
    elif x.ndim == 2:
        pass
    else:
        raise ValueError(f"unsupported input ndim {x.ndim}")
    return x


def cross_correlation(x: np.ndarray, y: np.ndarray, fs: float = 1.0,
                      biased: bool = True, max_lag: Optional[int] = None,
                      detrend: bool = True, normalize: bool = False
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Cross-correlation ``R_xy(tau)`` via FFT, ensemble-averaged.

    Returns the two-sided cross-correlation over lags ``[-max_lag, max_lag]``
    (or the full range by default), so the lag axis is symmetric around 0.

    Parameters
    ----------
    x, y : ndarray, shape ``(n_traj, n_time)`` or ``(n_time,)``
        Two traces (or ensembles of equal length). If both are 2D they must
        share the same leading axis.
    fs : float
        Sample rate [Hz] for the lag axis.
    biased : bool
        Normalize by ``N`` (biased) or ``N - |k|`` (unbiased).
    max_lag : int, optional
        Half-width of the returned lag window. Default: ``n_time - 1``.
    detrend : bool
        Subtract per-trajectory means.
    normalize : bool
        If True, return normalized cross-correlation coefficient (peak in [-1, 1]).
    """
    x = _as_2d(np.asarray(x, dtype=float))
    y = _as_2d(np.asarray(y, dtype=float))
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape, got {x.shape} vs {y.shape}")
    n_traj, n_time = x.shape
    if detrend:
        x = x - x.mean(axis=-1, keepdims=True)
        y = y - y.mean(axis=-1, keepdims=True)
    n_fft = 1
    while n_fft < 2 * n_time:
        n_fft *= 2
    X = np.fft.rfft(x, n=n_fft, axis=-1)
    Y = np.fft.rfft(y, n=n_fft, axis=-1)
    # cross-correlation = irfft(X * conj(Y)); convention: R_xy[k] = sum x[i] y[i+k]
    xcorr_full = np.fft.irfft(X * np.conj(Y), n=n_fft, axis=-1)
    # circular shift so that lag 0 is centered
    xcorr_full = np.fft.ifftshift(xcorr_full, axes=-1)
    if biased:
        xcorr_full = xcorr_full / n_time
    else:
        denom = np.arange(n_time, 0, -1, dtype=float)
        denom = np.concatenate([denom[::-1][:-1], denom])  # full two-sided
        xcorr_full = xcorr_full / denom[: xcorr_full.shape[-1]]
    R = xcorr_full.mean(axis=0)
    if normalize:
        sx = np.std(x)
        sy = np.std(y)
        denom = sx * sy
        denom = np.where(denom == 0, 1.0, denom)
        R = R / np.mean(denom)
    # build symmetric lag axis
    if max_lag is None:
        max_lag = n_time - 1
    max_lag = int(min(max_lag, n_time - 1))
    center = n_fft // 2
    lo = max(0, center - max_lag)
    hi = min(R.size, center + max_lag + 1)
    R = R[lo:hi]
    lags = np.arange(-max_lag, max_lag + 1) / fs
    return lags, R


def xcorr(x: np.ndarray, y: np.ndarray, fs: float = 1.0, **kw
          ) -> Tuple[np.ndarray, np.ndarray]:
    """Alias for :func:`cross_correlation`."""
    return cross_correlation(x, y, fs=fs, **kw)