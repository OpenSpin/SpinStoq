"""Overlapping Allan deviation via cumulative sums (O(N) per tau)."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

__all__ = ["allan_deviation", "allan_variance", "allan"]


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


def allan_variance(x: np.ndarray, fs: float = 1.0,
                   taus: Optional[np.ndarray] = None, max_n: int = 100
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Overlapping Allan variance, ensemble-averaged.

    Uses the cumulative-sum ``O(N)`` per-tau estimator:

    .. math::
        \\sigma_A^2(\\tau) = \\frac{1}{2(N-2m)} \\sum_{i} (\\bar y_{i+m} - \\bar y_i)^2

    where ``\\bar y_i`` is the average over ``m`` consecutive samples and
    ``m = tau * fs``. Overlapping means all possible (overlapping) blocks are
    used, which is the IEEE-standard estimator with minimum variance.

    Parameters
    ----------
    x : ndarray, shape ``(n_traj, [n_sites,] n_time)`` or ``(n_time,)``
    fs : float
        Sample rate [Hz].
    taus : ndarray, optional
        Requested averaging times [s]. Default: log-spaced from ``1/fs`` to
        ``n_time/(2*fs)``.
    max_n : int
        Cap on the number of tau points (log-spaced subsampling). Default 100
        — enough for a clean log-log plot; increase for finer resolution.

    Returns
    -------
    taus : ndarray, shape ``(n_tau,)``
    avar : ndarray, shape ``(n_tau,)``
        Ensemble-averaged overlapping Allan variance.
    """
    x = _as_2d(np.asarray(x, dtype=float))
    n_traj, n_time = x.shape
    dt = 1.0 / fs
    if taus is None:
        n_max = max(1, n_time // 2)
        m_grid = np.unique(np.round(np.logspace(0, np.log10(n_max), min(max_n, 100))).astype(int))
        m_grid = m_grid[m_grid >= 1]
        taus = m_grid * dt
    else:
        taus = np.asarray(taus, dtype=float)
        m_grid = np.unique(np.maximum(1, np.round(taus * fs).astype(int)))
        taus = m_grid * dt

    # cumulative sum for O(N) block-mean differences.
    # Prepend a zero so csum[i] = sum(x[0:i]).
    csum = np.cumsum(x, axis=-1)
    csum = np.concatenate([np.zeros((n_traj, 1)), csum], axis=-1)

    avar = np.empty(m_grid.size, dtype=float)
    for idx, m in enumerate(m_grid):
        m = int(m)
        if 2 * m > n_time:
            avar[idx] = np.nan
            continue
        # Combined second-difference to avoid intermediate temporaries:
        #   diff_i = ybar_{i+m} - ybar_i
        #          = (csum[i+2m] - 2*csum[i+m] + csum[i]) / m
        # One allocation instead of three (block, diff, diff**2).
        d = (csum[:, 2 * m:] - 2.0 * csum[:, m:-m] + csum[:, :-2 * m]) / m
        # mean of squares = sum(d**2) / d.size, computed in one pass via np.einsum
        # to avoid creating a full d**2 temporary.
        n_diff = d.shape[-1]
        if n_diff <= 0:
            avar[idx] = np.nan
            continue
        avar[idx] = 0.5 * np.einsum("ti,ti->", d, d) / (n_traj * n_diff)
    return taus, avar


def allan_deviation(x: np.ndarray, fs: float = 1.0, **kw) -> Tuple[np.ndarray, np.ndarray]:
    """Overlapping Allan deviation ``sigma_A(tau) = sqrt(allan_variance)``."""
    taus, avar = allan_variance(x, fs=fs, **kw)
    return taus, np.sqrt(avar)


def allan(x: np.ndarray, fs: float = 1.0, deviation: bool = True, **kw
          ) -> Tuple[np.ndarray, np.ndarray]:
    """Dispatch returning deviation by default (set ``deviation=False`` for variance)."""
    if deviation:
        return allan_deviation(x, fs=fs, **kw)
    return allan_variance(x, fs=fs, **kw)