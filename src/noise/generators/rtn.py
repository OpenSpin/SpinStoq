"""Random telegraph noise (RTN) / two-level fluctuators.

Single and ensemble two-level fluctuators with switching rates ``gamma_up`` (down->up)
and ``gamma_down`` (up->down). Exact via exponentially-distributed dwell times,
then resampled onto the grid; vectorized over the ensemble. Building block for
the distributed-fluctuator spatial generator (§6.4b) and for non-Gaussian noise.
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np

from ..core import NoiseResult, make_generator, make_time_grid

__all__ = ["generate_rtn", "generate_rtn_ensemble"]


def _sample_rtn_trajectory(rng: np.random.Generator, t: np.ndarray,
                           gamma_up: float, gamma_down: float,
                           x_low: float, x_high: float,
                           state0: Optional[int] = None) -> np.ndarray:
    """Sample one RTN trajectory by drawing exponentially-distributed dwell times.

    Vectorized over the ensemble via the outer caller; this handles one trace.
    """
    t = np.asarray(t, dtype=float)
    n = t.size
    x = np.empty(n, dtype=float)
    # initial state: draw from stationary distribution P(up) = gamma_up / (gamma_up+gamma_down)
    p_up = gamma_up / (gamma_up + gamma_down) if (gamma_up + gamma_down) > 0 else 0.5
    state = int(state0) if state0 is not None else int(rng.random() < p_up)
    x[0] = x_high if state == 1 else x_low
    # walk forward by dwell-time intervals
    pos = 0
    t_curr = t[0]
    while pos < n - 1:
        rate = gamma_down if state == 1 else gamma_up
        if rate <= 0:
            # no switching: fill the rest with the current state
            x[pos + 1:] = x_high if state == 1 else x_low
            return x
        dwell = rng.exponential(1.0 / rate)
        t_next = t_curr + dwell
        # advance all grid points within [t_curr, t_next) at the current state
        while pos + 1 < n and t[pos + 1] < t_next:
            x[pos + 1] = x_high if state == 1 else x_low
            pos += 1
        state = 1 - state
        t_curr = t_next
    return x


def generate_rtn(n_traj: int = 1, fs: float = 1.0, n_points: int = 2**16,
                 gamma_up: float = 1.0, gamma_down: float = 1.0,
                 x_low: float = -1.0, x_high: float = 1.0,
                 dtype=np.float64, seed=None, t: Optional[np.ndarray] = None
                 ) -> NoiseResult:
    """Generate random-telegraph-noise trajectories.

    Parameters
    ----------
    gamma_up, gamma_down : float
        Switching rates ``gamma_up`` (low->high) and ``gamma_down`` (high->low)
        in ``1/s``. Stationary ``P(high) = gamma_up / (gamma_up + gamma_down)``.
    x_low, x_high : float
        The two telegraph levels.
    """
    t_arr, fs_eff, n = make_time_grid(fs=fs, n_points=n_points, t=t)
    rng = make_generator(seed)
    x = np.empty((n_traj, n), dtype=dtype)
    for i in range(n_traj):
        x[i] = _sample_rtn_trajectory(rng, t_arr, gamma_up, gamma_down, x_low, x_high)
    spec = {"process": "rtn", "gamma_up": gamma_up, "gamma_down": gamma_down,
            "x_low": x_low, "x_high": x_high, "fs": fs_eff, "n_points": n,
            "n_traj": n_traj, "dtype": str(dtype)}
    return NoiseResult(traj=x, t=t_arr, fs=fs_eff, spec=spec, seed=seed, units="a.u.")


def generate_rtn_ensemble(n_traj: int = 1, fs: float = 1.0, n_points: int = 2**16,
                          n_fluctuators: int = 100, gamma_up: float = 1.0,
                          gamma_down: float = 1.0, x_low: float = -1.0,
                          x_high: float = 1.0, dtype=np.float64, seed=None,
                          t: Optional[np.ndarray] = None) -> NoiseResult:
    """Generate an ensemble of independent RTN fluctuators and sum them.

    Returns shape ``(n_traj, n_time)`` where each trajectory is the sum of
    ``n_fluctuators`` independent telegraph processes. By the central limit
    theorem this approaches Gaussian noise for large ``n_fluctuators``, but keeps
    the non-Gaussian tails for small counts. Building block for §6.4(b).
    """
    t_arr, fs_eff, n = make_time_grid(fs=fs, n_points=n_points, t=t)
    rng = make_generator(seed)
    x = np.zeros((n_traj, n), dtype=dtype)
    for _ in range(n_fluctuators):
        for i in range(n_traj):
            x[i] += _sample_rtn_trajectory(rng, t_arr, gamma_up, gamma_down, x_low, x_high)
    spec = {"process": "rtn_ensemble", "n_fluctuators": n_fluctuators,
            "gamma_up": gamma_up, "gamma_down": gamma_down, "x_low": x_low,
            "x_high": x_high, "fs": fs_eff, "n_points": n, "n_traj": n_traj, "dtype": str(dtype)}
    return NoiseResult(traj=x, t=t_arr, fs=fs_eff, spec=spec, seed=seed, units="a.u.")