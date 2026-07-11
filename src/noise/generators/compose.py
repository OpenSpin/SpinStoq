"""Compose / mix noise processes: sums, drift, quasi-static offsets, mixtures.

Sum independent processes, add slow drift and quasi-static (per-shot frozen)
offsets, and mixtures (e.g. 1/f + white + a dominant TLS). One
:class:`NoiseResult` out. All operations are vectorized over the ensemble.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Union

import numpy as np

from ..core import NoiseResult, make_generator, make_time_grid

__all__ = ["compose", "add_drift", "add_quasistatic", "add_white"]


def _coerce_results(results: Sequence[Union[NoiseResult, np.ndarray]],
                    t: Optional[np.ndarray] = None, fs: Optional[float] = None
                    ) -> List[np.ndarray]:
    """Extract trajectory arrays from a mix of NoiseResult / ndarray, checking shapes."""
    out = []
    ref_t = None
    for r in results:
        if isinstance(r, NoiseResult):
            arr = r.traj
            if ref_t is None:
                ref_t = r.t
            elif not np.allclose(ref_t, r.t, rtol=1e-9, atol=1e-12):
                raise ValueError("all processes must share the same time grid to compose")
        else:
            arr = np.asarray(r, dtype=float)
        out.append(arr)
    return out, ref_t


def compose(processes: Sequence[Union[NoiseResult, np.ndarray]],
            weights: Optional[Sequence[float]] = None,
            drift_rate: float = 0.0, quasistatic_std: float = 0.0,
            seed=None, t: Optional[np.ndarray] = None, fs: Optional[float] = None,
            units: str = "a.u.") -> NoiseResult:
    """Sum independent noise processes, optionally adding drift and quasi-static offsets.

    Parameters
    ----------
    processes : sequence of NoiseResult or ndarray
        Processes to sum. All must share the same time grid (or be raw arrays of
        identical shape). Arrays are interpreted as ``(n_traj, [n_sites,] n_time)``.
    weights : sequence of float, optional
        Per-process amplitude weights (default all 1).
    drift_rate : float
        Linear drift slope [units/s] added to every trajectory (same for all).
    quasistatic_std : float
        Std of a per-shot frozen offset drawn from ``N(0, quasistatic_std^2)``
        and added to each trajectory (constant in time within a shot).
    """
    arrays, ref_t = _coerce_results(processes, t=t, fs=fs)
    if weights is None:
        weights = np.ones(len(arrays))
    else:
        weights = np.asarray(weights, dtype=float)
    if weights.size != len(arrays):
        raise ValueError("weights length must match number of processes")

    # broadcast / align shapes
    ref_shape = arrays[0].shape
    n_traj = ref_shape[0]
    n_time = ref_shape[-1]
    n_sites = ref_shape[1] if len(ref_shape) == 3 else 0
    total = np.zeros(ref_shape, dtype=float)
    for w, arr in zip(weights, arrays):
        total = total + w * arr

    # time grid
    if ref_t is not None:
        t_arr = ref_t
        fs_eff = 1.0 / np.mean(np.diff(t_arr)) if t_arr.size > 1 else float("nan")
    elif t is not None:
        t_arr = np.asarray(t, dtype=float)
        fs_eff = 1.0 / np.mean(np.diff(t_arr)) if t_arr.size > 1 else float("nan")
    elif fs is not None:
        t_arr = np.arange(n_time, dtype=float) / fs
        fs_eff = float(fs)
    else:
        t_arr = np.arange(n_time, dtype=float)
        fs_eff = 1.0

    rng = make_generator(seed)
    # linear drift
    if drift_rate != 0.0:
        drift = drift_rate * t_arr
        if n_sites:
            total = total + drift[None, None, :]
        else:
            total = total + drift[None, :]
    # quasi-static per-shot offset
    if quasistatic_std > 0.0:
        if n_sites:
            offset = quasistatic_std * rng.standard_normal(size=(n_traj, n_sites, 1))
        else:
            offset = quasistatic_std * rng.standard_normal(size=(n_traj, 1))
        total = total + offset

    spec = {"process": "compose", "n_components": len(arrays), "weights": weights.tolist(),
            "drift_rate": drift_rate, "quasistatic_std": quasistatic_std,
            "fs": fs_eff, "n_points": n_time, "n_traj": n_traj, "dtype": str(total.dtype)}
    return NoiseResult(traj=total, t=t_arr, fs=fs_eff, spec=spec, seed=seed, units=units)


def add_drift(result: NoiseResult, drift_rate: float) -> NoiseResult:
    """Return a copy of ``result`` with a linear drift added."""
    drift = drift_rate * result.t
    traj = result.traj.copy()
    if traj.ndim == 3:
        traj = traj + drift[None, None, :]
    else:
        traj = traj + drift[None, :]
    spec = {**result.spec, "drift_rate": drift_rate}
    return NoiseResult(traj=traj, t=result.t, fs=result.fs, spec=spec,
                       seed=result.seed, units=result.units)


def add_quasistatic(result: NoiseResult, std: float, seed=None) -> NoiseResult:
    """Return a copy of ``result`` with a per-shot frozen offset added."""
    rng = make_generator(seed)
    traj = result.traj.copy()
    if traj.ndim == 3:
        offset = std * rng.standard_normal(size=(traj.shape[0], traj.shape[1], 1))
    else:
        offset = std * rng.standard_normal(size=(traj.shape[0], 1))
    traj = traj + offset
    spec = {**result.spec, "quasistatic_std": std}
    return NoiseResult(traj=traj, t=result.t, fs=result.fs, spec=spec,
                       seed=result.seed, units=result.units)


def add_white(result: NoiseResult, std: float, seed=None) -> NoiseResult:
    """Return a copy of ``result`` with additive white noise added."""
    rng = make_generator(seed)
    traj = result.traj.copy()
    white = std * rng.standard_normal(size=traj.shape)
    traj = traj + white
    spec = {**result.spec, "white_std": std}
    return NoiseResult(traj=traj, t=result.t, fs=result.fs, spec=spec,
                       seed=result.seed, units=result.units)