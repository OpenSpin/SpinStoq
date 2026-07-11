"""src.noise — fast correlated (temporal & spatial) charge/qubit noise generator.

Public API (spec §4):

    from src.noise import generate, calibrate, NoiseResult

    # by name
    res = generate("1/f", n_traj=1000, fs=1e4, n_points=2**16,
                   alpha=1.0, S0=1e-12, f0=1.0, seed=0, backend="numpy")
    res.traj   # (n_traj, n_time)
    res.t, res.fs, res.spec, res.seed, res.units

    # from a measured trace (estimate + resynthesize)
    gen = calibrate(trace, fs=1e4, method="circulant")
    res = gen.sample(n_traj=1000, n_points=2**16, seed=0)

Design principles (spec §2): NumPy-first; optional numba/jax acceleration behind
one API; vectorize over the ensemble; exact discrete-time schemes; reproducible
seeding via ``SeedSequence``; round-trip validated.
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np

from .core import (
    NoiseResult, make_generator, spawn_seeds, make_time_grid,
    units_helpers, convert_units, DEFAULT_FLOAT,
)
from .calibrate import calibrate, SurrogateGenerator
from .registry import PRESETS, register, get_preset, list_processes, resolve_fn
from .generators import (
    timmer_koenig, circulant_embedding, generate_spectral,
    generate_ou, generate_ou_multivariate,
    generate_ou_sum, fit_ou_weights, lorentzian_psd,
    generate_spatial_separable, generate_spatial_fluctuators,
    generate_rtn, generate_rtn_ensemble,
    compose, add_drift, add_quasistatic, add_white,
)
from .analysis import (
    welch, periodogram, multitaper, psd,
    autocorrelation, acf,
    allan_variance, allan_deviation, allan,
    cross_correlation, xcorr,
    cross_spectrum, coherence, cross_spectral_density,
)
from .backends import list_backends, get_backend

__version__ = "0.1.0"

__all__ = [
    # public API
    "generate", "calibrate", "NoiseResult",
    # core
    "make_generator", "spawn_seeds", "make_time_grid",
    "units_helpers", "convert_units", "DEFAULT_FLOAT",
    # registry
    "PRESETS", "register", "get_preset", "list_processes",
    # generators
    "timmer_koenig", "circulant_embedding", "generate_spectral",
    "generate_ou", "generate_ou_multivariate",
    "generate_ou_sum", "fit_ou_weights", "lorentzian_psd",
    "generate_spatial_separable", "generate_spatial_fluctuators",
    "generate_rtn", "generate_rtn_ensemble",
    "compose", "add_drift", "add_quasistatic", "add_white",
    # analysis
    "welch", "periodogram", "multitaper", "psd",
    "autocorrelation", "acf",
    "allan_variance", "allan_deviation", "allan",
    "cross_correlation", "xcorr",
    "cross_spectrum", "coherence", "cross_spectral_density",
    # backends
    "list_backends", "get_backend",
    "SurrogateGenerator",
]


def generate(name_or_trace: Union[str, np.ndarray], *,
             n_traj: int = 1, fs: Optional[float] = None,
             n_points: Optional[int] = None, t: Optional[np.ndarray] = None,
             seed=None, backend: str = "numpy", dtype=np.float64,
             **kwargs) -> NoiseResult:
    """Generate correlated noise trajectories by name or from a measured trace.

    Dispatches:
    * **string** -> look up the named process in the registry (:mod:`registry`),
      apply preset defaults, and call the generator with user overrides.
    * **array** -> ``calibrate(name_or_trace, fs=fs).sample(n_traj, n_points, seed)``
      (model-free surrogate, spec §7).

    Accept either ``fs`` + ``n_points`` (uniform grid) or an explicit ``t=`` array
    (arbitrary/non-uniform grid; honored by OU-family generators).

    Parameters
    ----------
    name_or_trace : str or ndarray
        Process name (e.g. ``"1/f"``, ``"ou"``, ``"ou_sum"``, ``"rtn"``,
        ``"spatial_fluctuators"``, ``"charge_noise_SiGe"``) or a measured trace.
    n_traj : int
        Number of trajectories to generate.
    fs : float
        Sample rate [Hz] (uniform grid).
    n_points : int
        Number of time points (uniform grid).
    t : ndarray, optional
        Explicit time grid [s] (arbitrary/non-uniform). Overrides ``fs``/``n_points``.
    seed : int, Sequence[int], SeedSequence, or Generator
        Reproducibility seed.
    backend : {"numpy", "numba", "jax"}
        Implementation of the OU time recursion (other generators are pure NumPy).
    dtype : np.dtype
        Output dtype (``float64`` default; ``float32`` for memory savings).
    **kwargs
        Process-specific parameters (override preset defaults). E.g. for ``"1/f"``:
        ``alpha``, ``S0``, ``f0``, ``f_min``.
    """
    # ---- from a measured trace ----
    if isinstance(name_or_trace, np.ndarray):
        if fs is None:
            raise ValueError("fs is required when generating from a trace")
        gen = calibrate(name_or_trace, fs=fs, method="circulant")
        n_pts = n_points if n_points is not None else name_or_trace.shape[-1]
        return gen.sample(n_traj=n_traj, n_points=n_pts, seed=seed, dtype=dtype)

    # ---- by name ----
    if not isinstance(name_or_trace, str):
        raise TypeError(f"name_or_trace must be a str or ndarray, got {type(name_or_trace)}")

    name = name_or_trace
    preset = get_preset(name)
    fn = resolve_fn(preset)

    # merge defaults with user overrides (user wins)
    params = dict(preset["defaults"])
    # pull units out of defaults (not a generator param)
    units = params.pop("units", preset.get("units", "a.u."))
    params.update(kwargs)

    # common args
    common = {"n_traj": n_traj, "dtype": dtype, "seed": seed}
    if t is not None:
        common["t"] = t
    else:
        if fs is None:
            raise ValueError(f"fs is required for process {name!r} (or pass t=)")
        common["fs"] = fs
        if n_points is None:
            n_points = 2**16
        common["n_points"] = n_points

    # backend only applies to OU-family generators
    if name in ("ou", "ou_sum") or "ou" in name or "backend" in params:
        common["backend"] = backend

    result = fn(**{**params, **common})
    # ensure units and process name are recorded
    if not isinstance(result, NoiseResult):
        raise TypeError(f"generator for {name!r} returned {type(result)}, not NoiseResult")
    result.units = units
    result.spec.setdefault("process", name)
    result.spec.setdefault("preset", name)
    return result