"""Named-process registry + presets (spec §11).

Maps names -> generator + default params, so ``generate("charge_noise_SiGe", ...)``
works. Ship a handful of literature-ish presets (a 1/f charge-noise default with a
sensible ``S0``, an OU-sum default, an RTN-dominated default). Presets are
overridable and documented with their assumptions/units.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np

from .core import NoiseResult
from .generators import (
    timmer_koenig, generate_ou, generate_ou_sum, generate_rtn,
    generate_spatial_separable, generate_spatial_fluctuators,
)

__all__ = ["REGISTRY", "PRESETS", "register", "get_preset", "list_processes"]


# Each entry: (generator_fn, default_kwargs, description)
PRESETS: Dict[str, dict] = {
    "1/f": {
        "fn": "timmer_koenig",
        "defaults": {"alpha": 1.0, "S0": 1.0, "f0": 1.0},
        "description": "Canonical 1/f^alpha via Timmer-König (method 1). S0 is the "
                       "one-sided PSD at f0 [units^2/Hz].",
        "units": "a.u.",
    },
    "ou": {
        "fn": "generate_ou",
        "defaults": {"gamma": 1.0, "sigma": 1.0, "F": 0.0},
        "description": "Exact scalar Ornstein-Uhlenbeck (method 2). Stationary "
                       "variance sigma^2/(2*gamma); ACF exp(-gamma|tau|).",
        "units": "a.u.",
    },
    "ou_sum": {
        "fn": "generate_ou_sum",
        "defaults": {"alpha": 1.0, "S0": 1.0, "f0": 1.0, "n_components_per_decade": 8},
        "description": "1/f^alpha as a sum of OU/Lorentzians (method 3). Physically "
                       "motivated, exactly stationary, extends to spatial correlations.",
        "units": "a.u.",
    },
    "rtn": {
        "fn": "generate_rtn",
        "defaults": {"gamma_up": 1.0, "gamma_down": 1.0, "x_low": -1.0, "x_high": 1.0},
        "description": "Random telegraph noise (two-level fluctuator). Non-Gaussian; "
                       "building block for spatial fluctuator ensembles.",
        "units": "a.u.",
    },
    "spatial_fluctuators": {
        "fn": "generate_spatial_fluctuators",
        "defaults": {"n_fluctuators": 1000, "kernel": "monopole", "fluctuator": "ou_sum"},
        "description": "Distributed-fluctuator spatial noise (method 4b). Physical "
                       "spatial + spectral structure via a coupling kernel.",
        "units": "a.u.",
    },
    # ---- literature-ish presets ----
    "charge_noise_SiGe": {
        "fn": "timmer_koenig",
        "defaults": {"alpha": 1.0, "S0": 1e-12, "f0": 1.0, "units": "V"},
        "description": "Si/SiGe spin qubit 1/f charge noise. S0 ~ 1e-12 V^2/Hz at 1 Hz "
                       "is a typical order of magnitude (overridable).",
        "units": "V",
    },
    "charge_noise_SiMOS": {
        "fn": "timmer_koenig",
        "defaults": {"alpha": 1.0, "S0": 5e-12, "f0": 1.0, "units": "V"},
        "description": "Si/SiO2 (MOS) spin qubit 1/f charge noise. Typically a fewx "
                       "higher than SiGe (overridable).",
        "units": "V",
    },
    "ou_sum_default": {
        "fn": "generate_ou_sum",
        "defaults": {"alpha": 1.0, "S0": 1e-12, "f0": 1.0, "n_components_per_decade": 8,
                     "units": "V"},
        "description": "OU-sum 1/f default with a sensible charge-noise S0.",
        "units": "V",
    },
    "rtn_dominated": {
        "fn": "generate_rtn",
        "defaults": {"gamma_up": 1e3, "gamma_down": 1e3, "x_low": -1e-6, "x_high": 1e-6,
                     "units": "V"},
        "description": "RTN-dominated noise default: symmetric switching at 1 kHz, "
                       "1 uV levels (overridable).",
        "units": "V",
    },
}


# Map preset "fn" strings to the actual callables
_FN_MAP: Dict[str, Callable] = {
    "timmer_koenig": timmer_koenig,
    "generate_ou": generate_ou,
    "generate_ou_sum": generate_ou_sum,
    "generate_rtn": generate_rtn,
    "generate_spatial_separable": generate_spatial_separable,
    "generate_spatial_fluctuators": generate_spatial_fluctuators,
}


def register(name: str, fn: Callable, defaults: Optional[dict] = None,
             description: str = "", units: str = "a.u.") -> None:
    """Register a new named process at runtime."""
    PRESETS[name] = {
        "fn": fn.__name__ if hasattr(fn, "__name__") else str(fn),
        "defaults": defaults or {},
        "description": description,
        "units": units,
        "_callable": fn,  # direct callable for runtime-registered functions
    }


def get_preset(name: str) -> dict:
    """Return the preset dict for ``name`` (raises KeyError if unknown)."""
    if name not in PRESETS:
        raise KeyError(f"unknown process {name!r}; known: {list_processes()}")
    return PRESETS[name]


def list_processes() -> list[str]:
    """Return the list of registered process names."""
    return sorted(PRESETS.keys())


def resolve_fn(preset: dict) -> Callable:
    """Resolve the callable for a preset (handles both string and direct-callable)."""
    if "_callable" in preset:
        return preset["_callable"]
    return _FN_MAP[preset["fn"]]


# Backwards-compatible alias
REGISTRY = PRESETS