"""Noise generators for ``openspin.noise``.

Methods (spec §6):
* :mod:`spectral` — 1/f via Timmer-König + circulant embedding (method 1)
* :mod:`ou` — scalar + multivariate exact OU (method 2)
* :mod:`ou_sum` — 1/f as a sum of OU / Lorentzians (method 3)
* :mod:`spatial` — separable + fluctuator-ensemble spatial (method 4)
* :mod:`rtn` — random telegraph / two-level fluctuators (extra)
* :mod:`compose` — sums/mixtures, drift, quasi-static
"""
from .spectral import timmer_koenig, circulant_embedding, generate_spectral
from .ou import generate_ou, generate_ou_multivariate
from .ou_sum import generate_ou_sum, fit_ou_weights, lorentzian_psd
from .spatial import generate_spatial_separable, generate_spatial_fluctuators
from .rtn import generate_rtn, generate_rtn_ensemble
from .compose import compose, add_drift, add_quasistatic, add_white

__all__ = [
    "timmer_koenig", "circulant_embedding", "generate_spectral",
    "generate_ou", "generate_ou_multivariate",
    "generate_ou_sum", "fit_ou_weights", "lorentzian_psd",
    "generate_spatial_separable", "generate_spatial_fluctuators",
    "generate_rtn", "generate_rtn_ensemble",
    "compose", "add_drift", "add_quasistatic", "add_white",
]