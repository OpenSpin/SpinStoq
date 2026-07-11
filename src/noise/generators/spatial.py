"""Method 4 — spatially correlated noise (efficient).

Two fast paths:

**(a) Separable covariance** ``C(i,j;tau) = K_ij * R(tau)`` — recommended default
when you just need ``M`` correlated sites. Generate ``M`` independent temporal
traces ``W`` (shape ``(n_traj, M, n_time)``) with the desired temporal spectrum
via §6.1 or §6.3, then mix: ``Y = chol(K) @ W`` along the site axis. ``K`` is
``M x M`` (M = number of dots/sensors, small) -> one tiny Cholesky + one
``einsum``. Cost is dominated by the temporal generation, not the spatial mixing.

**(b) Distributed-fluctuator ensemble** — recommended when spatial *and* spectral
structure must be physical (realistic charge noise). Place ``N_f`` two-level
fluctuators (or OU sources) at random positions ``r_f``; each couples to site
``i`` via a kernel ``g_{i,f} = kernel(|x_i - r_f|)`` (monopole ``1/r``, dipole
``1/r^3``, or screened). Site noise ``n_i(t) = sum_f g_{i,f} s_f(t)``, i.e.
``N = G @ S`` where ``G`` is ``(M, N_f)`` and ``S`` is ``(N_f, n_time)`` from the
RTN/OU batch engines. One matmul -> naturally spatially correlated 1/f with the
right cross-spectra. Fully vectorized; ``N_f`` up to ~10^4 is a single BLAS matmul.
"""
from __future__ import annotations

from typing import Callable, Optional, Union

import numpy as np

from ..core import NoiseResult, make_generator, make_time_grid
from .spectral import timmer_koenig
from .ou_sum import generate_ou_sum
from .rtn import generate_rtn_ensemble

__all__ = ["generate_spatial_separable", "generate_spatial_fluctuators"]


def generate_spatial_separable(K: np.ndarray, n_traj: int = 1, fs: float = 1.0,
                               n_points: int = 2**16, temporal: str = "1/f",
                               dtype=np.float64, seed=None,
                               t: Optional[np.ndarray] = None, **temporal_kw
                               ) -> NoiseResult:
    """Spatially correlated noise via a separable covariance ``K_ij * R(tau)``.

    Parameters
    ----------
    K : ndarray, shape ``(M, M)``
        Spatial covariance (positive-definite). ``K[i, j] = Cov(site_i, site_j)``.
    temporal : {"1/f", "ou_sum"}
        Temporal spectrum engine for the independent traces before mixing.
    temporal_kw : dict
        Forwarded to the temporal generator (e.g. ``alpha``, ``S0``, ``f0``).
    """
    K = np.asarray(K, dtype=float)
    M = K.shape[0]
    if K.shape != (M, M):
        raise ValueError("K must be (M, M)")
    # Cholesky of the spatial covariance
    try:
        L_K = np.linalg.cholesky(0.5 * (K + K.T))
    except np.linalg.LinAlgError:
        L_K = np.linalg.cholesky(0.5 * (K + K.T) + 1e-12 * np.eye(M))

    # generate M independent temporal traces: shape (n_traj, M, n_time)
    # do it as one batch of n_traj*M trajectories, then reshape
    t_arr, fs_eff, n = make_time_grid(fs=fs, n_points=n_points, t=t)
    rng = make_generator(seed)
    if temporal == "1/f":
        res = timmer_koenig(n_traj=n_traj * M, fs=fs_eff, n_points=n, seed=rng,
                            dtype=dtype, **temporal_kw)
        W = res.traj.reshape(n_traj, M, n)
    elif temporal == "ou_sum":
        res = generate_ou_sum(n_traj=n_traj * M, fs=fs_eff, n_points=n, seed=rng,
                              dtype=dtype, t=t_arr, **temporal_kw)
        W = res.traj.reshape(n_traj, M, n)
    else:
        raise ValueError(f"unknown temporal engine {temporal!r}")

    # mix along the site axis: Y = einsum('ij,tjk->tik', L_K, W)
    Y = np.einsum("ij,tjk->tik", L_K, W, optimize=True).astype(dtype)

    spec = {"process": "spatial_separable", "method": "separable_cholesky",
            "M": M, "temporal": temporal, "fs": fs_eff, "n_points": n,
            "n_traj": n_traj, "dtype": str(dtype), "temporal_kw": temporal_kw}
    return NoiseResult(traj=Y, t=t_arr, fs=fs_eff, spec=spec, seed=seed, units="a.u.")


def _kernel_monopole(r: np.ndarray) -> np.ndarray:
    return 1.0 / np.maximum(r, 1e-12)


def _kernel_dipole(r: np.ndarray) -> np.ndarray:
    return 1.0 / np.maximum(r, 1e-12) ** 3


def _kernel_screened(r: np.ndarray, lam: float = 1.0) -> np.ndarray:
    return np.exp(-r / lam) / np.maximum(r, 1e-12)


def generate_spatial_fluctuators(site_positions: np.ndarray, n_traj: int = 1,
                                 fs: float = 1.0, n_points: int = 2**16,
                                 n_fluctuators: int = 1000, kernel: str = "monopole",
                                 kernel_params: Optional[dict] = None,
                                 fluctuator: str = "ou_sum", dtype=np.float64,
                                 seed=None, t: Optional[np.ndarray] = None,
                                 **fluctuator_kw) -> NoiseResult:
    """Spatially correlated noise from a distributed-fluctuator ensemble.

    Parameters
    ----------
    site_positions : ndarray, shape ``(M,)`` or ``(M, d)``
        Positions of the ``M`` sites (dots/sensors).
    n_fluctuators : int
        Number ``N_f`` of distributed fluctuators.
    kernel : {"monopole", "dipole", "screened"}
        Spatial coupling kernel ``g_{i,f} = kernel(|x_i - r_f|)``.
    fluctuator : {"ou_sum", "rtn"}
        Temporal engine for the fluctuator traces.
    """
    site_positions = np.atleast_2d(np.asarray(site_positions, dtype=float))
    M = site_positions.shape[0]
    kernel_params = kernel_params or {}
    t_arr, fs_eff, n = make_time_grid(fs=fs, n_points=n_points, t=t)
    rng = make_generator(seed)

    # place fluctuators uniformly in a box spanning [min, max] of site positions, padded
    pad = 0.5 * (site_positions.max(axis=0) - site_positions.min(axis=0)).max()
    lo = site_positions.min(axis=0) - pad
    hi = site_positions.max(axis=0) + pad
    r_f = rng.uniform(lo, hi, size=(n_fluctuators, site_positions.shape[1]))

    # coupling matrix G[i, f] = kernel(|x_i - r_f|)
    if kernel == "monopole":
        kfn = _kernel_monopole
    elif kernel == "dipole":
        kfn = _kernel_dipole
    elif kernel == "screened":
        kfn = lambda r: _kernel_screened(r, **kernel_params)
    elif callable(kernel):
        kfn = kernel
    else:
        raise ValueError(f"unknown kernel {kernel!r}")

    G = np.empty((M, n_fluctuators), dtype=float)
    for i in range(M):
        dist = np.linalg.norm(site_positions[i] - r_f, axis=1)
        G[i] = kfn(dist)
    # normalize so each site has unit-variance contribution before fluctuator scaling
    G = G / np.sqrt(np.mean(G**2, axis=1, keepdims=True))

    # generate fluctuator traces S: (n_traj, N_f, n_time)
    if fluctuator == "ou_sum":
        res = generate_ou_sum(n_traj=n_traj * n_fluctuators, fs=fs_eff, n_points=n,
                              seed=rng, dtype=dtype, t=t_arr, **fluctuator_kw)
        S = res.traj.reshape(n_traj, n_fluctuators, n)
    elif fluctuator == "rtn":
        # use a single summed ensemble per trajectory is not what we want here;
        # we need per-fluctuator traces. Generate them as n_traj*N_f single RTNs.
        S = np.empty((n_traj, n_fluctuators, n), dtype=dtype)
        from .rtn import _sample_rtn_trajectory
        kw = {k: v for k, v in fluctuator_kw.items()
              if k in ("gamma_up", "gamma_down", "x_low", "x_high")}
        for ti in range(n_traj):
            for fi in range(n_fluctuators):
                S[ti, fi] = _sample_rtn_trajectory(rng, t_arr, **kw)
    else:
        raise ValueError(f"unknown fluctuator engine {fluctuator!r}")

    # site noise N = G @ S : (n_traj, M, n_time) via einsum
    N = np.einsum("mf,tfp->tmp", G, S, optimize=True).astype(dtype)

    spec = {"process": "spatial_fluctuators", "method": "distributed_fluctuators",
            "M": M, "n_fluctuators": n_fluctuators, "kernel": kernel,
            "fluctuator": fluctuator, "fs": fs_eff, "n_points": n,
            "n_traj": n_traj, "dtype": str(dtype), "fluctuator_kw": fluctuator_kw}
    return NoiseResult(traj=N, t=t_arr, fs=fs_eff, spec=spec, seed=seed, units="a.u.")