"""NumPy backend for the OU time recursion (the only sequential kernel).

Implements the exact exponential OU propagator (spec §6.2) for both the scalar
and the multivariate/coupled case. The recursion over time is the single hot
path that the numba/jax backends accelerate; everything else is already fully
vectorized NumPy.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

__all__ = ["ou_scalar_step_numpy", "ou_scalar_recursion_numpy",
           "ou_multivariate_recursion_numpy", "available"]


def available() -> bool:
    return True


# --------------------------------------------------------------------------- #
# Scalar OU: x[n+1] = a[n]*x[n] + b[n] + s[n]*Z[n]
#   a[n] = exp(-gamma * dt[n])
#   b[n] = (F/gamma) * (1 - a[n])          (exact deterministic drive term)
#   s[n] = sigma * sqrt((1 - a[n]^2) / (2*gamma))
# Vectorized over (n_traj, n_components); only loop is over time.
# --------------------------------------------------------------------------- #
def ou_scalar_recursion_numpy(a: np.ndarray, b: np.ndarray, s: np.ndarray,
                              z: np.ndarray, x0: np.ndarray) -> np.ndarray:
    """Run the exact scalar OU recursion.

    Parameters
    ----------
    a, b, s : ndarray, shape ``(n_time-1,)`` or ``(n_time-1, n_comp)``
        Per-step propagator coefficients (broadcastable to the ensemble).
    z : ndarray, shape ``(n_traj, [n_comp,] n_time-1)``
        Standard-normal innovations.
    x0 : ndarray, shape ``(n_traj, [n_comp,])``
        Initial state (drawn from the stationary distribution by the caller).

    Returns
    -------
    x : ndarray, shape ``(n_traj, [n_comp,] n_time)``
    """
    n_time = z.shape[-1] + 1
    x = np.empty(z.shape[:-1] + (n_time,), dtype=np.result_type(z, x0, a, b, s))
    x[..., 0] = x0
    # broadcast coefficients to the ensemble shape for the per-step op
    a_b = np.broadcast_to(a, z.shape) if a.shape != z.shape else a
    b_b = np.broadcast_to(b, z.shape) if b.shape != z.shape else b
    s_b = np.broadcast_to(s, z.shape) if s.shape != z.shape else s
    for n in range(n_time - 1):
        x[..., n + 1] = a_b[..., n] * x[..., n] + b_b[..., n] + s_b[..., n] * z[..., n]
    return x


def ou_scalar_step_numpy(x: np.ndarray, a: float, b: float, s: float, z: float) -> np.ndarray:
    """Single exact OU step (scalar coefficients). Convenience helper."""
    return a * x + b + s * z


# --------------------------------------------------------------------------- #
# Multivariate / coupled OU:
#   x[n+1] = A[n] @ x[n] + d[n] + L[n] @ Z[n]
#   A[n] = expm(-B dt[n]),  d[n] = B^{-1}(I - A[n]) F,  Cov(eta[n]) = Sigma_ss - A Sigma_ss A^T
# Vectorized over n_traj; loop over time. Per-step cost: one matvec + one matmul.
# --------------------------------------------------------------------------- #
def ou_multivariate_recursion_numpy(A: np.ndarray, d: np.ndarray, L: np.ndarray,
                                    z: np.ndarray, x0: np.ndarray) -> np.ndarray:
    """Run the exact multivariate OU recursion.

    Parameters
    ----------
    A : ndarray, shape ``(n_time-1, dim, dim)`` or ``(dim, dim)``
        Per-step (or constant) state-transition matrices.
    d : ndarray, shape ``(n_time-1, dim)`` or ``(dim,)``
        Per-step (or constant) deterministic drive ``B^{-1}(I - A) F``.
    L : ndarray, shape ``(n_time-1, dim, dim)`` or ``(dim, dim)``
        Per-step (or constant) Cholesky factors of the innovation covariance.
    z : ndarray, shape ``(n_traj, n_time-1, dim)``
        Standard-normal innovations.
    x0 : ndarray, shape ``(n_traj, dim)``
        Initial state.

    Returns
    -------
    x : ndarray, shape ``(n_traj, n_time, dim)``
    """
    n_traj, n_steps, dim = z.shape
    n_time = n_steps + 1
    x = np.empty((n_traj, n_time, dim), dtype=np.result_type(z, x0, A, d, L))
    x[:, 0, :] = x0
    # allow constant coefficients (broadcast over time)
    if A.ndim == 2:
        A = np.broadcast_to(A, (n_steps, dim, dim))
    if d.ndim == 1:
        d = np.broadcast_to(d, (n_steps, dim))
    if L.ndim == 2:
        L = np.broadcast_to(L, (n_steps, dim, dim))
    for n in range(n_steps):
        # x_{n+1} = A_n x_n + d_n + L_n z_n   (per trajectory)
        xn = x[:, n, :]                      # (n_traj, dim)
        x[:, n + 1, :] = (xn @ A[n].T) + d[n] + (z[:, n, :] @ L[n].T)
    return x