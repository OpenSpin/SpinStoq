"""Method 2 — exact Ornstein-Uhlenbeck (scalar + multivariate/coupled).

Overdamped Langevin with an explicit force term ``F``: ``dx = (F - gamma x) dt + sigma dW``.

**Exact exponential propagator** (valid for *any* dt, spec §6.2):

    x[n+1] = x[n] * exp(-gamma*dt[n]) + (F/gamma)*(1 - exp(-gamma*dt[n]))
             + sigma * sqrt((1 - exp(-2*gamma*dt[n]))/(2*gamma)) * Z[n]

This is the exact solution — **not** Euler-Maruyama. The deterministic part is
propagated as ``(F/gamma)(1 - e^{-gamma dt})`` (correct at any step size), and
the noise term has the exact stationary variance ``sigma^2/(2*gamma)`` in the
``dt -> inf`` limit. ``F`` defaults to 0 (zero-mean OU); set ``F != 0`` for a
nonzero steady-state mean ``F/gamma``. ``F`` may be per-step piecewise-constant
(deterministic drive/tilt). ``dt[n]`` is per-step -> arbitrary/non-uniform grids
supported directly. Initialize ``x[0]`` from the stationary distribution
``N(F/gamma, sigma^2/(2*gamma))`` (no burn-in).

The multivariate/coupled form uses the matrix-exponential scheme:

    x[n+1] = A[n] x[n] + B^{-1}(I - A[n]) F + eta[n],   A[n] = expm(-B dt[n])
    Cov(eta[n]) = Sigma_ss - A[n] Sigma_ss A[n]^T,     B Sigma_ss + Sigma_ss B^T = L L^T (Lyapunov)

which reduces to the scalar form when ``B = gamma``. This gives arbitrary-dt
transitions and cross-correlated components; it doubles as a spatial-correlation
engine (§6.4).
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

from ..core import NoiseResult, make_generator, make_time_grid
from ..backends import ou_scalar_recursion, ou_multivariate_recursion

__all__ = ["generate_ou", "generate_ou_multivariate"]


def generate_ou(n_traj: int = 1, fs: float = 1.0, n_points: int = 2**16,
                gamma: float = 1.0, sigma: float = 1.0, F: Union[float, np.ndarray] = 0.0,
                x0: Optional[np.ndarray] = None, dtype=np.float64, seed=None,
                t: Optional[np.ndarray] = None, backend: str = "numpy") -> NoiseResult:
    """Generate scalar exact-OU trajectories.

    Parameters
    ----------
    gamma : float
        Drift rate ``gamma`` (a.k.a. ``theta``) in ``1/s``. Stationary variance
        ``sigma^2 / (2*gamma)``; ACF ``exp(-gamma|tau|)``; Lorentzian corner
        ``f = gamma / (2*pi)``.
    sigma : float
        Diffusion amplitude. ``sigma dW`` drives the process.
    F : float or ndarray, shape ``(n_time,)`` or ``(n_time-1,)``
        Force term. Default 0 -> zero-mean OU. Constant -> steady-state mean
        ``F/gamma``. Per-step piecewise-constant array -> exact deterministic
        drive/tilt (e.g. a control ramp) while keeping the update exact.
    x0 : ndarray, optional
        Initial state. If None, drawn from the stationary distribution
        ``N(F/gamma, sigma^2/(2*gamma))`` (no burn-in).
    backend : {"numpy", "numba", "jax"}
        Implementation of the time recursion.
    """
    t_arr, fs_eff, n = make_time_grid(fs=fs, n_points=n_points, t=t)
    rng = make_generator(seed)
    dt = np.diff(t_arr)  # per-step, length n-1

    # per-step propagator coefficients (exact exponential update)
    a = np.exp(-gamma * dt)                                   # x[n] multiplier
    b = (F / gamma) * (1.0 - a) if not np.ndim(F) else None    # deterministic drive
    s = sigma * np.sqrt((1.0 - np.exp(-2.0 * gamma * dt)) / (2.0 * gamma))  # noise amplitude

    # handle per-step F (piecewise-constant drive): F[n] applies over step n->n+1
    if np.ndim(F) > 0:
        F_arr = np.asarray(F, dtype=float)
        if F_arr.size == n - 1:
            F_step = F_arr
        elif F_arr.size == n:
            F_step = F_arr[:-1]  # last point unused
        else:
            raise ValueError(f"F array length must be {n-1} or {n}, got {F_arr.size}")
        b = (F_step / gamma) * (1.0 - a)

    # stationary mean / variance for initialization
    F0 = float(F) if np.ndim(F) == 0 else float(np.asarray(F).flat[0])
    mean_ss = F0 / gamma
    var_ss = sigma**2 / (2.0 * gamma)
    std_ss = np.sqrt(var_ss) if var_ss > 0 else 0.0

    # innovations and initial state, vectorized over n_traj
    z = rng.standard_normal(size=(n_traj, n - 1))
    if x0 is None:
        x0_arr = mean_ss + std_ss * rng.standard_normal(size=(n_traj,))
    else:
        x0_arr = np.broadcast_to(np.asarray(x0, dtype=float), (n_traj,)).astype(float)

    x = ou_scalar_recursion(a, b, s, z, x0_arr, backend=backend).astype(dtype)

    spec = {"process": "ou", "method": "exact_exponential", "gamma": gamma,
            "sigma": sigma, "F": F0, "fs": fs_eff, "n_points": n, "n_traj": n_traj,
            "backend": backend, "dtype": str(dtype), "stationary_var": var_ss,
            "stationary_mean": mean_ss}
    return NoiseResult(traj=x, t=t_arr, fs=fs_eff, spec=spec, seed=seed, units="a.u.")


def generate_ou_multivariate(n_traj: int = 1, fs: float = 1.0, n_points: int = 2**16,
                             B: np.ndarray = None, L: np.ndarray = None,
                             F: Optional[np.ndarray] = None, x0: Optional[np.ndarray] = None,
                             dtype=np.float64, seed=None, t: Optional[np.ndarray] = None,
                             backend: str = "numpy") -> NoiseResult:
    """Generate multivariate/coupled exact-OU trajectories.

    SDE: ``dx = (F - B x) dt + L dW``, with force vector ``F`` (default 0).

    Exact scheme (spec §6.2):

        A[n] = expm(-B dt[n])
        d[n] = B^{-1} (I - A[n]) F
        Cov(eta[n]) = Sigma_ss - A[n] Sigma_ss A[n]^T
        x[n+1] = A[n] x[n] + d[n] + chol(Cov(eta[n])) Z[n]

    where ``Sigma_ss`` solves the Lyapunov equation ``B Sigma_ss + Sigma_ss B^T = L L^T``.
    Steady-state mean ``B^{-1} F``. On a uniform grid, ``A``, ``d``, and
    ``chol(Cov(eta))`` are constant -> precomputed once, then per-step cost is
    one small matvec + one matmul with the cached Cholesky factor.

    The returned ``traj`` has shape ``(n_traj, n_time, dim)``; the leading axis
    is the trajectory batch. (For spatial use, callers can reinterpret ``dim`` as
    sites — see :mod:`openspin.noise.generators.spatial`.)
    """
    if B is None:
        raise ValueError("B (drift matrix) is required for the multivariate OU")
    B = np.asarray(B, dtype=float)
    dim = B.shape[0]
    if B.shape != (dim, dim):
        raise ValueError("B must be square (dim, dim)")
    if L is None:
        L = np.eye(dim)
    L = np.asarray(L, dtype=float)
    if F is None:
        F = np.zeros(dim)
    F = np.asarray(F, dtype=float)
    if F.shape != (dim,) and F.shape != (n_points, dim) and F.shape != (n_points - 1, dim):
        raise ValueError(f"F must have shape (dim,), (n_points, dim), or (n_points-1, dim); got {F.shape}")

    t_arr, fs_eff, n = make_time_grid(fs=fs, n_points=n_points, t=t)
    rng = make_generator(seed)
    dt = np.diff(t_arr)  # (n-1,)

    # steady-state covariance via Lyapunov: B Sigma + Sigma B^T = L L^T
    Sigma_ss = _solve_lyapunov(B, L @ L.T)
    B_inv = np.linalg.inv(B)
    mean_ss = B_inv @ (F if F.ndim == 1 else F[0])

    # per-step coefficients
    A = np.empty((n - 1, dim, dim))
    d = np.empty((n - 1, dim))
    L_eta = np.empty((n - 1, dim, dim))
    for k in range(n - 1):
        A[k] = _expm(-B * dt[k])
        if F.ndim == 1:
            d[k] = B_inv @ (np.eye(dim) - A[k]) @ F
        else:
            Fk = F[k] if F.shape[0] == n - 1 else F[k + 1]
            d[k] = B_inv @ (np.eye(dim) - A[k]) @ Fk
        cov_eta = Sigma_ss - A[k] @ Sigma_ss @ A[k].T
        # symmetrize for numerical stability then Cholesky
        cov_eta = 0.5 * (cov_eta + cov_eta.T)
        try:
            L_eta[k] = np.linalg.cholesky(cov_eta)
        except np.linalg.LinAlgError:
            # add tiny jitter if not PD (shouldn't happen for valid Sigma_ss)
            L_eta[k] = np.linalg.cholesky(cov_eta + 1e-12 * np.eye(dim))

    # uniform grid fast path: collapse to constant coefficients
    if np.allclose(dt, dt[0]):
        A = A[0]
        d = d[0]
        L_eta = L_eta[0]

    # innovations and initial state
    z = rng.standard_normal(size=(n_traj, n - 1, dim))
    if x0 is None:
        std_ss = np.linalg.cholesky(0.5 * (Sigma_ss + Sigma_ss.T))
        x0_arr = mean_ss + rng.standard_normal(size=(n_traj, dim)) @ std_ss.T
    else:
        x0_arr = np.broadcast_to(np.asarray(x0, dtype=float), (n_traj, dim)).astype(float)

    x = ou_multivariate_recursion(A, d, L_eta, z, x0_arr, backend=backend).astype(dtype)

    spec = {"process": "ou_multivariate", "method": "exact_matrix_exp", "dim": dim,
            "fs": fs_eff, "n_points": n, "n_traj": n_traj, "backend": backend,
            "dtype": str(dtype), "stationary_mean": mean_ss.tolist()}
    return NoiseResult(traj=x, t=t_arr, fs=fs_eff, spec=spec, seed=seed, units="a.u.")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _expm(M: np.ndarray) -> np.ndarray:
    """Matrix exponential (SciPy-free for small matrices via eigendecomposition).

    For the symmetric-ish drift matrices we encounter, an eigendecomposition is
    both fast and dependency-free. Falls back to a Padé-style scaling-and-squaring
    if the eigendecomposition is non-diagonalizable.
    """
    M = np.asarray(M, dtype=float)
    if M.shape[0] <= 64:
        try:
            vals, vecs = np.linalg.eig(M)
            return (vecs * np.exp(vals)) @ np.linalg.inv(vecs)
        except np.linalg.LinAlgError:
            pass
    # scaling and squaring with a truncated Taylor series (fallback)
    norm = np.linalg.norm(M, ord=np.inf)
    s = max(0, int(np.ceil(np.log2(norm + 1e-30))) + 1)
    Ms = M / (2 ** s)
    term = np.eye(M.shape[0])
    E = term.copy()
    for k in range(1, 20):
        term = term @ Ms / k
        E = E + term
    for _ in range(s):
        E = E @ E
    return E


def _solve_lyapunov(B: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Solve ``B X + X B^T = Q`` for ``X`` (SciPy-free, via vectorization).

    Uses the Kronecker-product formulation: ``(I kron B + B kron I) vec(X) = vec(Q)``.
    Fine for the small ``dim`` (sites/components) we use; for large ``dim`` a
    Bartels-Stewart solver would be preferable, but that needs SciPy.
    """
    dim = B.shape[0]
    I = np.eye(dim)
    K = np.kron(I, B) + np.kron(B, I)
    vec_Q = Q.reshape(-1, order="F")
    vec_X = np.linalg.solve(K, vec_Q)
    return vec_X.reshape(dim, dim, order="F")