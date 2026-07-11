"""Method 3 — 1/f as a sum of OU / Lorentzians (TLS ensemble).

Physical model: an ensemble of fluctuators with relaxation rates ``theta_k``
log-spaced over ``[theta_min, theta_max]``. Each OU has one-sided PSD
``S_k(f) = 2 sigma_k^2 / (theta_k^2 + (2*pi*f)^2)`` (Lorentzian, corner
``theta_k / 2*pi``). Their sum approximates ``1/f^alpha`` between
``theta_min/2*pi`` and ``theta_max/2*pi``.

* Canonical ``alpha=1``: rates log-uniform, equal weight per decade.
* General ``alpha``: weights ``w_k`` from a fast non-negative least-squares fit
  of ``sum_k w_k S_k(f)`` to the target on a log-f grid (cached at setup).
* Reuses the vectorized OU engine (§6.2) — generate all ``n_components`` OU
  processes as a batch and sum. Cheap, exactly stationary, extends naturally to
  spatial correlations.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..core import NoiseResult, make_generator, make_time_grid
from ..backends import ou_scalar_recursion

__all__ = ["generate_ou_sum", "fit_ou_weights", "lorentzian_psd"]


def lorentzian_psd(f: np.ndarray, theta: float, sigma: float) -> np.ndarray:
    """One-sided Lorentzian PSD ``S(f) = 2 sigma^2 / (theta^2 + (2*pi*f)^2)``.

    This is the one-sided PSD of an OU process with rate ``theta`` and diffusion
    ``sigma`` (stationary variance ``sigma^2 / (2*theta)``). It integrates to
    ``sigma^2 / (2*theta)`` = the variance, consistent with the convention
    ``var = integral_0^inf S(f) df``.
    """
    f = np.asarray(f, dtype=float)
    return 2.0 * sigma**2 / (theta**2 + (2.0 * np.pi * f) ** 2)


def fit_ou_weights(thetas: np.ndarray, sigmas: np.ndarray, target_psd: np.ndarray,
                   f_grid: np.ndarray) -> np.ndarray:
    """Non-negative least-squares fit of ``sum_k w_k S_k(f)`` to ``target_psd``.

    Solves ``min_w || A w - y ||_2`` subject to ``w >= 0``, where ``A[:, k] =
    S_k(f_grid)``. Uses a simple projected-gradient / active-set NNLS (NumPy
    only, no SciPy). Returns the weight vector ``w``.
    """
    A = np.stack([lorentzian_psd(f_grid, t, s) for t, s in zip(thetas, sigmas)], axis=1)
    y = np.asarray(target_psd, dtype=float)
    return _nnls(A, y)


def _nnls(A: np.ndarray, b: np.ndarray, max_iter: int = 200, tol: float = 1e-9) -> np.ndarray:
    """Minimal non-negative least-squares (NumPy-only, Lawson-Hanson-ish active set).

    Adequate for the small component counts (tens) we use. For very large
    problems, SciPy's optimized NNLS would be preferable.
    """
    m, n = A.shape
    w = np.zeros(n, dtype=float)
    P = np.zeros(n, dtype=bool)  # passive (free) set
    R = np.ones(n, dtype=bool)   # active (zero) set
    AtA = A.T @ A
    Atb = A.T @ b

    for _ in range(max_iter):
        if not np.any(R):
            break
        # gradient for the zero variables: pick the most negative
        w_full = w.copy()
        grad = AtA @ w_full - Atb
        idx = np.where(R)[0]
        j = idx[np.argmin(grad[idx])]
        if grad[j] >= -tol:
            break  # KKT satisfied
        P[j] = True
        R[j] = False
        # solve least squares on the passive set
        for _inner in range(max_iter):
            P_idx = np.where(P)[0]
            if P_idx.size == 0:
                break
            Ap = AtA[np.ix_(P_idx, P_idx)]
            bp = Atb[P_idx]
            try:
                wp = np.linalg.solve(Ap, bp)
            except np.linalg.LinAlgError:
                wp = np.linalg.lstsq(Ap, bp, rcond=None)[0]
            # check feasibility
            if np.all(wp > 0):
                w[P_idx] = wp
                break
            # find the step that keeps w >= 0
            alpha = 1.0
            for k in P_idx:
                if wp[np.searchsorted(P_idx, k)] < 0:
                    ratio = w[k] / (w[k] - wp[np.searchsorted(P_idx, k)] + 1e-30)
                    alpha = min(alpha, ratio)
            w[P_idx] = w[P_idx] + alpha * (wp - w[P_idx])
            # move any zeroed-out variable back to the active set
            zeroed = P_idx[np.where(w[P_idx] <= tol)[0]]
            P[zeroed] = False
            R[zeroed] = True
            w[zeroed] = 0.0
    return w


def generate_ou_sum(n_traj: int = 1, fs: float = 1.0, n_points: int = 2**16,
                    alpha: float = 1.0, S0: float = 1.0, f0: float = 1.0,
                    n_components_per_decade: int = 8,
                    theta_min: Optional[float] = None, theta_max: Optional[float] = None,
                    dtype=np.float64, seed=None, t: Optional[np.ndarray] = None,
                    backend: str = "numpy", weights: Optional[np.ndarray] = None
                    ) -> NoiseResult:
    """Generate 1/f^alpha noise as a sum of OU/Lorentzian components.

    Parameters
    ----------
    alpha : float
        Target spectral exponent. ``1`` -> canonical 1/f (log-uniform rates,
    n_components_per_decade : int
        OU components per decade of rate. Default 8.
    theta_min, theta_max : float, optional
        Rate bracket ``[theta_min, theta_max]`` in ``1/s``. Defaults to a few
        decades bracketing the requested band: ``2*pi*f_min`` to ``2*pi*f_max``
        where ``f_min = fs/n_points`` and ``f_max = fs/2``.
    weights : ndarray, optional
        Pre-fit component weights. If None, computed from the target PSD.
    """
    t_arr, fs_eff, n = make_time_grid(fs=fs, n_points=n_points, t=t)
    rng = make_generator(seed)
    dt = np.diff(t_arr)

    # default rate bracket: the 1/f plateau spans [theta_min/2pi, theta_max/2pi].
    # Add ~1 decade margin below the lowest requested freq (slower fluctuators)
    # so the low-f end is flat. Do NOT extend theta_max beyond Nyquist: components
    # with theta*dt >> 1 have a white discrete-time PSD (not Lorentzian), which
    # flattens the measured slope. Keep theta_max at ~Nyquist.
    if theta_min is None:
        theta_min = 2.0 * np.pi * fs_eff / n / 10.0
    if theta_max is None:
        theta_max = 2.0 * np.pi * fs_eff / 2.0
    n_decades = np.log10(theta_max / theta_min)
    n_comp = max(1, int(np.ceil(n_components_per_decade * n_decades)))
    thetas = np.logspace(np.log10(theta_min), np.log10(theta_max), n_comp)

    # per-component sigma chosen so each Lorentzian has unit variance contribution
    # (sigma_k^2 / (2 theta_k) = 1) -> sigma_k = sqrt(2 theta_k); weights rescale.
    sigmas = np.sqrt(2.0 * thetas)

    # compute weights to match the target PSD.
    # The sum-of-Lorentzians gives 1/f^alpha when weights scale as
    # w_k ~ theta_k^(1-alpha) with log-uniform rates: steeper spectra (alpha>1)
    # need more weight on slow (low-theta) components whose 1/f^2 tails dominate
    # at higher frequencies; shallower spectra (alpha<1) need more weight on fast
    # components. alpha=1 recovers equal weights. This is the analytic
    # continuation of the canonical 1/f result and avoids a fragile NNLS fit.
    # (The NNLS path is kept available via fit_ou_weights for users who want to
    # fit a tabulated target explicitly.)
    if weights is None:
        weights = thetas ** (1.0 - alpha)
        # normalize weights so the total PSD matches S0 at f0
        f_norm = np.array([f0])
        S_sum = np.sum([w * lorentzian_psd(f_norm, t, s)[0]
                        for w, t, s in zip(weights, thetas, sigmas)])
        if S_sum > 0:
            S_target = S0 * (f0 / f_norm) ** alpha
            weights = weights * (S_target[0] / S_sum)

    # effective per-component sigma after weighting
    sigma_eff = sigmas * np.sqrt(weights)  # variance scales as sigma^2

    # vectorized OU over (n_traj, n_comp): build coefficients per component.
    # Time is the LAST axis so they broadcast against z=(n_traj, n_comp, n_time-1).
    a = np.exp(-thetas[None, :] * dt[:, None]).T              # (n_comp, n-1)
    s = (sigma_eff[None, :] * np.sqrt((1.0 - np.exp(-2.0 * thetas[None, :] * dt[:, None])) / (2.0 * thetas[None, :]))).T
    b = np.zeros_like(a)  # zero-mean OU per component

    # innovations: (n_traj, n_comp, n-1)
    z = rng.standard_normal(size=(n_traj, n_comp, n - 1))
    # initial state from stationary distribution per component
    var_ss = sigma_eff**2 / (2.0 * thetas)  # (n_comp,)
    x0 = np.sqrt(var_ss)[None, :] * rng.standard_normal(size=(n_traj, n_comp))

    # run the scalar recursion with a component axis
    x_comp = ou_scalar_recursion(a, b, s, z, x0, backend=backend)  # (n_traj, n_comp, n)
    x = x_comp.sum(axis=1).astype(dtype)  # sum over components -> (n_traj, n)

    spec = {"process": "ou_sum", "method": "sum_of_lorentzians", "alpha": alpha,
            "S0": S0, "f0": f0, "n_components": n_comp, "thetas": thetas.tolist(),
            "weights": weights.tolist(), "theta_min": theta_min, "theta_max": theta_max,
            "fs": fs_eff, "n_points": n, "n_traj": n_traj, "backend": backend, "dtype": str(dtype)}
    return NoiseResult(traj=x, t=t_arr, fs=fs_eff, spec=spec, seed=seed, units="a.u.")