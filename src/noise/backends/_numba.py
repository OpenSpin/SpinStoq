"""Numba backend for the OU time recursion (optional, lazily imported).

Accelerates the only sequential kernel (the OU time recursion) with a JIT-compiled
loop. Public API and results are identical to the numpy backend up to floating
point for a fixed seed policy.
"""
from __future__ import annotations

import numpy as np

__all__ = ["ou_scalar_recursion_numba", "ou_multivariate_recursion_numba", "available"]

try:
    from numba import njit, prange
    _NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover
    _NUMBA_AVAILABLE = False


def available() -> bool:
    return _NUMBA_AVAILABLE


if _NUMBA_AVAILABLE:

    @njit(cache=True, fastmath=True)
    def _scalar_recursion(a, b, s, z, x0, x):
        n_time = z.shape[-1] + 1
        # z shape: (n_traj, [n_comp,] n_time-1); x same leading + n_time.
        # Coefficients a/b/s are broadcast to z's shape (time is the LAST axis).
        if z.ndim == 2:
            n_traj = z.shape[0]
            x[:, 0] = x0
            for n in range(n_time - 1):
                for i in range(n_traj):
                    x[i, n + 1] = a[i, n] * x[i, n] + b[i, n] + s[i, n] * z[i, n]
        elif z.ndim == 3:
            n_traj = z.shape[0]
            n_comp = z.shape[1]
            x[:, :, 0] = x0
            for n in range(n_time - 1):
                for i in range(n_traj):
                    for c in range(n_comp):
                        x[i, c, n + 1] = a[i, c, n] * x[i, c, n] + b[i, c, n] + s[i, c, n] * z[i, c, n]
        return x

    @njit(cache=True, fastmath=True)
    def _multivariate_recursion(A, d, L, z, x0, x):
        n_traj, n_steps, dim = z.shape
        n_time = n_steps + 1
        x[:, 0, :] = x0
        for n in range(n_steps):
            for i in range(n_traj):
                for r in range(dim):
                    acc = 0.0
                    for c in range(dim):
                        acc += A[n, r, c] * x[i, n, c]
                    acc += d[n, r]
                    for c in range(dim):
                        acc += L[n, r, c] * z[i, n, c]
                    x[i, n + 1, r] = acc
        return x

    def ou_scalar_recursion_numba(a, b, s, z, x0):
        n_time = z.shape[-1] + 1
        x = np.empty(z.shape[:-1] + (n_time,), dtype=np.result_type(z, x0, a, b, s))
        a = np.ascontiguousarray(np.broadcast_to(a, z.shape), dtype=x.dtype)
        b = np.ascontiguousarray(np.broadcast_to(b, z.shape), dtype=x.dtype)
        s = np.ascontiguousarray(np.broadcast_to(s, z.shape), dtype=x.dtype)
        zc = np.ascontiguousarray(z, dtype=x.dtype)
        x0c = np.ascontiguousarray(x0, dtype=x.dtype)
        return _scalar_recursion(a, b, s, zc, x0c, x)

    def ou_multivariate_recursion_numba(A, d, L, z, x0):
        n_traj, n_steps, dim = z.shape
        n_time = n_steps + 1
        x = np.empty((n_traj, n_time, dim), dtype=np.result_type(z, x0, A, d, L))
        if A.ndim == 2:
            A = np.broadcast_to(A, (n_steps, dim, dim))
        if d.ndim == 1:
            d = np.broadcast_to(d, (n_steps, dim))
        if L.ndim == 2:
            L = np.broadcast_to(L, (n_steps, dim, dim))
        return _multivariate_recursion(
            np.ascontiguousarray(A, dtype=x.dtype),
            np.ascontiguousarray(d, dtype=x.dtype),
            np.ascontiguousarray(L, dtype=x.dtype),
            np.ascontiguousarray(z, dtype=x.dtype),
            np.ascontiguousarray(x0, dtype=x.dtype),
            x,
        )

else:  # pragma: no cover

    def ou_scalar_recursion_numba(*args, **kwargs):
        raise ImportError("numba backend requires the 'numba' extra: pip install openspin-noise[numba]")

    def ou_multivariate_recursion_numba(*args, **kwargs):
        raise ImportError("numba backend requires the 'numba' extra: pip install openspin-noise[numba]")