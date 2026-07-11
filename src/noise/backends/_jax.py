"""JAX backend for the OU time recursion (optional, lazily imported).

Uses ``lax.scan`` for the time recursion so it runs on CPU/GPU/TPU and is
JIT-friendly. Public API and results are identical to the numpy backend up to
floating point for a fixed seed policy.
"""
from __future__ import annotations

import numpy as np

__all__ = ["ou_scalar_recursion_jax", "ou_multivariate_recursion_jax", "available"]

try:
    import jax
    import jax.numpy as jnp
    from jax import lax
    _JAX_AVAILABLE = True
except Exception:  # pragma: no cover - jax import can fail on unsupported CPUs
    _JAX_AVAILABLE = False


def available() -> bool:
    return _JAX_AVAILABLE


if _JAX_AVAILABLE:

    def ou_scalar_recursion_jax(a, b, s, z, x0):
        a = jnp.asarray(a); b = jnp.asarray(b); s = jnp.asarray(s)
        z = jnp.asarray(z); x0 = jnp.asarray(x0)

        def step(x_n, inputs):
            a_n, b_n, s_n, z_n = inputs
            x_np1 = a_n * x_n + b_n + s_n * z_n
            return x_np1, x_np1

        # transpose so time is leading for scan: (n_time-1, ...)
        inputs = (a, b, s, z)
        # broadcast coefficients to z's shape, then move time axis to front
        a_b = jnp.broadcast_to(a, z.shape)
        b_b = jnp.broadcast_to(b, z.shape)
        s_b = jnp.broadcast_to(s, z.shape)
        if z.ndim == 2:
            seq = (a_b.T, b_b.T, s_b.T, z.T)  # each (n_time-1, n_traj)
        else:
            seq = (jnp.moveaxis(a_b, -1, 0), jnp.moveaxis(b_b, -1, 0),
                   jnp.moveaxis(s_b, -1, 0), jnp.moveaxis(z, -1, 0))
        _, xs = lax.scan(step, x0, seq)
        # xs has time leading; move time back to last axis
        x = jnp.moveaxis(xs, 0, -1)
        x0_b = jnp.expand_dims(x0, -1)
        return np.array(jnp.concatenate([x0_b, x], axis=-1))

    def ou_multivariate_recursion_jax(A, d, L, z, x0):
        A = jnp.asarray(A); d = jnp.asarray(d); L = jnp.asarray(L)
        z = jnp.asarray(z); x0 = jnp.asarray(x0)
        n_traj, n_steps, dim = z.shape
        if A.ndim == 2:
            A = jnp.broadcast_to(A, (n_steps, dim, dim))
        if d.ndim == 1:
            d = jnp.broadcast_to(d, (n_steps, dim))
        if L.ndim == 2:
            L = jnp.broadcast_to(L, (n_steps, dim, dim))

        def step(x_n, inputs):
            A_n, d_n, L_n, z_n = inputs
            x_np1 = x_n @ A_n.T + d_n + z_n @ L_n.T
            return x_np1, x_np1

        seq = (A, d, L, jnp.moveaxis(z, 1, 0))  # time-leading
        _, xs = lax.scan(step, x0, seq)
        x = jnp.moveaxis(xs, 0, 1)  # (n_traj, n_time-1, dim)
        x0_b = jnp.expand_dims(x0, 1)
        return np.array(jnp.concatenate([x0_b, x], axis=1))

else:  # pragma: no cover

    def ou_scalar_recursion_jax(*args, **kwargs):
        raise ImportError("jax backend requires the 'jax' extra: pip install openspin-noise[jax]")

    def ou_multivariate_recursion_jax(*args, **kwargs):
        raise ImportError("jax backend requires the 'jax' extra: pip install openspin-noise[jax]")