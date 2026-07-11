"""Backend dispatcher for the OU time recursion.

A ``backend=`` switch (``"numpy"`` default, ``"numba"``, ``"jax"``) selects the
implementation of the only unavoidable sequential kernel. Public API and results
(for a fixed seed policy) are identical across backends up to floating point.
"""
from __future__ import annotations

from typing import Literal

import numpy as np

from . import _numpy
from . import _numba
from . import _jax

__all__ = ["get_backend", "list_backends", "ou_scalar_recursion", "ou_multivariate_recursion"]

Backend = Literal["numpy", "numba", "jax"]


def list_backends() -> list[str]:
    """Return the list of available backends on this machine."""
    out = ["numpy"]
    if _numba.available():
        out.append("numba")
    if _jax.available():
        out.append("jax")
    return out


def get_backend(name: str):
    """Return the backend module for ``name`` (raises with a helpful message if unavailable)."""
    if name == "numpy":
        return _numpy
    if name == "numba":
        if not _numba.available():
            raise ImportError("numba backend requires the 'numba' extra: pip install openspin-noise[numba]")
        return _numba
    if name == "jax":
        if not _jax.available():
            raise ImportError("jax backend requires the 'jax' extra: pip install openspin-noise[jax]")
        return _jax
    raise ValueError(f"unknown backend {name!r}; choose from {list_backends()}")


def ou_scalar_recursion(a, b, s, z, x0, backend: Backend = "numpy"):
    """Dispatch the scalar OU recursion to the requested backend."""
    mod = get_backend(backend)
    if backend == "numpy":
        return mod.ou_scalar_recursion_numpy(a, b, s, z, x0)
    if backend == "numba":
        return mod.ou_scalar_recursion_numba(a, b, s, z, x0)
    if backend == "jax":
        return mod.ou_scalar_recursion_jax(a, b, s, z, x0)
    raise ValueError(f"unknown backend {backend!r}")


def ou_multivariate_recursion(A, d, L, z, x0, backend: Backend = "numpy"):
    """Dispatch the multivariate OU recursion to the requested backend."""
    mod = get_backend(backend)
    if backend == "numpy":
        return mod.ou_multivariate_recursion_numpy(A, d, L, z, x0)
    if backend == "numba":
        return mod.ou_multivariate_recursion_numba(A, d, L, z, x0)
    if backend == "jax":
        return mod.ou_multivariate_recursion_jax(A, d, L, z, x0)
    raise ValueError(f"unknown backend {backend!r}")