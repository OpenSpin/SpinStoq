"""Core containers, seeding, time grids, and unit helpers for ``openspin.noise``.

Conventions (enforced across the package, see spec §5):

* PSD is **one-sided**, ``S(f)`` for ``f >= 0``, in ``units^2 / Hz`` and satisfies
  ``var(x) = integral_0^{fs/2} S(f) df``.
* Wiener-Khinchin: ``R(tau) = integral_0^inf S(f) cos(2*pi*f*tau) df``.
* Cyclic frequency ``f`` in **Hz** everywhere (not angular omega).
* OU rate ``theta`` in ``1/s``; its Lorentzian corner is at ``f = theta / (2*pi)``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "NoiseResult",
    "make_generator",
    "spawn_seeds",
    "make_time_grid",
    "units_helpers",
    "convert_units",
    "DEFAULT_FLOAT",
]

DEFAULT_FLOAT = np.float64


# --------------------------------------------------------------------------- #
# Seeding
# --------------------------------------------------------------------------- #
def make_generator(seed: Optional[Union[int, Sequence[int], np.random.SeedSequence, np.random.Generator]] = None
                   ) -> np.random.Generator:
    """Return a ``numpy.random.Generator`` from any reasonable seed-like input.

    Accepts ``None`` (non-reproducible entropy), an int, a sequence of ints, an
    existing ``SeedSequence`` or an existing ``Generator`` (returned as-is).
    """
    if isinstance(seed, np.random.Generator):
        return seed
    if isinstance(seed, np.random.SeedSequence):
        return np.random.default_rng(seed)
    return np.random.default_rng(seed)


def spawn_seeds(seed: Optional[Union[int, Sequence[int], np.random.SeedSequence]] = None,
                n: int = 1) -> list[np.random.SeedSequence]:
    """Spawn ``n`` independent ``SeedSequence`` streams from a root seed.

    Identical results regardless of how work is later chunked across these
    streams, as long as each child stream is consumed in the same order.
    """
    ss = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    return ss.spawn(n)


# --------------------------------------------------------------------------- #
# Time grids
# --------------------------------------------------------------------------- #
def make_time_grid(fs: Optional[float] = None,
                   n_points: Optional[int] = None,
                   t: Optional[Sequence[float]] = None,
                   duration: Optional[float] = None
                   ) -> Tuple[np.ndarray, float, int]:
    """Resolve a time grid from either ``(fs, n_points)`` or an explicit ``t``.

    Returns ``(t, fs_eff, n_points)``. When ``t`` is given, ``fs_eff`` is the
    mean sample rate (used only for reporting / PSD frequency axis on uniform
    grids; OU-family generators honor the actual per-step ``dt``).
    """
    if t is not None:
        t_arr = np.asarray(t, dtype=float)
        if t_arr.ndim != 1 or t_arr.size < 2:
            raise ValueError("explicit `t` must be a 1D array with >= 2 points")
        dt = np.diff(t_arr)
        fs_eff = float(1.0 / np.mean(dt)) if dt.size else float("nan")
        return t_arr, fs_eff, t_arr.size
    if fs is None or n_points is None:
        raise ValueError("provide either `fs` and `n_points`, or an explicit `t` array")
    if fs <= 0:
        raise ValueError("fs must be positive")
    if n_points < 2:
        raise ValueError("n_points must be >= 2")
    if duration is not None:
        n_points = int(round(duration * fs)) + 1
    dt = 1.0 / float(fs)
    t_arr = np.arange(n_points, dtype=float) * dt
    return t_arr, float(fs), int(n_points)


# --------------------------------------------------------------------------- #
# Units
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class UnitsHelpers:
    """Tiny unit-conversion helpers (integration point with electrostatics/CSD).

    Conversions go through a lever arm ``alpha`` [eV per V] (or per unit of the
    noise process). ``V -> eV`` multiplies by ``alpha``; ``eV -> detuning`` is a
    placeholder hook for the CSD module to fill in.
    """

    lever_arm: float = 1.0  # eV per V (or per process unit)

    def to_ev(self, x: np.ndarray, from_unit: str = "V") -> np.ndarray:
        if from_unit == "eV":
            return np.asarray(x, dtype=float)
        if from_unit == "V":
            return np.asarray(x, dtype=float) * self.lever_arm
        raise ValueError(f"unknown source unit {from_unit!r}")

    def from_ev(self, x: np.ndarray, to_unit: str = "V") -> np.ndarray:
        if to_unit == "eV":
            return np.asarray(x, dtype=float)
        if to_unit == "V":
            return np.asarray(x, dtype=float) / self.lever_arm
        raise ValueError(f"unknown target unit {to_unit!r}")


def units_helpers(lever_arm: float = 1.0) -> UnitsHelpers:
    return UnitsHelpers(lever_arm=lever_arm)


def convert_units(x: np.ndarray, lever_arm: float = 1.0, frm: str = "V", to: str = "eV") -> np.ndarray:
    """Free-function convenience wrapper around :class:`UnitsHelpers`."""
    h = UnitsHelpers(lever_arm=lever_arm)
    ev = h.to_ev(x, frm)
    return h.from_ev(ev, to)


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class NoiseResult:
    """Thin dataclass wrapping a noise ensemble + full provenance metadata.

    Attributes
    ----------
    traj : ndarray, shape ``(n_traj, [n_sites,] n_time)``
        The generated trajectories. ``n_sites`` axis present only for spatial
        generators.
    t : ndarray, shape ``(n_time,)``
        Time grid [s].
    fs : float
        Sample rate [Hz]. ``nan`` for non-uniform grids.
    spec : dict
        Full process specification that produced this result (process name,
        parameters, backend, ...). Round-trippable for reproducibility.
    seed : int or Sequence[int] or None
        Root seed used. Combined with ``spec`` this fully reproduces the result.
    units : str
        Physical units of ``traj`` (e.g. ``"V"``, ``"eV"``, ``"a.u."``).
    """

    traj: np.ndarray
    t: np.ndarray
    fs: float
    spec: dict = field(default_factory=dict)
    seed: Optional[Union[int, Sequence[int]]] = None
    units: str = "a.u."

    # ---- shape conveniences ----
    @property
    def n_traj(self) -> int:
        return int(self.traj.shape[0])

    @property
    def n_time(self) -> int:
        return int(self.traj.shape[-1])

    @property
    def n_sites(self) -> int:
        return int(self.traj.shape[1]) if self.traj.ndim == 3 else 0

    @property
    def dt(self) -> np.ndarray:
        """Per-step time increments (length ``n_time-1``). Uniform grid -> scalar-ish."""
        return np.diff(self.t)

    @property
    def is_uniform(self) -> bool:
        dt = self.dt
        if dt.size == 0:
            return True
        return bool(np.allclose(dt, dt[0], rtol=1e-9, atol=1e-12))

    # ---- serialization ----
    def to_xarray(self):
        """Return an :class:`xarray.DataArray` (requires the ``xarray`` extra)."""
        try:
            import xarray as xr
        except ImportError as e:  # pragma: no cover
            raise ImportError("to_xarray() requires the 'xarray' extra: pip install openspin-noise[xarray]") from e
        coords: dict[str, np.ndarray] = {"time": self.t, "traj": np.arange(self.n_traj)}
        dims = ["traj", "time"]
        if self.traj.ndim == 3:
            coords["site"] = np.arange(self.n_sites)
            dims = ["traj", "site", "time"]
        return xr.DataArray(self.traj, dims=dims, coords=coords, attrs={**self.spec, "units": self.units, "seed": self.seed})

    def save(self, path: str) -> None:
        """Save to a ``.npz`` archive (round-trippable via :meth:`load`)."""
        np.savez(path,
                 traj=self.traj, t=self.t, fs=np.array(self.fs, dtype=float),
                 spec=np.array(self.spec, dtype=object),
                 seed=np.array(self.seed, dtype=object),
                 units=np.array(self.units))

    @classmethod
    def load(cls, path: str) -> "NoiseResult":
        with np.load(path, allow_pickle=True) as z:
            spec = z["spec"].item() if z["spec"].dtype == object else dict(z["spec"])
            seed = z["seed"].item() if z["seed"].dtype == object else (None if z["seed"].size == 0 else z["seed"].tolist())
            units = str(z["units"].item()) if z["units"].dtype == object else str(z["units"])
            return cls(traj=z["traj"], t=z["t"], fs=float(z["fs"]), spec=spec, seed=seed, units=units)

    def __repr__(self) -> str:
        kind = "spatial" if self.n_sites else "scalar"
        return (f"NoiseResult(traj={self.traj.shape}, {kind}, n_time={self.n_time}, "
                f"fs={self.fs:.4g} Hz, units={self.units!r}, process={self.spec.get('process', '?')!r})")