s# Design Principles

## Core Philosophy

SpinStoq balances speed, accuracy, and usability:

1. **NumPy-first** — Pure vectorized NumPy + `numpy.fft`. No SciPy hard-dependency.
2. **Optional acceleration behind one API** — `backend="numpy"` (default), `"numba"`, `"jax"`.
3. **Vectorize over ensemble** — Generate 1000s of trajectories in one shot; no per-sample loops on the hot path.
4. **Exactness where free** — Exact OU propagator (not Euler-Maruyama); circulant embedding for spectral methods.
5. **Reproducibility** — Explicit `SeedSequence`; every result records seed and full process spec.
6. **Round-trip validated** — Tests assert measured PSD/ACF matches target across all methods.

## Conventions

- **PSD is one-sided**, S(f) for f ≥ 0, in units²/Hz
  - Variance = ∫₀^{fs/2} S(f) df (Wiener-Khinchin integral)
- **Cyclic frequency f in Hz** everywhere
- **OU rate θ in 1/s**; Lorentzian corner at f = θ/2π
- **Time grid t in seconds**; always included in results

## Trajectory Shape

Results always include:
- `traj`: ndarray shape `(n_traj, [n_sites,] n_time)`
- `t`: ndarray shape `(n_time,)` — time grid in seconds
- Uniform or non-uniform grids supported

Single trajectory: `res.traj.shape = (1, n_time)`, not `(n_time,)`

## Backends

| Backend | Use Case | Speed | Notes |
|---------|----------|-------|-------|
| numpy | Default, portable | ×1 | Pure NumPy, no dependencies |
| numba | CPU-bound OU | ×5-10 | JIT-compiled time recursion |
| jax | GPU/TPU | ×10-100 | Compiled to XLA, needs CUDA/ROCm |

Only affects OU-family generators (`"ou"`, `"ou_sum"`, spatial with OU). Other methods are pure NumPy.

## Why No SciPy Dependency?

- Core algorithms are O(N log N) via FFT (numpy only)
- Levinson recursion for ACF ⇒ O(N²) but not on hot path
- Reduces install friction; users opt-in for scipy if needed

## Testing Strategy

1. **Convention tests**: Variance = ∫ PSD (Wiener-Khinchin)
2. **Method validation**: OU exactness at large Δt, OU-sum slope, spatial K recovery
3. **Backend parity**: All backends produce same moments
4. **From-trace surrogates**: Measured PSD matches target
5. **Reproducibility**: Same seed → identical trajectories

## Performance Targets

Laptop, single core, numpy backend:
- 1/f: 10⁴ traj × 2¹⁶ pts in ≲ 1 s
- OU-sum: 10³ traj × 10⁵ pts in ≲ 1 s
- Analysis (Welch, ACF, Allan): 10³ × 10⁵ in ≲ 1 s
