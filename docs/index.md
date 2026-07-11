# SpinStoq Documentation

Fast correlated (temporal & spatial) charge/qubit noise generator for spin-qubit simulation.

## Quick Links

- **[Installation](installation.md)** — Get started
- **[API Reference](api.md)** — Full API documentation
- **[Examples](examples.md)** — Usage examples and notebooks
- **[Design](design.md)** — Design principles and conventions

## What is SpinStoq?

SpinStoq generates realistic noise trajectories for spin-qubit simulations:

- **Fast**: NumPy-first, O(N log N) via FFT where possible
- **Flexible**: 1/f, OU processes, RTN, spatial noise, surrogates from measured traces
- **Reproducible**: Explicit seeding via `SeedSequence`, full spec recorded
- **Validated**: Round-trip tests ensure PSD/ACF match target

## Core API

```python
from src.noise import generate, calibrate

# By name
res = generate("1/f", n_traj=1000, fs=1e4, n_points=2**16, 
               alpha=1.0, S0=1e-12, f0=1.0, seed=0)
res.traj  # (1000, 65536) trajectories
res.t     # (65536,) time grid in seconds

# From measured trace
gen = calibrate(measured_trace, fs=1e4)
res = gen.sample(n_traj=1000, n_points=2**16, seed=0)
```

Each result includes:
- `res.traj`: trajectory data `(n_traj, [n_sites,] n_time)`
- `res.t`: time grid in seconds
- `res.fs`: sample rate [Hz]
- `res.seed`: seed used
- `res.spec`: full process specification (reproducible)
- `res.units`: units (e.g., "V", "a.u.")
