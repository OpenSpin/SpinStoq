# API Reference

## Main Functions

### `generate(name_or_trace, *, n_traj, fs, n_points, t, seed, backend, dtype, **kwargs)`

Generate correlated noise trajectories by name or from a measured trace.

**Parameters:**
- `name_or_trace` (str or ndarray): Process name (e.g., "1/f", "ou", "ou_sum", "rtn") or measured trace
- `n_traj` (int, default=1): Number of trajectories
- `fs` (float): Sample rate [Hz] (uniform grid)
- `n_points` (int, optional): Number of time points (uniform grid)
- `t` (ndarray, optional): Explicit time grid [s] (overrides fs/n_points)
- `seed`: Reproducibility seed (int, SeedSequence, etc.)
- `backend` (str, default="numpy"): {"numpy", "numba", "jax"} — OU recursion only
- `dtype` (type, default=np.float64): Output dtype (float32 for memory savings)
- `**kwargs`: Process-specific parameters (override preset defaults)

**Returns:**
`NoiseResult` — dataclass with trajectories, timestamps, and metadata

**Example:**
```python
res = generate("1/f", n_traj=100, fs=1e4, n_points=2**14,
               alpha=1.0, S0=1e-12, f0=1.0, seed=42)
```

### `calibrate(trace, fs, method="circulant")`

Estimate second-order statistics from a measured trace and return a surrogate generator.

**Parameters:**
- `trace` (ndarray): Measured data (1D or 2D with channels)
- `fs` (float): Sample rate [Hz]
- `method` (str, default="circulant"): Estimation method

**Returns:**
`SurrogateGenerator` — call `.sample(n_traj, n_points, seed)` to generate

**Example:**
```python
gen = calibrate(measured_trace, fs=1e4)
res = gen.sample(n_traj=1000, n_points=2**16, seed=0)
```

## Result Object

### `NoiseResult`

Thin dataclass wrapping arrays + metadata.

**Attributes:**
- `traj` (ndarray): Shape `(n_traj, [n_sites,] n_time)`
- `t` (ndarray): Time grid [s], shape `(n_time,)`
- `fs` (float or None): Sample rate [Hz], or None for non-uniform grids
- `spec` (dict): Full process specification (parameters, method, etc.)
- `seed`: Seed used for reproducibility
- `units` (str): Units (e.g., "V", "a.u.")
- `n_traj`, `n_time`, `n_sites`: Dimensions
- `is_uniform`: Whether time grid is uniform

**Methods:**
- `save(path)`: Save to `.npz` file (includes all metadata)
- `NoiseResult.load(path)`: Load from `.npz`
- `to_xarray()`: Convert to xarray DataArray (requires xarray)

## Analysis Functions

Fast spectral and statistical analysis (all vectorized over ensemble):

```python
from src.noise import welch, allan_deviation, autocorrelation

# Power spectral density
f, S = welch(res.traj, fs=res.fs)

# Overlapping Allan deviation
taus, sigma = allan_deviation(res.traj, fs=res.fs)

# Autocorrelation (via Wiener-Khinchin)
tau, acf = autocorrelation(res.traj, fs=res.fs)
```

See main README for full list.

## Named Processes

Access via `generate(name, ...)`:

| Name | Type | Description |
|------|------|-------------|
| `1/f` | Spectral (FFT) | Stationary Gaussian 1/f^α |
| `ou` | OU process | Scalar Ornstein-Uhlenbeck |
| `ou_sum` | Sum of Lorentzians | 1/f^α via OU superposition |
| `rtn` | Random telegraph | Two-level fluctuators |
| `spatial_fluctuators` | Spatial | Correlated noise via kernel |
| `charge_noise_SiGe` | Preset | Literature SiGe 1/f noise |
| `charge_noise_SiMOS` | Preset | Literature SiMOS 1/f noise |
| `rtn_dominated` | Preset | RTN-dominated noise |

List all: `from src.noise import list_processes; print(list_processes())`

## Backends

Acceleration for OU-family generators:

```python
# NumPy (default, pure Python)
res = generate("ou_sum", ..., backend="numpy")

# Numba JIT (requires numba)
res = generate("ou_sum", ..., backend="numba")

# JAX (GPU/TPU, requires jax)
res = generate("ou_sum", ..., backend="jax")
```
