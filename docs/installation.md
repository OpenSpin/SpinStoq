# Installation

## Basic install

```bash
pip install -e .
```

This installs the core package (requires NumPy ≥ 1.24, Python ≥ 3.10).

## Optional backends

For accelerated OU recursion:

```bash
# JIT-compiled (Numba)
pip install -e ".[numba]"

# GPU/TPU (JAX)
pip install -e ".[jax]"
```

## Optional extras

```bash
# Visualization (matplotlib)
pip install -e ".[viz]"

# xarray support (for convenient labeled arrays)
pip install -e ".[xarray]"

# Development (tests + viz)
pip install -e ".[dev]"
```

## Verify installation

```python
from src.noise import generate, list_processes
print(list_processes())

# Generate a test trajectory
res = generate("1/f", n_traj=1, fs=1e4, n_points=100, seed=0)
print(res.traj.shape)  # (1, 100)
```

## Troubleshooting

**JAX installation fails**: JAX requires a working CUDA/ROCm setup. On macOS, use Numba instead.

**Numba not speeding up**: Warm-up first call is slow; subsequent calls are fast.
