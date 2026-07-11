# Examples

## Jupyter Notebooks

- **[Quick Start](../examples/quick_start.ipynb)** — Complete walkthrough of main API

## Key Snippets

### 1. Generate 1/f noise

```python
from src.noise import generate

res = generate(
    "1/f",
    n_traj=1000,
    fs=1e4,              # sample rate [Hz]
    n_points=2**16,      # 65536 points
    alpha=1.0,           # 1/f^alpha
    S0=1e-12, f0=1.0,    # PSD normalization
    seed=0,
)
res.traj   # (1000, 65536) array
res.t      # (65536,) time grid in seconds
```

### 2. OU-sum (physically-motivated)

```python
res = generate(
    "ou_sum",
    n_traj=1000,
    fs=1e4,
    n_points=2**16,
    alpha=1.0,
    S0=1e-12, f0=1.0,
    n_components_per_decade=8,
    seed=0,
    backend="numba",  # JIT acceleration
)
```

### 3. From measured trace

```python
import numpy as np
from src.noise import calibrate

trace = np.load("my_charge_sensor_trace.npy")
gen = calibrate(trace, fs=1e4, method="circulant")
res = gen.sample(n_traj=1000, n_points=2**16, seed=0)
```

### 4. Plot spectrum

```python
from src.noise import welch
import matplotlib.pyplot as plt

res = generate("1/f", n_traj=200, fs=1e4, n_points=2**14, seed=0)
f, S = welch(res.traj, fs=res.fs)

plt.loglog(f[1:], S[1:], label="measured")
plt.loglog(f[1:], 1.0/f[1:], "k--", label="1/f")
plt.xlabel("Frequency [Hz]")
plt.ylabel("PSD [units²/Hz]")
plt.legend()
plt.show()
```

### 5. List available processes

```python
from src.noise import list_processes
print(list_processes())
# ['1/f', 'charge_noise_SiGe', 'charge_noise_SiMOS', 'ou', 'ou_sum',
#  'ou_sum_default', 'rtn', 'rtn_dominated', 'spatial_fluctuators']
```

### 6. Save and load results

```python
res.save("my_noise.npz")

from src.noise import NoiseResult
res_loaded = NoiseResult.load("my_noise.npz")
# All metadata preserved: seed, spec, fs, units
```

### 7. Custom time grid (non-uniform)

```python
import numpy as np

# Logarithmic time grid
t = np.logspace(-4, -1, 1000)  # 0.1ms to 0.1s

res = generate("ou", n_traj=100, t=t, seed=0,
               theta=10.0, S0=1e-12)
# Works with OU-family generators
```

### 8. Analyze trajectories

```python
from src.noise import welch, allan_deviation, autocorrelation

res = generate("1/f", n_traj=100, fs=1e4, n_points=2**14, seed=0)

# Power spectrum
f, S = welch(res.traj, fs=res.fs)

# Allan deviation
taus, sigma_a = allan_deviation(res.traj, fs=res.fs)

# Autocorrelation
lags, acf_vals = autocorrelation(res.traj, fs=res.fs)
```
