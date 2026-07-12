# SpinStoq

**Fast correlated (temporal & spatial) charge/qubit noise generator for spin-qubit simulation.**

NumPy-first, FFT/`O(N log N)` where possible, no Python-level per-sample loops on the hot path. Vectorized over the ensemble — generate thousands of trajectories in one shot.

## Install

```bash
# spinspectro isn't on PyPI yet — install it from the sibling SpinSkills repo first
pip install -e ../SpinSkills/SpinSkills/spinspectro
pip install -e .
# optional acceleration backends
pip install openspin-noise[numba]   # JIT-compiled OU recursion
pip install openspin-noise[jax]     # GPU/TPU OU recursion
pip install openspin-noise[viz]     # matplotlib for the Skill plots
```

## Quick start — 3 copy-paste examples

### 1. By name: 1/f noise

```python
from src.noise import generate

res = generate(
    "1/f",
    n_traj=1000,
    fs=1e4,               # sample rate [Hz]
    n_points=2**16,
    alpha=1.0,            # 1/f^alpha
    S0=1e-12, f0=1.0,     # PSD normalization: S(f0)=S0  [units^2/Hz]
    seed=0,
)
res.traj   # ndarray (1000, 65536), float64
res.t      # ndarray (65536,) time grid [s]
res.fs, res.spec, res.seed, res.units
```

### 2. OU-sum (physically-motivated 1/f)

```python
from src.noise import generate

res = generate(
    "ou_sum",
    n_traj=1000,
    fs=1e4,
    n_points=2**16,
    alpha=1.0,            # 1/f^alpha via a sum of Lorentzians
    S0=1e-12, f0=1.0,
    n_components_per_decade=8,
    seed=0,
    backend="numba",      # optional: JIT-compiled time recursion
)
```

### 3. From a measured trace (estimate + resynthesize)

```python
import numpy as np
from src.noise import calibrate

# trace: 1D array (n_time,) or 2D (n_channels, n_time) from your experiment
trace = np.load("my_charge_sensor_trace.npy")
gen = calibrate(trace, fs=1e4, method="circulant")  # model-free surrogate
res = gen.sample(n_traj=1000, n_points=2**16, seed=0)
```

## Generate a figure

```python
from src.noise import generate, welch
import matplotlib.pyplot as plt

res = generate("1/f", n_traj=200, fs=1e4, n_points=2**14,
               alpha=1.0, S0=1.0, f0=1.0, seed=0)
f, S = welch(res.traj, fs=res.fs)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(res.t[:1000], res.traj[0, :1000])
ax1.set(xlabel="Time [s]", ylabel="Noise", title="Sample trajectory")
ax2.loglog(f[1:], S[1:], label="measured")
ax2.loglog(f[1:], 1.0/f[1:], "k--", alpha=0.5, label="target 1/f")
ax2.set(xlabel="Frequency [Hz]", ylabel="PSD [units²/Hz]", title="Spectrum")
ax2.legend()
fig.tight_layout()
fig.savefig("openspin_noise_demo.png", dpi=120)
```

## Named processes (registry)

```python
from src.noise import list_processes
print(list_processes())
# ['1/f', 'charge_noise_SiGe', 'charge_noise_SiMOS', 'ou', 'ou_sum',
#  'ou_sum_default', 'rtn', 'rtn_dominated', 'spatial_fluctuators']
```

| Name | Method | Description |
|------|--------|-------------|
| `1/f` | Timmer-König | Stationary Gaussian 1/f^alpha via FFT |
| `ou` | exact OU | Scalar/multivariate Ornstein-Uhlenbeck (exact propagator) |
| `ou_sum` | sum of Lorentzians | 1/f^alpha as a sum of OU components |
| `rtn` | random telegraph | Two-level fluctuators |
| `spatial_fluctuators` | distributed fluctuators | Spatially correlated noise via a kernel |
| `charge_noise_SiGe` | preset | Literature-ish SiGe 1/f charge noise |
| `charge_noise_SiMOS` | preset | Literature-ish SiMOS 1/f charge noise |
| `rtn_dominated` | preset | RTN-dominated noise |

## Analysis (via the `spinspectro` Skill)

Trace characterization (PSD, ACF, Allan deviation, cross-correlation,
cross-spectrum/coherence) is not implemented here — it lives in
[`spinspectro`](../SpinSkills/SpinSkills/spinspectro), the shared SpinSkills
package/Claude Skill, so results stay comparable across the OpenSpin
community instead of drifting between per-repo copies. `generate()` and
`calibrate()` re-export its functions for convenience:

```python
from src.noise import generate, welch, allan_deviation

res = generate("1/f", n_traj=100, fs=1e4, n_points=2**14, seed=0)
f, S = welch(res.traj, fs=res.fs)
taus, sigma = allan_deviation(res.traj, fs=res.fs)
```

Or call it directly (same functions, plus file I/O and plotting — see
[spinspectro's SKILL.md](../SpinSkills/SpinSkills/spinspectro/SKILL.md)):

```python
from spinspectro import compute_spectrum, compute_allan

f, S = compute_spectrum("trace.npy", fs=1e4, method="welch")
taus, sigma = compute_allan("trace.npy", fs=1e4)
```

## Design principles

1. **NumPy-first.** Pure vectorized NumPy + `numpy.fft`. No SciPy hard-dependency.
2. **Optional acceleration behind one API.** `backend="numpy"` (default), `"numba"`, `"jax"`.
3. **Vectorize over the ensemble.** All generators produce `(n_traj, [n_sites,] n_time)` in one shot.
4. **Exactness where it's free.** Exact OU propagator (not Euler-Maruyama); circulant embedding.
5. **Reproducibility.** Explicit `SeedSequence`; every result records its seed and full process spec.
6. **Round-trip validated.** Tests assert measured PSD/ACF matches the target.

## Conventions

- **PSD is one-sided**, `S(f)` for `f ≥ 0`, in `units²/Hz`, with `var(x) = ∫₀^{fs/2} S(f) df`.
- Wiener–Khinchin: `R(τ) = ∫₀^∞ S(f) cos(2πfτ) df`.
- Cyclic frequency `f` in **Hz** everywhere. OU rate `θ` in `1/s`; Lorentzian corner at `f = θ/2π`.

## API reference

### `generate(name_or_trace, *, n_traj, fs, n_points, t, seed, backend, dtype, **kwargs)`

Generate noise by name or from a measured trace.

- **string** → registry lookup (e.g. `"1/f"`, `"ou"`, `"ou_sum"`, `"rtn"`)
- **ndarray** → `calibrate(trace, fs).sample(...)` (model-free surrogate)
- Accept either `fs` + `n_points` (uniform grid) or `t=` (arbitrary/non-uniform grid)

### `calibrate(trace, fs, method="circulant")`

Estimate second-order statistics from a measured trace and return a
`SurrogateGenerator`. Call `gen.sample(n_traj, n_points, seed)` to emit
surrogate trajectories that reproduce the measured PSD/ACF (and cross-spectra
for multi-channel input).

### `NoiseResult`

Thin dataclass wrapping the array + metadata:

```python
res.traj      # ndarray (n_traj, [n_sites,] n_time)
res.t         # ndarray (n_time,) time grid [s]
res.fs        # float, sample rate [Hz] (None for non-uniform)
res.spec      # dict, full process specification
res.seed      # the seed used
res.units     # str, e.g. "V" or "a.u."
res.n_traj, res.n_time, res.n_sites
res.is_uniform
res.save(path) / NoiseResult.load(path)   # .npz
res.to_xarray()                            # optional, requires xarray
```

## Testing

```bash
pytest tests/ -q
```

The suite includes the convention test (variance = ∫ PSD), OU exactness at
large Δt, OU-sum slope, spatial K recovery, backend parity, and from-trace
surrogate validation.

## Benchmarks

```bash
python benchmarks/bench_noise.py
```

Indicative targets (laptop, single core, numpy backend):
- 1/f: 10⁴ traj × 2¹⁶ pts in ≲ 1 s
- OU-sum: 10³ traj × 10⁵ pts in ≲ 1 s
- All analysis: 10³ × 10⁵ in ≲ 1 s

## License

MIT