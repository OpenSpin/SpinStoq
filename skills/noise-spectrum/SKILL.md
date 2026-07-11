---
name: noise-spectrum
description: Compute the one-sided power spectral density (Welch/periodogram/multitaper) of a noise trace or NoiseResult and save a log-log plot.
---

# noise-spectrum

Compute the one-sided power spectral density (PSD) of a noise trace and produce a quick log-log plot.

## Inputs

- `trace`: path to a `.npy`/`.csv`/`.npz` file, or an in-memory `NoiseResult` / ndarray of shape `(n_traj, [n_sites,] n_time)` or `(n_time,)`.
- `fs`: sample rate in Hz (required if a raw array is given; inferred from a `NoiseResult`).
- `method`: `welch` (default), `periodogram`, or `multitaper`.
- `nperseg`: Welch segment length (optional).
- `output`: directory for the saved figure and `.npz` of values (default `outputs/`).

## Outputs

- `outputs/noise_spectrum.png`: log-log PSD plot (ensemble-averaged), with the target PSD overlaid if the input is a `NoiseResult` with a known target.
- `outputs/noise_spectrum.npz`: `f`, `S` arrays.

## Usage

```bash
python skills/noise-spectrum/run.py trace.npy --fs 1e4 --method welch --output outputs
```

```python
from openspin.noise import generate
from skills.noise_spectrum.run import compute_spectrum
res = generate("1/f", n_traj=100, fs=1e4, n_points=2**14, seed=0)
f, S = compute_spectrum(res, method="welch")
```

## Notes

- Calls `openspin.noise.analysis.welch` / `periodogram` / `multitaper` so results are identical whether invoked as a library or a Skill.
- One-sided PSD in `units^2/Hz` satisfying `var(x) = integral_0^{fs/2} S(f) df`.
- FFT-based and vectorized over the ensemble; `O(n_traj * N log N)`.