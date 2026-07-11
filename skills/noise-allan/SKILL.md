---
name: noise-allan
description: Compute the overlapping Allan deviation of a noise trace and save a log-log plot with slope annotations.
---

# noise-allan

Compute the overlapping Allan deviation `σ_A(τ)` via cumulative sums (`O(N)` per τ, τ log-spaced). Good 1/f diagnostic; annotate slopes.

## Inputs

- `trace`: path to a `.npy`/`.csv`/`.npz` file, or an in-memory `NoiseResult` / ndarray.
- `fs`: sample rate in Hz (required if a raw array is given; inferred from a `NoiseResult`).
- `taus`: optional requested averaging times [s] (default: log-spaced from `1/fs` to `n_time/(2*fs)`).
- `output`: directory for the saved figure and `.npz` of values (default `outputs/`).

## Outputs

- `outputs/noise_allan.png`: log-log Allan deviation with reference slope lines (white `-1/2`, 1/f `0`, random-walk `+1/2`).
- `outputs/noise_allan.npz`: `taus`, `sigma` arrays.

## Usage

```bash
python skills/noise-allan/run.py trace.npy --fs 1e4 --output outputs
```

## Notes

- Calls `openspin.noise.analysis.allan_deviation` so results are identical to library use.
- Overlapping estimator (IEEE standard), ensemble-averaged, `O(N)` per τ via cumulative sums.