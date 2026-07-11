---
name: noise-crosscorrelation
description: Compute the cross-correlation between two noise traces/sites (FFT-based, with lag axis) and save a plot.
---

# noise-crosscorrelation

Compute the cross-correlation `R_xy(τ)` between two traces/sites, FFT-based, with a symmetric lag axis.

## Inputs

- `x`, `y`: paths to `.npy`/`.csv`/`.npz` files, or in-memory `NoiseResult` / ndarrays of shape `(n_traj, n_time)` or `(n_time,)`.
- `fs`: sample rate in Hz (required if raw arrays are given; inferred from a `NoiseResult`).
- `max_lag`: optional half-width of the lag window (default: `n_time - 1`).
- `normalize`: bool (default `False`); return normalized cross-correlation coefficient.
- `output`: directory for the saved figure and `.npz` of values (default `outputs/`).

## Outputs

- `outputs/noise_crosscorrelation.png`: cross-correlation vs lag.
- `outputs/noise_crosscorrelation.npz`: `lags`, `R` arrays.

## Usage

```bash
python skills/noise-crosscorrelation/run.py x.npy y.npy --fs 1e4 --output outputs
```

## Notes

- Calls `openspin.noise.analysis.cross_correlation` so results are identical to library use.
- FFT-based, zero-padded to `>= 2N`; ensemble-averaged.