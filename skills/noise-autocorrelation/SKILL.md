---
name: noise-autocorrelation
description: Compute the autocorrelation function (ACF) of a noise trace via FFT (Wiener-Khinchin) and save a plot.
---

# noise-autocorrelation

Compute the autocorrelation function `R(τ)` via FFT (Wiener-Khinchin), ensemble-averaged, with biased/unbiased option.

## Inputs

- `trace`: path to a `.npy`/`.csv`/`.npz` file, or an in-memory `NoiseResult` / ndarray.
- `fs`: sample rate in Hz (required if a raw array is given; inferred from a `NoiseResult`).
- `biased`: bool (default `True`); normalize by `N` (biased) or `N-k` (unbiased).
- `max_lag`: optional maximum lag (default: all positive lags).
- `output`: directory for the saved figure and `.npz` of values (default `outputs/`).

## Outputs

- `outputs/noise_autocorrelation.png`: ACF vs lag.
- `outputs/noise_autocorrelation.npz`: `tau`, `R` arrays.

## Usage

```bash
python skills/noise-autocorrelation/run.py trace.npy --fs 1e4 --output outputs
```

## Notes

- Calls `openspin.noise.analysis.autocorrelation` so results are identical to library use.
- FFT-based, zero-padded to `>= 2N` to avoid circular correlation; ensemble-averaged.