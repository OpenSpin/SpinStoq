---
name: noise-crossspectrum
description: Compute the cross-spectral density, magnitude-squared coherence, and phase between two noise traces and save plots.
---

# noise-crossspectrum

Compute the cross-spectral density `S_xy(f)`, magnitude-squared coherence `γ²(f)`, and phase `∠S_xy(f)` between two traces.

## Inputs

- `x`, `y`: paths to `.npy`/`.csv`/`.npz` files, or in-memory `NoiseResult` / ndarrays.
- `fs`: sample rate in Hz (required if raw arrays are given; inferred from a `NoiseResult`).
- `nperseg`: Welch segment length (optional).
- `output`: directory for the saved figures and `.npz` of values (default `outputs/`).

## Outputs

- `outputs/noise_crossspectrum.png`: three-panel figure (cross-spectrum magnitude, coherence, phase).
- `outputs/noise_crossspectrum.npz`: `f`, `S_xy`, `gamma2`, `phase` arrays.

## Usage

```bash
python skills/noise-crossspectrum/run.py x.npy y.npy --fs 1e4 --output outputs
```

## Notes

- Calls `openspin.noise.analysis.coherence` so results are identical to library use.
- `γ²(f) = |S_xy|² / (S_xx * S_yy)` in `[0, 1]`; phase in radians.
- FFT-based, ensemble-averaged.