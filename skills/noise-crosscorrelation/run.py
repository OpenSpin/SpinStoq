"""noise-crosscorrelation Skill: cross-correlation between two traces/sites."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _common import load_trace, save_figure, save_values, setup_plot

from openspin.noise.analysis import cross_correlation


def compute_crosscorrelation(x, y, fs=None, max_lag=None, normalize=False,
                             output="outputs", plot=True, **kw):
    """Compute and optionally plot/save the cross-correlation. Returns ``(lags, R)``."""
    x_arr, fs_x, _ = load_trace(x, fs=fs)
    y_arr, fs_y, _ = load_trace(y, fs=fs)
    fs_eff = fs_x if fs is None else fs
    lags, R = cross_correlation(x_arr, y_arr, fs=fs_eff, max_lag=max_lag,
                                normalize=normalize, **kw)

    if plot:
        fig = setup_plot()
        ax = fig.add_subplot(111)
        ax.plot(lags, R, "-", label="R_xy(τ)")
        ax.axhline(0, color="k", lw=0.5)
        ax.axvline(0, color="k", lw=0.5, ls="--", alpha=0.5)
        ax.set_xlabel("Lag τ [s]")
        ax.set_ylabel("R_xy(τ)" + (" (normalized)" if normalize else ""))
        ax.set_title("Cross-correlation")
        ax.legend()
        ax.grid(True, ls=":", alpha=0.4)
        save_figure(fig, output, "noise_crosscorrelation.png")
    save_values(output, "noise_crosscorrelation.npz", lags=lags, R=R)
    return lags, R


def main():
    ap = argparse.ArgumentParser(description="Compute cross-correlation between two traces.")
    ap.add_argument("x", help="path to first trace (.npy/.csv/.npz) or NoiseResult")
    ap.add_argument("y", help="path to second trace (.npy/.csv/.npz) or NoiseResult")
    ap.add_argument("--fs", type=float, default=None, help="sample rate [Hz]")
    ap.add_argument("--max-lag", type=int, default=None)
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--output", default="outputs", help="output directory")
    args = ap.parse_args()
    lags, R = compute_crosscorrelation(args.x, args.y, fs=args.fs, max_lag=args.max_lag,
                                       normalize=args.normalize, output=args.output)
    print(f"Cross-correlation: {lags.size} lags, peak={R[np.argmax(np.abs(R))]:.4g} at τ={lags[np.argmax(np.abs(R))]:.4g} s")
    print(f"Saved to {args.output}/noise_crosscorrelation.png and .npz")


if __name__ == "__main__":
    main()