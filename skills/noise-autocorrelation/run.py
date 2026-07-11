"""noise-autocorrelation Skill: ACF via FFT (Wiener-Khinchin)."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _common import load_trace, save_figure, save_values, setup_plot

from openspin.noise.analysis import autocorrelation


def compute_autocorrelation(trace, fs=None, biased=True, max_lag=None,
                            output="outputs", plot=True, **kw):
    """Compute and optionally plot/save the ACF. Returns ``(tau, R)``."""
    arr, fs_eff, spec = load_trace(trace, fs=fs)
    tau, R = autocorrelation(arr, fs=fs_eff, biased=biased, max_lag=max_lag, **kw)

    if plot:
        fig = setup_plot()
        ax = fig.add_subplot(111)
        ax.plot(tau, R, "-", label="ACF")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("Lag τ [s]")
        ax.set_ylabel("R(τ)")
        ax.set_title(f"Autocorrelation ({'biased' if biased else 'unbiased'})")
        ax.legend()
        ax.grid(True, ls=":", alpha=0.4)
        save_figure(fig, output, "noise_autocorrelation.png")
    save_values(output, "noise_autocorrelation.npz", tau=tau, R=R)
    return tau, R


def main():
    ap = argparse.ArgumentParser(description="Compute autocorrelation via FFT.")
    ap.add_argument("trace", help="path to .npy/.csv/.npz trace, or a NoiseResult")
    ap.add_argument("--fs", type=float, default=None, help="sample rate [Hz]")
    ap.add_argument("--unbiased", action="store_true", help="use unbiased normalization")
    ap.add_argument("--max-lag", type=int, default=None)
    ap.add_argument("--output", default="outputs", help="output directory")
    args = ap.parse_args()
    tau, R = compute_autocorrelation(args.trace, fs=args.fs, biased=not args.unbiased,
                                     max_lag=args.max_lag, output=args.output)
    print(f"ACF: {tau.size} lags, R(0)={R[0]:.4g}")
    print(f"Saved to {args.output}/noise_autocorrelation.png and .npz")


if __name__ == "__main__":
    main()