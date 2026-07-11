"""noise-spectrum Skill: one-sided PSD via Welch/periodogram/multitaper.

Takes a trace file (`.npy`/`.csv`/`.npz`) or a `NoiseResult` and returns the
ensemble-averaged one-sided PSD + a log-log plot. Calls the shared
`openspin.noise.analysis` functions so results are identical to library use.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# allow running both as `python skills/noise-spectrum/run.py` and as a module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _common import load_trace, save_figure, save_values, setup_plot

from openspin.noise.analysis import welch, periodogram, multitaper


def compute_spectrum(trace, fs=None, method="welch", nperseg=None, output="outputs",
                     plot=True, **kw):
    """Compute and optionally plot/save the one-sided PSD.

    Returns ``(f, S)``.
    """
    arr, fs_eff, spec = load_trace(trace, fs=fs)
    if method == "welch":
        f, S = welch(arr, fs=fs_eff, nperseg=nperseg, **kw)
    elif method == "periodogram":
        f, S = periodogram(arr, fs=fs_eff, **kw)
    elif method == "multitaper":
        f, S = multitaper(arr, fs=fs_eff, **kw)
    else:
        raise ValueError(f"unknown method {method!r}")

    if plot:
        fig = setup_plot()
        ax = fig.add_subplot(111)
        mask = (f > 0) & (S > 0)
        ax.loglog(f[mask], S[mask], label=f"measured ({method})")
        # overlay target if known
        if spec and "alpha" in spec and "S0" in spec and "f0" in spec:
            S_target = spec["S0"] * (spec["f0"] / f[mask]) ** spec["alpha"]
            ax.loglog(f[mask], S_target, "k--", alpha=0.5, label=f"target 1/f^{spec['alpha']}")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel(f"PSD [units²/Hz]")
        ax.set_title("Noise spectrum")
        ax.legend()
        ax.grid(True, which="both", ls=":", alpha=0.4)
        save_figure(fig, output, "noise_spectrum.png")
    save_values(output, "noise_spectrum.npz", f=f, S=S)
    return f, S


def main():
    ap = argparse.ArgumentParser(description="Compute one-sided noise PSD (Welch/periodogram/multitaper).")
    ap.add_argument("trace", help="path to .npy/.csv/.npz trace, or a NoiseResult")
    ap.add_argument("--fs", type=float, default=None, help="sample rate [Hz] (if raw array)")
    ap.add_argument("--method", default="welch", choices=["welch", "periodogram", "multitaper"])
    ap.add_argument("--nperseg", type=int, default=None)
    ap.add_argument("--output", default="outputs", help="output directory")
    args = ap.parse_args()
    f, S = compute_spectrum(args.trace, fs=args.fs, method=args.method,
                            nperseg=args.nperseg, output=args.output)
    print(f"PSD: {f.size} freq bins, fs={1.0/(f[1]-f[0]) if f.size>1 else '?'} Hz")
    print(f"Saved to {args.output}/noise_spectrum.png and .npz")


if __name__ == "__main__":
    main()