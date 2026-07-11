"""noise-allan Skill: overlapping Allan deviation via cumulative sums."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _common import load_trace, save_figure, save_values, setup_plot

from openspin.noise.analysis import allan_deviation


def compute_allan(trace, fs=None, taus=None, output="outputs", plot=True, **kw):
    """Compute and optionally plot/save the overlapping Allan deviation.

    Returns ``(taus, sigma)``.
    """
    arr, fs_eff, spec = load_trace(trace, fs=fs)
    taus, sigma = allan_deviation(arr, fs=fs_eff, taus=taus, **kw)

    if plot:
        fig = setup_plot()
        ax = fig.add_subplot(111)
        mask = np.isfinite(sigma) & (sigma > 0)
        ax.loglog(taus[mask], sigma[mask], "o-", ms=3, label="measured")
        # reference slope lines
        if np.any(mask):
            t0 = taus[mask][0]
            s0 = sigma[mask][0]
            for slope, label, color in [(-0.5, "white (-1/2)", "C1"),
                                        (0.0, "1/f (0)", "C2"),
                                        (0.5, "random-walk (+1/2)", "C3")]:
                ref = s0 * (taus[mask] / t0) ** slope
                ax.loglog(taus[mask], ref, "--", alpha=0.4, color=color, label=label)
        ax.set_xlabel("Averaging time τ [s]")
        ax.set_ylabel("Allan deviation σ_A(τ)")
        ax.set_title("Overlapping Allan deviation")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", ls=":", alpha=0.4)
        save_figure(fig, output, "noise_allan.png")
    save_values(output, "noise_allan.npz", taus=taus, sigma=sigma)
    return taus, sigma


def main():
    ap = argparse.ArgumentParser(description="Compute overlapping Allan deviation.")
    ap.add_argument("trace", help="path to .npy/.csv/.npz trace, or a NoiseResult")
    ap.add_argument("--fs", type=float, default=None, help="sample rate [Hz]")
    ap.add_argument("--output", default="outputs", help="output directory")
    args = ap.parse_args()
    taus, sigma = compute_allan(args.trace, fs=args.fs, output=args.output)
    print(f"Allan deviation: {taus.size} tau points")
    print(f"Saved to {args.output}/noise_allan.png and .npz")


if __name__ == "__main__":
    main()