"""noise-crossspectrum Skill: cross-spectral density, coherence, and phase."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _common import load_trace, save_figure, save_values, setup_plot

from openspin.noise.analysis import coherence


def compute_crossspectrum(x, y, fs=None, nperseg=None, output="outputs",
                          plot=True, **kw):
    """Compute and optionally plot/save cross-spectrum, coherence, phase.

    Returns ``(f, gamma2, phase, S_xy)``.
    """
    x_arr, fs_x, _ = load_trace(x, fs=fs)
    y_arr, fs_y, _ = load_trace(y, fs=fs)
    fs_eff = fs_x if fs is None else fs
    f, gamma2, phase, S_xy = coherence(x_arr, y_arr, fs=fs_eff, nperseg=nperseg, **kw)

    if plot:
        fig = setup_plot()
        fig.set_size_inches(7, 10)
        # panel 1: cross-spectrum magnitude
        ax1 = fig.add_subplot(3, 1, 1)
        mask = f > 0
        ax1.loglog(f[mask], np.abs(S_xy[mask]), "-")
        ax1.set_ylabel("|S_xy(f)|")
        ax1.set_title("Cross-spectrum")
        ax1.grid(True, which="both", ls=":", alpha=0.4)
        # panel 2: coherence
        ax2 = fig.add_subplot(3, 1, 2)
        ax2.semilogx(f[mask], gamma2[mask], "-")
        ax2.set_ylabel("Coherence γ²(f)")
        ax2.set_ylim(0, 1.05)
        ax2.grid(True, which="both", ls=":", alpha=0.4)
        # panel 3: phase
        ax3 = fig.add_subplot(3, 1, 3)
        ax3.semilogx(f[mask], np.degrees(phase[mask]), "-")
        ax3.set_ylabel("Phase [deg]")
        ax3.set_xlabel("Frequency [Hz]")
        ax3.grid(True, which="both", ls=":", alpha=0.4)
        save_figure(fig, output, "noise_crossspectrum.png")
    save_values(output, "noise_crossspectrum.npz", f=f, S_xy=S_xy,
                gamma2=gamma2, phase=phase)
    return f, gamma2, phase, S_xy


def main():
    ap = argparse.ArgumentParser(description="Compute cross-spectrum, coherence, and phase.")
    ap.add_argument("x", help="path to first trace (.npy/.csv/.npz) or NoiseResult")
    ap.add_argument("y", help="path to second trace (.npy/.csv/.npz) or NoiseResult")
    ap.add_argument("--fs", type=float, default=None, help="sample rate [Hz]")
    ap.add_argument("--nperseg", type=int, default=None)
    ap.add_argument("--output", default="outputs", help="output directory")
    args = ap.parse_args()
    f, gamma2, phase, S_xy = compute_crossspectrum(args.x, args.y, fs=args.fs,
                                                   nperseg=args.nperseg, output=args.output)
    print(f"Cross-spectrum: {f.size} freq bins, mean coherence={np.mean(gamma2[1:]):.3f}")
    print(f"Saved to {args.output}/noise_crossspectrum.png and .npz")


if __name__ == "__main__":
    main()