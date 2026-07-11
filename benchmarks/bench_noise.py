"""Benchmarks for openspin.noise — regression guards for the §9 performance targets.

Run directly:
    python benchmarks/bench_noise.py

Or via pytest (does not assert times, just reports):
    pytest benchmarks/bench_noise.py -s

Indicative targets (laptop, single core, numpy backend):
- 1/f (method 1): 10^4 traj x 2^16 pts in <= 1 s
- OU-sum (method 3, 8 comps/decade, 4 decades): 10^3 traj x 10^5 pts in <= 1 s
- All analysis Skills: 10^3 x 10^5 in <= 1 s
"""
from __future__ import annotations

import os
import sys
import time

# allow running as `python benchmarks/bench_noise.py` without installing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from openspin.noise import generate, welch, autocorrelation, allan_deviation
from openspin.noise.analysis import cross_correlation, coherence


def _time(fn, *args, **kw):
    t0 = time.perf_counter()
    res = fn(*args, **kw)
    return time.perf_counter() - t0, res


def bench_1f():
    """1/f via Timmer-König: 10^4 traj x 2^16 pts."""
    t, res = _time(generate, "1/f", n_traj=10_000, fs=1e4, n_points=2**16,
                   alpha=1.0, S0=1.0, f0=1.0, seed=0)
    print(f"  1/f (10^4 x 2^16): {t:.3f} s  [target <= 1.0 s]  {'OK' if t <= 1.0 else 'SLOW'}")
    return t


def bench_ou_sum():
    """OU-sum: 10^3 traj x 10^5 pts, 8 comps/decade, ~4 decades."""
    t, res = _time(generate, "ou_sum", n_traj=1000, fs=1e4, n_points=100_000,
                   alpha=1.0, S0=1.0, f0=1.0, seed=0)
    print(f"  OU-sum (10^3 x 10^5): {t:.3f} s  [target <= 1.0 s]  {'OK' if t <= 1.0 else 'SLOW'}")
    return t


def bench_analysis():
    """Analysis functions: 10^3 traj x 10^5 pts."""
    res = generate("1/f", n_traj=1000, fs=1e4, n_points=100_000, seed=0)
    x = res.traj
    # split into two halves for cross-spectrum
    x1, x2 = x[:500], x[500:]

    t1, _ = _time(welch, x, fs=1e4, nperseg=2**12)
    print(f"  welch (10^3 x 10^5): {t1:.3f} s")
    t2, _ = _time(autocorrelation, x, fs=1e4, max_lag=1000)
    print(f"  autocorrelation (10^3 x 10^5): {t2:.3f} s")
    t3, _ = _time(allan_deviation, x, fs=1e4)
    print(f"  allan_deviation (10^3 x 10^5): {t3:.3f} s")
    t4, _ = _time(cross_correlation, x1, x2, fs=1e4, max_lag=1000)
    print(f"  cross_correlation (500 x 10^5): {t4:.3f} s")
    t5, _ = _time(coherence, x1, x2, fs=1e4, nperseg=2**12)
    print(f"  coherence (500 x 10^5): {t5:.3f} s")
    total = t1 + t2 + t3 + t4 + t5
    print(f"  total analysis: {total:.3f} s  [target <= 1.0 s]  {'OK' if total <= 1.0 else 'SLOW'}")
    return total


def bench_ou_backends():
    """OU recursion: numpy vs numba, 10^3 traj x 10^5 pts."""
    from openspin.noise import list_backends
    for backend in ["numpy"]:
        t, _ = _time(generate, "ou", n_traj=1000, fs=1e4, n_points=100_000,
                     gamma=10.0, sigma=1.0, seed=0, backend=backend)
        print(f"  OU {backend} (10^3 x 10^5): {t:.3f} s")
    if "numba" in list_backends():
        t, _ = _time(generate, "ou", n_traj=1000, fs=1e4, n_points=100_000,
                     gamma=10.0, sigma=1.0, seed=0, backend="numba")
        print(f"  OU numba (10^3 x 10^5): {t:.3f} s  [target >= 5x faster than numpy]")


def main():
    print("openspin.noise benchmarks (spec §9 targets)")
    print("=" * 60)
    print("Generation:")
    bench_1f()
    bench_ou_sum()
    print("\nOU backends:")
    bench_ou_backends()
    print("\nAnalysis:")
    bench_analysis()
    print("=" * 60)


if __name__ == "__main__":
    main()