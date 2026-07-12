"""Tests for calibrate (from-trace surrogate) and the analysis functions.

Covers spec §7, §10:
- From-trace: surrogate PSD matches input-trace PSD.
- Multi-channel coherence preserved.
- Analysis functions (ACF, Allan, cross-correlation, cross-spectrum) sanity.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.noise import generate, calibrate, welch, autocorrelation, allan_deviation
from spinspectro.characterize import cross_correlation, coherence

# Statistical tolerances (duplicated from conftest to avoid a `tests` package
# name collision with site-packages on some installs).
RTOL_STAT = 0.15


class TestCalibrateSingleChannel:
    def test_surrogate_psd_matches_input(self):
        """Surrogate PSD matches the input-trace PSD within tolerance."""
        # make a 1/f trace to calibrate from
        src = generate("1/f", n_traj=1, fs=1e4, n_points=2**12,
                       alpha=1.0, S0=1.0, f0=1.0, f_min=1.0, seed=42)
        trace = src.traj[0]
        gen = calibrate(trace, fs=1e4)
        # generate many surrogates and measure their PSD
        res = gen.sample(n_traj=200, n_points=2**12, seed=0)
        f_in, S_in = welch(trace[np.newaxis, :], fs=1e4, nperseg=2**10)
        f_out, S_out = welch(res.traj, fs=1e4, nperseg=2**10)
        # compare in a band away from DC/Nyquist
        mask = (f_in > 2.0) & (f_in < 1e3)
        ratio = S_out[mask] / S_in[mask]
        # the surrogate reproduces the PSD statistically; allow generous scatter
        assert abs(np.median(ratio) - 1.0) < RTOL_STAT, (
            f"surrogate PSD ratio {np.median(ratio):.3f} != 1")

    def test_alpha_reported(self):
        """calibrate reports a reasonable alpha for a 1/f input."""
        src = generate("1/f", n_traj=1, fs=1e4, n_points=2**12,
                       alpha=1.0, S0=1.0, f0=1.0, f_min=1.0, seed=42)
        gen = calibrate(src.traj[0], fs=1e4)
        assert 0.5 < gen.alpha < 1.5

    def test_generate_from_array(self):
        """generate(ndarray, fs=...) dispatches to calibrate + sample."""
        src = generate("1/f", n_traj=1, fs=1e4, n_points=2**12, seed=42)
        res = generate(src.traj[0], n_traj=5, fs=1e4, n_points=2**10, seed=0)
        assert res.traj.shape == (5, 2**10)


class TestCalibrateMultiChannel:
    def test_multichannel_coherence_preserved(self):
        """Multi-channel surrogate preserves inter-channel coherence."""
        # build a 2-channel trace with known coherence via spatial separable
        K = np.array([[1.0, 0.7], [0.7, 1.0]])
        from src.noise.generators import generate_spatial_separable
        src = generate_spatial_separable(K, n_traj=1, fs=1e4, n_points=2**12,
                                         temporal="1/f", alpha=1.0, S0=1.0,
                                         f0=1.0, f_min=1.0, seed=42)
        trace = src.traj[0]  # (2, n_time)
        gen = calibrate(trace, fs=1e4)
        res = gen.sample(n_traj=100, n_points=2**12, seed=0)
        # measure coherence between the two channels in the surrogate
        x = res.traj[:, 0, :]  # (n_traj, n_time)
        y = res.traj[:, 1, :]
        f, gamma2, phase, _ = coherence(x, y, fs=1e4, nperseg=2**10)
        # coherence should be meaningfully > 0 in the band
        mask = (f > 2.0) & (f < 1e3)
        assert np.mean(gamma2[mask]) > 0.1, (
            f"surrogate coherence {np.mean(gamma2[mask]):.3f} too low")


class TestAnalysisFunctions:
    def test_acf_of_white_noise(self):
        """ACF of white noise ~ delta function (R[0] = var, R[k>0] ~ 0)."""
        rng = np.random.default_rng(0)
        x = rng.standard_normal(size=(200, 4096))
        tau, R = autocorrelation(x, fs=1e3, max_lag=100)
        assert abs(R[0] - np.var(x)) / np.var(x) < RTOL_STAT
        assert np.max(np.abs(R[1:])) / R[0] < RTOL_STAT

    def test_allan_white_noise_slope(self):
        """Allan deviation of white noise falls as tau^-1/2."""
        rng = np.random.default_rng(0)
        x = rng.standard_normal(size=(100, 8192))
        taus, sigma = allan_deviation(x, fs=1e3)
        # fit slope in log-log
        mask = np.isfinite(sigma) & (sigma > 0) & (taus > 2e-3) & (taus < 1.0)
        logf = np.log10(taus[mask])
        logS = np.log10(sigma[mask])
        A = np.vstack([np.ones_like(logf), logf]).T
        coef, *_ = np.linalg.lstsq(A, logS, rcond=None)
        slope = coef[1]
        assert abs(slope - (-0.5)) < 0.2, f"Allan slope {slope:.3f} != -0.5"

    def test_cross_correlation_symmetric(self):
        """Cross-correlation of a signal with itself == autocorrelation (symmetric)."""
        rng = np.random.default_rng(0)
        x = rng.standard_normal(size=(50, 1024))
        lags, Rxx = cross_correlation(x, x, fs=1.0, max_lag=50)
        # should be symmetric around lag 0
        center = np.argmax(np.abs(Rxx))
        assert abs(Rxx[center + 1] - Rxx[center - 1]) / abs(Rxx[center]) < RTOL_STAT or center == 0

    def test_coherence_of_identical_signals(self):
        """Coherence of identical signals ~ 1."""
        rng = np.random.default_rng(0)
        x = rng.standard_normal(size=(50, 1024))
        f, gamma2, phase, _ = coherence(x, x, fs=1e3, nperseg=512)
        mask = (f > 1.0) & (f < 400)
        assert np.mean(gamma2[mask]) > 0.9, (
            f"coherence of identical signals {np.mean(gamma2[mask]):.3f} < 0.9")