"""Tests for the noise generators: round-trip PSD/ACF, OU exactness, slopes, spatial, backend parity.

Covers spec §10:
- Round-trip: measured PSD/ACF matches target within tolerance.
- Convention test: variance = integral of one-sided PSD.
- OU exactness: stationary variance sigma^2/(2*theta) and ACF exp(-theta|tau|)
  at large AND pathologically large dt (proves the exact scheme vs Euler).
- OU-sum: slope ~ -alpha in log-log PSD across [theta_min, theta_max]/2pi.
- Spatial: recovered K/coherence matches the target.
- Backend parity: numpy vs numba agree within fp tolerance for a fixed seed.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.noise import (
    generate, NoiseResult, list_backends, welch, autocorrelation,
)
from src.noise.generators import (
    timmer_koenig, generate_ou, generate_ou_sum, generate_rtn,
    generate_spatial_separable, generate_spatial_fluctuators, compose,
)
from spinspectro.characterize import coherence

# Statistical tolerances (duplicated from conftest to avoid a `tests` package
# name collision with site-packages on some installs).
RTOL_STAT = 0.15   # ±15% for PSD/variance round-trips with n_traj ~ 200
RTOL_SLOPE = 0.25  # ±25% for the OU-sum 1/f slope (log-log fit noise)
ATOL_FP = 1e-9     # backend parity (numpy vs numba) for a fixed seed


# --------------------------------------------------------------------------- #
# Convention test (spec §5): variance = integral of one-sided PSD
# --------------------------------------------------------------------------- #
class TestConvention:
    def test_variance_equals_psd_integral_white(self):
        """White noise: var == sum(S) * df."""
        rng = np.random.default_rng(0)
        x = rng.standard_normal(size=(200, 4096))
        fs = 1e3
        f, S = welch(x, fs=fs, window="rect", detrend="mean")
        df = f[1] - f[0]
        var = np.var(x)
        integral = np.sum(S) * df
        assert abs(integral - var) / var < RTOL_STAT, (
            f"PSD integral {integral:.4g} != variance {var:.4g}")

    def test_variance_equals_psd_integral_1f(self):
        """1/f noise: var == integral of measured PSD."""
        res = generate("1/f", n_traj=200, fs=1e4, n_points=2**12,
                       alpha=1.0, S0=1.0, f0=1.0, f_min=1.0, seed=0)
        f, S = welch(res.traj, fs=res.fs, window="rect", detrend="mean")
        df = f[1] - f[0]
        var = np.var(res.traj)
        integral = np.sum(S) * df
        assert abs(integral - var) / var < RTOL_STAT, (
            f"PSD integral {integral:.4g} != variance {var:.4g}")

    def test_one_sided_psd_doubles_positive_freqs(self):
        """One-sided PSD should be ~2x the two-sided at positive freqs."""
        rng = np.random.default_rng(0)
        x = rng.standard_normal(size=(400, 8192))
        fs = 1e3
        f, S_one = welch(x, fs=fs, window="rect", detrend="mean")
        # two-sided via full FFT (no doubling), ensemble-averaged
        X = np.fft.fft(x - x.mean(axis=-1, keepdims=True), axis=-1)
        S_two = np.mean(np.abs(X) ** 2, axis=0) / (fs * x.shape[-1])
        # compare at a positive frequency bin (ensemble-averaged both sides)
        k = 10
        assert abs(S_one[k] - 2.0 * S_two[k]) / S_one[k] < RTOL_STAT, (
            f"one-sided {S_one[k]:.4e} != 2*two-sided {2.0*S_two[k]:.4e}")


# --------------------------------------------------------------------------- #
# OU exactness (spec §6.2, §10)
# --------------------------------------------------------------------------- #
class TestOUExactness:
    @pytest.mark.parametrize("gamma,sigma", [(1.0, 1.0), (10.0, 2.0), (100.0, 0.5)])
    def test_stationary_variance(self, gamma, sigma):
        """Measured variance matches sigma^2/(2*gamma) (exact, any dt)."""
        res = generate_ou(n_traj=400, fs=1e3, n_points=2**12,
                          gamma=gamma, sigma=sigma, seed=0)
        target = sigma**2 / (2.0 * gamma)
        measured = np.var(res.traj)
        assert abs(measured - target) / target < RTOL_STAT, (
            f"OU var {measured:.4g} != target {target:.4g}")

    def test_acf_exponential(self):
        """ACF ~ exp(-gamma|tau|)."""
        gamma = 5.0
        res = generate_ou(n_traj=400, fs=1e3, n_points=2**12,
                          gamma=gamma, sigma=1.0, seed=0)
        tau, R = autocorrelation(res.traj, fs=res.fs, max_lag=200)
        R_norm = R / R[0]
        target = np.exp(-gamma * tau)
        # compare in the early lags where the estimate is reliable
        mask = tau < 0.3
        err = np.max(np.abs(R_norm[mask] - target[mask]))
        assert err < RTOL_STAT, f"ACF max error {err:.3f}"

    def test_exact_at_large_dt(self):
        """Exact scheme correct at pathologically large dt (gamma*dt >> 1).

        Euler-Maruyama would blow up / drift here; the exact propagator stays
        at the stationary variance.
        """
        gamma = 1e3  # huge rate
        dt = 1.0     # gamma*dt = 1000 >> 1
        n = 2**10
        t = np.arange(n) * dt
        res = generate_ou(n_traj=400, t=t, gamma=gamma, sigma=1.0, seed=0)
        target = 1.0 / (2.0 * gamma)
        measured = np.var(res.traj)
        # at large dt, consecutive samples are ~independent draws from the
        # stationary distribution, so the variance should still match.
        assert abs(measured - target) / target < RTOL_STAT, (
            f"OU var at large dt {measured:.4g} != target {target:.4g}")

    def test_nonzero_force_mean(self):
        """F != 0 -> steady-state mean F/gamma."""
        gamma, sigma, F = 5.0, 1.0, 2.0
        res = generate_ou(n_traj=400, fs=1e3, n_points=2**12,
                          gamma=gamma, sigma=sigma, F=F, seed=0)
        target_mean = F / gamma
        measured = np.mean(res.traj)
        assert abs(measured - target_mean) / abs(target_mean) < RTOL_STAT, (
            f"OU mean {measured:.4g} != target {target_mean:.4g}")


# --------------------------------------------------------------------------- #
# OU-sum slope (spec §6.3, §10)
# --------------------------------------------------------------------------- #
class TestOUSumSlope:
    @pytest.mark.parametrize("alpha", [1.0, 1.5, 0.7])
    def test_psd_slope(self, alpha):
        """Measured PSD slope ~ -alpha across [theta_min, theta_max]/2pi."""
        fs = 1e4
        n = 2**14
        res = generate_ou_sum(n_traj=200, fs=fs, n_points=n, alpha=alpha,
                              S0=1.0, f0=1.0, n_components_per_decade=8, seed=0)
        f, S = welch(res.traj, fs=fs, nperseg=2**12)
        # fit slope in the band bracketed by the rate range
        theta_min = res.spec["theta_min"]
        theta_max = res.spec["theta_max"]
        f_lo = theta_min / (2 * np.pi) * 2
        f_hi = theta_max / (2 * np.pi) / 2
        mask = (f > f_lo) & (f < f_hi) & (S > 0)
        logf = np.log10(f[mask])
        logS = np.log10(S[mask])
        A = np.vstack([np.ones_like(logf), logf]).T
        coef, *_ = np.linalg.lstsq(A, logS, rcond=None)
        measured_alpha = -coef[1]
        assert abs(measured_alpha - alpha) / alpha < RTOL_SLOPE, (
            f"OU-sum slope alpha={measured_alpha:.3f} != target {alpha}")


# --------------------------------------------------------------------------- #
# 1/f round-trip (spec §6.1, §10)
# --------------------------------------------------------------------------- #
class TestSpectralRoundTrip:
    def test_1f_psd_matches_target(self):
        """Measured PSD of 1/f matches the analytic target shape."""
        alpha, S0, f0 = 1.0, 2.0, 1.0
        res = timmer_koenig(n_traj=200, fs=1e4, n_points=2**14,
                            alpha=alpha, S0=S0, f0=f0, f_min=1.0, seed=0)
        f, S = welch(res.traj, fs=res.fs, nperseg=2**12)
        # compare in a band away from DC and Nyquist
        mask = (f > 2.0) & (f < 1e3) & (S > 0)
        target = S0 * (f0 / f[mask]) ** alpha
        # ratio should be ~1 (statistical scatter)
        ratio = S[mask] / target
        assert abs(np.median(ratio) - 1.0) < RTOL_STAT, (
            f"1/f PSD median ratio {np.median(ratio):.3f} != 1")


# --------------------------------------------------------------------------- #
# RTN (spec §6.5)
# --------------------------------------------------------------------------- #
class TestRTN:
    def test_rtn_levels(self):
        """RTN only takes the two telegraph levels."""
        res = generate_rtn(n_traj=10, fs=1e4, n_points=2**10,
                           gamma_up=100.0, gamma_down=100.0,
                           x_low=-1.0, x_high=1.0, seed=0)
        unique = np.unique(res.traj)
        assert len(unique) == 2
        assert -1.0 in unique or np.any(np.isclose(unique, -1.0))
        assert 1.0 in unique or np.any(np.isclose(unique, 1.0))

    def test_rtn_stationary_probability(self):
        """P(high) ~ gamma_up / (gamma_up + gamma_down)."""
        gu, gd = 200.0, 100.0
        res = generate_rtn(n_traj=50, fs=1e4, n_points=2**12,
                           gamma_up=gu, gamma_down=gd, seed=0)
        p_high = np.mean(res.traj > 0)
        target = gu / (gu + gd)
        assert abs(p_high - target) < RTOL_STAT


# --------------------------------------------------------------------------- #
# Spatial (spec §6.4, §10)
# --------------------------------------------------------------------------- #
class TestSpatial:
    def test_separable_recovers_K(self):
        """Spatial separable: recovered covariance ~ target K.

        We measure the per-trajectory time-averaged covariance across sites,
        then average over the ensemble. This avoids the bias from flattening
        correlated time samples into ``np.cov``.
        """
        K = np.array([[1.0, 0.5, 0.2],
                      [0.5, 1.0, 0.5],
                      [0.2, 0.5, 1.0]])
        res = generate_spatial_separable(K, n_traj=400, fs=1e4, n_points=2**12,
                                         temporal="1/f", alpha=1.0, S0=1.0,
                                         f0=1.0, f_min=1.0, seed=0)
        x = res.traj  # (n_traj, M, n_time)
        # per-trajectory time-averaged covariance: (M, M) per traj
        K_per = np.empty((x.shape[0], x.shape[1], x.shape[1]))
        for i in range(x.shape[0]):
            K_per[i] = np.cov(x[i])  # cov across the time axis for each traj
        K_meas = np.mean(K_per, axis=0)
        # normalize by diagonal (the temporal variance is arbitrary)
        d = np.sqrt(np.diag(K_meas))
        K_meas_norm = K_meas / np.outer(d, d)
        assert np.allclose(K_meas_norm, K, atol=0.2), (
            f"recovered K\n{K_meas_norm}\n!= target\n{K}")

    def test_fluctuators_shape(self):
        """Spatial fluctuators produce the right output shape."""
        sites = np.array([[0.0], [1.0], [2.0]])
        res = generate_spatial_fluctuators(sites, n_traj=3, fs=1e4,
                                           n_points=2**10, n_fluctuators=50, seed=0)
        assert res.traj.shape == (3, 3, 2**10)


# --------------------------------------------------------------------------- #
# Backend parity (spec §2, §10)
# --------------------------------------------------------------------------- #
class TestBackendParity:
    def test_numpy_numba_ou_identical(self):
        """numpy vs numba OU: identical results for a fixed seed."""
        if "numba" not in list_backends():
            pytest.skip("numba backend not available")
        r_np = generate_ou(n_traj=20, fs=1e3, n_points=2**10,
                           gamma=5.0, sigma=1.0, seed=42, backend="numpy")
        r_nb = generate_ou(n_traj=20, fs=1e3, n_points=2**10,
                           gamma=5.0, sigma=1.0, seed=42, backend="numba")
        assert np.allclose(r_np.traj, r_nb.traj, atol=ATOL_FP), (
            "numpy and numba OU backends disagree")

    def test_numpy_numba_ou_sum_identical(self):
        """numpy vs numba OU-sum: identical results for a fixed seed."""
        if "numba" not in list_backends():
            pytest.skip("numba backend not available")
        r_np = generate_ou_sum(n_traj=20, fs=1e3, n_points=2**10,
                               alpha=1.0, S0=1.0, f0=1.0, seed=42, backend="numpy")
        r_nb = generate_ou_sum(n_traj=20, fs=1e3, n_points=2**10,
                               alpha=1.0, S0=1.0, f0=1.0, seed=42, backend="numba")
        assert np.allclose(r_np.traj, r_nb.traj, atol=ATOL_FP), (
            "numpy and numba OU-sum backends disagree")


# --------------------------------------------------------------------------- #
# Reproducibility (spec §2)
# --------------------------------------------------------------------------- #
class TestReproducibility:
    def test_same_seed_same_result(self):
        r1 = generate("1/f", n_traj=10, fs=1e3, n_points=2**10, seed=123)
        r2 = generate("1/f", n_traj=10, fs=1e3, n_points=2**10, seed=123)
        assert np.array_equal(r1.traj, r2.traj)

    def test_different_seed_different_result(self):
        r1 = generate("1/f", n_traj=10, fs=1e3, n_points=2**10, seed=1)
        r2 = generate("1/f", n_traj=10, fs=1e3, n_points=2**10, seed=2)
        assert not np.array_equal(r1.traj, r2.traj)

    def test_spec_records_seed_and_process(self):
        res = generate("ou", n_traj=4, fs=1e3, n_points=2**8, seed=7)
        assert res.seed == 7
        assert res.spec["process"] == "ou"


# --------------------------------------------------------------------------- #
# Compose (spec §6.5)
# --------------------------------------------------------------------------- #
class TestCompose:
    def test_compose_sums(self):
        r1 = generate("1/f", n_traj=5, fs=1e3, n_points=2**8, seed=0)
        r2 = generate("1/f", n_traj=5, fs=1e3, n_points=2**8, seed=1)
        c = compose([r1, r2], weights=[1.0, 1.0], seed=0)
        assert np.allclose(c.traj, r1.traj + r2.traj)

    def test_compose_drift(self):
        r = generate("1/f", n_traj=5, fs=1e3, n_points=2**8, seed=0)
        c = compose([r], drift_rate=0.5, seed=0)
        # drift adds 0.5 * t to every trajectory
        expected = r.traj + 0.5 * r.t[None, :]
        assert np.allclose(c.traj, expected)

    def test_compose_quasistatic(self):
        r = generate("1/f", n_traj=50, fs=1e3, n_points=2**8, seed=0)
        c = compose([r], quasistatic_std=1.0, seed=0)
        # the per-shot offset is constant in time within each trajectory
        diff = c.traj - r.traj
        assert np.allclose(diff[:, 0], diff[:, -1])
        # std of the offset ~ 1.0
        assert abs(np.std(diff[:, 0]) - 1.0) < RTOL_STAT


# --------------------------------------------------------------------------- #
# NoiseResult container (spec §4)
# --------------------------------------------------------------------------- #
class TestNoiseResult:
    def test_save_load_roundtrip(self, tmp_path):
        res = generate("1/f", n_traj=3, fs=1e3, n_points=2**8, seed=0)
        p = tmp_path / "test.npz"
        res.save(str(p))
        loaded = NoiseResult.load(str(p))
        assert np.array_equal(loaded.traj, res.traj)
        assert np.array_equal(loaded.t, res.t)
        assert loaded.fs == res.fs
        assert loaded.units == res.units

    def test_shape_properties(self):
        res = generate("1/f", n_traj=7, fs=1e3, n_points=2**8, seed=0)
        assert res.n_traj == 7
        assert res.n_time == 2**8
        assert res.n_sites == 0
        assert res.is_uniform