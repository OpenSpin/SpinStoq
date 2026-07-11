"""Shared fixtures and tolerances for the openspin.noise test suite."""
import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(0)


# Statistical tolerances: generous enough for modest ensemble sizes but tight
# enough to catch normalization / convention bugs.
RTOL_STAT = 0.15   # ±15% for PSD/variance round-trips with n_traj ~ 200
RTOL_SLOPE = 0.25  # ±25% for the OU-sum 1/f slope (log-log fit noise)
ATOL_FP = 1e-9     # backend parity (numpy vs numba) for a fixed seed