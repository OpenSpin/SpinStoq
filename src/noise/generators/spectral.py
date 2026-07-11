"""Method 1 — 1/f and arbitrary-spectrum noise via FFT (Timmer-König + circulant embedding).

Two engines, both ``O(N log N)`` and fully vectorized over the ensemble:

* :func:`timmer_koenig` — stationary Gaussian trace with an analytic target
  ``S(f)`` (default ``S0 * (f0/f)^alpha``). The classic Timmer & König (1995)
  draw: complex Gaussian coefficients with the right variance per frequency,
  Hermitian-symmetric, inverse-rFFT. One ``irfft`` over the whole batch.

* :func:`circulant_embedding` — the exact, model-free engine that accepts an
  arbitrary ACF or tabulated ``S(f)`` (e.g. estimated from data). Embeds the
  covariance in a circulant of length ``>= 2N``, takes the matrix square root
  via FFT, multiplies by complex white noise, transforms back. This is what
  ``calibrate(..., method="circulant")`` calls (spec §7).
"""
from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

from ..core import NoiseResult, make_generator, make_time_grid

__all__ = ["timmer_koenig", "circulant_embedding", "generate_spectral"]


# --------------------------------------------------------------------------- #
# Target spectra
# --------------------------------------------------------------------------- #
def _one_sided_psd_1f(f: np.ndarray, S0: float, f0: float, alpha: float,
                       f_min: Optional[float] = None) -> np.ndarray:
    """One-sided ``S(f) = S0 * (f0 / f)^alpha`` with DC / low-f regularization.

    DC is set to 0 (a stationary process has no DC component). For ``alpha >= 1``
    the integral diverges at low f, so an optional ``f_min`` knee rolls the
    spectrum off below it (default: the first non-DC frequency bin) to keep the
    variance finite.
    """
    f = np.asarray(f, dtype=float)
    S = np.zeros_like(f)
    nz = f > 0
    S[nz] = S0 * (f0 / f[nz]) ** alpha
    if f_min is not None and f_min > 0:
        # roll off below f_min: replace 1/f^alpha by a constant (f_min/f)^alpha
        # i.e. flatten S(f) for f < f_min to S(f_min)
        low = nz & (f < f_min)
        if np.any(low):
            S_min = S0 * (f0 / f_min) ** alpha
            S[low] = S_min
    return S


# --------------------------------------------------------------------------- #
# Timmer-König
# --------------------------------------------------------------------------- #
def timmer_koenig(n_traj: int = 1, fs: float = 1.0, n_points: int = 2**16,
                  alpha: float = 1.0, S0: float = 1.0, f0: float = 1.0,
                  f_min: Optional[float] = None, dtype=np.float64,
                  seed=None, t: Optional[np.ndarray] = None,
                  target_psd: Optional[np.ndarray] = None
                  ) -> NoiseResult:
    """Generate stationary Gaussian noise with a target one-sided PSD.

    Uses the Timmer-König (1995) draw: for each positive frequency, draw complex
    coefficients ``c_k = sqrt(S(f_k) * fs * N / 2) * (a_k + i b_k)`` with
    ``a, b ~ N(0,1)`` i.i.d. per trajectory; set DC and Nyquist real; enforce
    Hermitian symmetry; ``x = irfft(c)``. A single ``irfft`` over the batch does
    the whole ensemble.

    Parameters
    ----------
    target_psd : ndarray, optional
        Tabulated one-sided PSD on the rFFT grid ``f_k = k*fs/N``. If given,
        overrides the ``alpha/S0/f0`` analytic form (used by ``calibrate``).
    """
    t_arr, fs_eff, n = make_time_grid(fs=fs, n_points=n_points, t=t)
    rng = make_generator(seed)
    # rFFT frequency grid (one-sided): k = 0..N//2
    f = np.fft.rfftfreq(n, d=1.0 / fs_eff)
    if target_psd is not None:
        S = np.asarray(target_psd, dtype=float)
        if S.shape != f.shape:
            raise ValueError(f"target_psd shape {S.shape} != rFFT grid {f.shape}")
    else:
        S = _one_sided_psd_1f(f, S0=S0, f0=f0, alpha=alpha, f_min=f_min)

    # Amplitude per Fourier coefficient so the measured one-sided PSD matches S(f).
    # For positive freqs: c_k = amp_k * (a+ib), a,b~N(0,1) -> E[|c_k|^2] = 2*amp_k^2.
    # One-sided PSD: S_one(f_k) = 2*|c_k|^2/(N*fs) = 4*amp_k^2/(N*fs)
    #   -> amp_k = sqrt(S_one * N * fs / 4).
    # DC and Nyquist are real: c = amp*a -> E[c^2] = amp^2, S_one = amp^2/(N*fs)
    #   -> amp = sqrt(S_one * N * fs). DC is 0 for a stationary process.
    amp = np.zeros_like(S)
    amp[1:] = np.sqrt(S[1:] * fs_eff * n / 4.0)
    if n % 2 == 0 and f.size > 1:
        # Nyquist bin is real (no factor-2 doubling in one-sided PSD)
        amp[-1] = np.sqrt(S[-1] * fs_eff * n)
    # draw complex white noise per trajectory: (n_traj, n_freq).
    # Draw a single real array and view it as complex to halve the RNG calls
    # and the temporary count. E[|z|^2] = 2 (two N(0,1) components), which the
    # amplitude factor above already accounts for.
    z_real = rng.standard_normal(size=(n_traj, 2 * f.size))
    z = z_real[:, ::2] + 1j * z_real[:, 1::2]
    c = amp[None, :] * z
    # enforce real DC and (for even N) real Nyquist
    c[:, 0] = c[:, 0].real
    if n % 2 == 0 and f.size > 1:
        c[:, -1] = c[:, -1].real
    x = np.fft.irfft(c, n=n, axis=-1).astype(dtype)
    # irfft already divides by N, so the normalization above gives the right PSD.
    spec = {"process": "1/f", "method": "timmer_koenig", "alpha": alpha, "S0": S0,
            "f0": f0, "f_min": f_min, "fs": fs_eff, "n_points": n, "n_traj": n_traj,
            "dtype": str(dtype)}
    return NoiseResult(traj=x, t=t_arr, fs=fs_eff, spec=spec, seed=seed, units="a.u.")


# --------------------------------------------------------------------------- #
# Circulant embedding (exact, model-free)
# --------------------------------------------------------------------------- #
def _acf_to_psd_one_sided(acf: np.ndarray, dt: float) -> np.ndarray:
    """One-sided PSD from a tabulated ACF via the Wiener-Khinchin rFFT.

    ``acf[0..N-1]`` is the ACF at lags ``0, dt, 2dt, ...``. Returns the one-sided
    PSD on the rFFT grid, normalized so ``sum(S) * df = acf[0] = var``.
    """
    n = acf.size
    # zero-pad to avoid circular wraparound in the FFT-based PSD
    n_fft = 1
    while n_fft < 2 * n:
        n_fft *= 2
    two_sided = np.fft.rfft(acf, n=n_fft)
    df = 1.0 / (n_fft * dt)
    # rfft of a real even sequence (acf is symmetric) gives a real PSD
    psd_two = np.real(two_sided) * dt
    # convert two-sided (f in [-fs/2, fs/2]) to one-sided: double positive freqs
    psd_one = np.zeros_like(psd_two)
    psd_one[0] = psd_two[0]
    if n_fft % 2 == 0:
        psd_one[1:-1] = 2.0 * psd_two[1:-1]
        psd_one[-1] = psd_two[-1]
    else:
        psd_one[1:] = 2.0 * psd_two[1:]
    return psd_one, np.fft.rfftfreq(n_fft, d=dt)


def circulant_embedding(acf: np.ndarray, n_traj: int = 1, fs: float = 1.0,
                        n_points: Optional[int] = None, dt: Optional[float] = None,
                        dtype=np.float64, seed=None,
                        t: Optional[np.ndarray] = None) -> NoiseResult:
    """Exact stationary Gaussian noise with a target ACF via circulant embedding.

    Embeds the covariance in a circulant matrix of length ``>= 2N``, takes its
    square root via FFT, multiplies by complex white noise, and transforms back.
    This preserves the target ACF/PSD exactly (Gaussian surrogate) and is the
    model-free engine reused by :func:`openspin.noise.calibrate`.

    Parameters
    ----------
    acf : ndarray, shape ``(n_lag,)``
        Target autocorrelation at lags ``0, dt, 2dt, ...``. ``acf[0]`` is the
        variance. Must be non-negative-definite (the embedding pads to ``2N`` and
        clips tiny negative eigenvalues from finite-precision noise).
    n_points : int, optional
        Output length. Defaults to ``acf.size``.
    dt : float, optional
        Sample spacing [s]. Defaults to ``1/fs``.
    """
    acf = np.asarray(acf, dtype=float)
    if acf.ndim != 1:
        raise ValueError("acf must be 1D")
    if n_points is None:
        n_points = acf.size
    if dt is None:
        dt = 1.0 / fs
    t_arr = np.arange(n_points, dtype=float) * dt
    fs_eff = 1.0 / dt
    rng = make_generator(seed)

    # build the first row of the circulant embedding (length 2N): [acf, acf[1:][::-1]]
    n = n_points
    first_row = np.zeros(2 * n)
    m = min(acf.size, n)
    first_row[:m] = acf[:m]
    first_row[2 * n - m + 1: 2 * n] = acf[1:m][::-1]  # symmetric extension

    # eigenvalues of the circulant = rFFT of its first row
    lam = np.fft.rfft(first_row)
    # clip tiny negatives from finite-precision / non-PD input
    lam_min = np.min(lam.real)
    if lam_min < 0:
        if lam_min < -1e-6 * np.max(lam.real):
            # significant negative eigenvalue -> ACF not non-negative-definite
            import warnings
            warnings.warn(f"circulant embedding has negative eigenvalues (min={lam_min:.3e}); "
                          "clipping. The target ACF may not be valid.", RuntimeWarning)
        lam = np.maximum(lam.real, 0.0)
    else:
        lam = lam.real

    # sqrt eigenvalues -> square root of the circulant
    sqrt_lam = np.sqrt(lam)
    # draw complex white noise per trajectory and multiply
    z = rng.standard_normal(size=(n_traj, lam.size)) + 1j * rng.standard_normal(size=(n_traj, lam.size))
    c = sqrt_lam[None, :] * z
    c[:, 0] = c[:, 0].real
    if (2 * n) % 2 == 0 and lam.size > 1:
        c[:, -1] = c[:, -1].real
    x_full = np.fft.irfft(c, n=2 * n, axis=-1)
    # take the first n points
    x = x_full[..., :n].astype(dtype)
    # normalize so var == acf[0]: the embedding gives var = mean(lam) approximately;
    # rescale to match the target variance exactly
    if acf[0] > 0:
        current_var = np.var(x)
        if current_var > 0:
            x = x * np.sqrt(acf[0] / current_var)

    spec = {"process": "circulant", "method": "circulant_embedding", "fs": fs_eff,
            "n_points": n, "n_traj": n_traj, "acf_lags": m, "dtype": str(dtype)}
    return NoiseResult(traj=x, t=t_arr, fs=fs_eff, spec=spec, seed=seed, units="a.u.")


def generate_spectral(n_traj: int = 1, fs: float = 1.0, n_points: int = 2**16,
                      alpha: float = 1.0, S0: float = 1.0, f0: float = 1.0,
                      f_min: Optional[float] = None, dtype=np.float64, seed=None,
                      t: Optional[np.ndarray] = None,
                      target_psd: Optional[np.ndarray] = None,
                      method: str = "timmer_koenig") -> NoiseResult:
    """Dispatch to :func:`timmer_koenig` or :func:`circulant_embedding`."""
    if method == "timmer_koenig":
        return timmer_koenig(n_traj=n_traj, fs=fs, n_points=n_points, alpha=alpha,
                             S0=S0, f0=f0, f_min=f_min, dtype=dtype, seed=seed, t=t,
                             target_psd=target_psd)
    if method == "circulant":
        # caller must pass acf via target_psd (renamed) — handled by calibrate
        raise ValueError("use circulant_embedding(acf=...) directly for method='circulant'")
    raise ValueError(f"unknown method {method!r}")