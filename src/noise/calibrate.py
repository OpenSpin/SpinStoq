"""From a measured trace: estimate + resynthesize (model-free, spec §7).

``calibrate(trace, fs, method="circulant")`` returns a :class:`SurrogateGenerator`:

1. **Estimate** the target second-order statistics from the trace: one-sided PSD
   (Welch, :mod:`spinspectro.characterize.psd`) and/or ACF (Wiener-Khinchin,
   :mod:`spinspectro.characterize.acf`). Handle detrending, windowing, and
   (optionally) a robust log-log fit only for *reporting* ``alpha`` — generation
   stays model-free.
2. **Resynthesize** surrogate trajectories that reproduce those statistics via
   **circulant embedding** (§6.1 general path). This preserves the measured
   spectrum/ACF exactly (Gaussian surrogate) without committing to a parametric
   model.
3. Multi-channel input -> estimate the **cross-spectral matrix** and resynthesize
   jointly (per-frequency Cholesky of the CSD matrix), preserving inter-channel
   coherence.

``gen.sample(n_traj, n_points, seed)`` then emits as many trajectories as wanted.

Parametric fitting to a named model is explicitly out of scope (TODO/hook only).
"""
from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

from .core import NoiseResult, make_generator, make_time_grid
from spinspectro.characterize import welch, autocorrelation, cross_spectrum
from .generators.spectral import circulant_embedding

__all__ = ["calibrate", "SurrogateGenerator"]


def _estimate_alpha(f: np.ndarray, S: np.ndarray, f_min: Optional[float] = None,
                    f_max: Optional[float] = None) -> float:
    """Robust log-log slope of ``S(f)`` vs ``f`` for *reporting* only."""
    mask = (f > 0) & (S > 0)
    if f_min is not None:
        mask &= (f >= f_min)
    if f_max is not None:
        mask &= (f <= f_max)
    if np.sum(mask) < 4:
        return float("nan")
    logf = np.log10(f[mask])
    logS = np.log10(S[mask])
    # ordinary least squares slope
    A = np.vstack([np.ones_like(logf), logf]).T
    coef, *_ = np.linalg.lstsq(A, logS, rcond=None)
    return float(-coef[1])  # alpha = -slope


class SurrogateGenerator:
    """Model-free surrogate generator estimated from a measured trace.

    Call :meth:`sample` to emit new trajectories that reproduce the measured
    second-order statistics (PSD/ACF, and cross-spectra for multi-channel input).
    """

    def __init__(self, trace: np.ndarray, fs: float, method: str = "circulant",
                 nperseg: Optional[int] = None, detrend: str = "mean"):
        trace = np.asarray(trace, dtype=float)
        self.fs = float(fs)
        self.method = method
        self.nperseg = nperseg
        self.detrend = detrend
        self.is_multichannel = trace.ndim >= 2 and trace.shape[0] > 1 if trace.ndim == 2 else trace.ndim == 3

        if trace.ndim == 1:
            self._fit_single(trace)
        elif trace.ndim == 2:
            # (n_channels, n_time): route single-channel to _fit_single
            if trace.shape[0] == 1:
                self._fit_single(trace[0])
            else:
                self._fit_multichannel(trace)
        elif trace.ndim == 3:
            # (n_traj, n_channels, n_time) — average the estimate over trajectories
            n_traj, n_ch, n_time = trace.shape
            if n_ch == 1:
                self._fit_single(trace[:, 0, :].reshape(n_traj, n_time))
            else:
                self._fit_multichannel(trace.mean(axis=0))
        else:
            raise ValueError(f"trace ndim must be 1, 2, or 3; got {trace.ndim}")

    # ---- single-channel fit ----
    def _fit_single(self, trace: np.ndarray):
        trace = np.atleast_2d(np.asarray(trace, dtype=float))
        f, S = welch(trace, fs=self.fs, nperseg=self.nperseg, detrend=self.detrend)
        self.f_psd = f
        self.psd = S
        self.alpha = _estimate_alpha(f, S)
        # ACF for circulant embedding: estimate directly from the trace via FFT
        # (Wiener-Khinchin). This avoids the PSD->ACF interpolation/normalization
        # pitfalls and gives a valid non-negative-definite ACF by construction.
        n_time = trace.shape[-1]
        tau, acf = autocorrelation(trace, fs=self.fs, biased=True)
        self.acf = acf
        self.n_channels = 1

    # ---- multi-channel fit ----
    def _fit_multichannel(self, trace: np.ndarray):
        """Estimate per-channel PSDs and the cross-spectral matrix."""
        trace = np.asarray(trace, dtype=float)
        if trace.ndim == 1:
            trace = trace[np.newaxis, :]
        n_ch, n_time = trace.shape
        self.n_channels = n_ch
        # per-channel PSD
        f, S0 = welch(trace[0:1], fs=self.fs, nperseg=self.nperseg, detrend=self.detrend)
        psds = np.empty((n_ch, f.size), dtype=float)
        psds[0] = S0
        for c in range(1, n_ch):
            _, psds[c] = welch(trace[c:c+1], fs=self.fs, nperseg=self.nperseg, detrend=self.detrend)
        self.f_psd = f
        self.psd = psds.mean(axis=0)  # average for reporting
        self.psds = psds
        self.alpha = _estimate_alpha(f, self.psd)
        # cross-spectral matrix on the Welch grid: (n_ch, n_ch, n_freq)
        csd = np.zeros((n_ch, n_ch, f.size), dtype=complex)
        for i in range(n_ch):
            for j in range(i, n_ch):
                _, Pij = cross_spectrum(trace[i:i+1], trace[j:j+1], fs=self.fs,
                                        nperseg=self.nperseg)
                csd[i, j] = Pij
                if i != j:
                    csd[j, i] = np.conj(Pij)
        self.csd = csd
        # per-frequency Cholesky factors for joint resynthesis
        # (clip non-PD matrices from finite-sample noise)
        self.csd_chol = np.empty_like(csd)
        for k in range(f.size):
            C = csd[:, :, k]
            C = 0.5 * (C + C.conj().T)
            try:
                self.csd_chol[:, :, k] = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # regularize
                ev = np.linalg.eigvalsh(C.real)
                C = C + (max(0.0, -ev.min()) + 1e-12 * ev.max()) * np.eye(n_ch)
                self.csd_chol[:, :, k] = np.linalg.cholesky(C)
        # ACF per channel (for the circulant path)
        self.acf = None

    # ---- sampling ----
    def sample(self, n_traj: int = 1, n_points: Optional[int] = None, seed=None,
               dtype=np.float64) -> NoiseResult:
        """Emit ``n_traj`` surrogate trajectories reproducing the measured stats."""
        if n_points is None:
            n_points = self.acf.size if self.acf is not None else len(self.f_psd)
        t_arr, fs_eff, n = make_time_grid(fs=self.fs, n_points=n_points)
        rng = make_generator(seed)

        if self.n_channels == 1:
            # single-channel circulant embedding using the estimated ACF
            res = circulant_embedding(self.acf, n_traj=n_traj, fs=self.fs,
                                      n_points=n, dtype=dtype, seed=rng, t=t_arr)
            spec = {"process": "surrogate", "method": "circulant", "source": "calibrate",
                    "alpha_reported": self.alpha, "fs": fs_eff, "n_points": n,
                    "n_traj": n_traj, "dtype": str(dtype)}
            res.spec = spec
            res.seed = seed
            return res

        # multi-channel joint resynthesis via per-frequency Cholesky of the CSD
        f = np.fft.rfftfreq(n, d=1.0 / self.fs)
        # interpolate the CSD Cholesky onto the output rFFT grid
        n_freq = f.size
        Lf = np.zeros((self.n_channels, self.n_channels, n_freq), dtype=complex)
        f_src = self.f_psd
        for i in range(self.n_channels):
            for j in range(self.n_channels):
                for comp in (0, 1):  # interpolate real & imag parts
                    Lf[i, j, :].real = np.interp(f, f_src, self.csd_chol[i, j, :].real,
                                                 left=self.csd_chol[i, j, 0].real,
                                                 right=self.csd_chol[i, j, -1].real)
                    Lf[i, j, :].imag = np.interp(f, f_src, self.csd_chol[i, j, :].imag,
                                                 left=self.csd_chol[i, j, 0].imag,
                                                 right=self.csd_chol[i, j, -1].imag)
        # draw independent complex white noise per channel per trajectory
        z = (rng.standard_normal(size=(n_traj, self.n_channels, n_freq)) +
             1j * rng.standard_normal(size=(n_traj, self.n_channels, n_freq)))
        # mix: c[t, :, k] = Lf[:, :, k] @ z[t, :, k]
        c = np.einsum("ijk,tjk->tik", Lf, z, optimize=True)
        c[:, :, 0] = c[:, :, 0].real
        if n % 2 == 0 and n_freq > 1:
            c[:, :, -1] = c[:, :, -1].real
        x = np.fft.irfft(c, n=n, axis=-1).astype(dtype)

        spec = {"process": "surrogate_multichannel", "method": "csd_cholesky",
                "source": "calibrate", "alpha_reported": self.alpha,
                "n_channels": self.n_channels, "fs": fs_eff, "n_points": n,
                "n_traj": n_traj, "dtype": str(dtype)}
        return NoiseResult(traj=x, t=t_arr, fs=fs_eff, spec=spec, seed=seed, units="a.u.")

    def __repr__(self) -> str:
        ch = f", n_channels={self.n_channels}" if self.n_channels > 1 else ""
        return (f"SurrogateGenerator(method={self.method!r}, fs={self.fs:.4g} Hz"
                f", alpha~{self.alpha:.2f}{ch})")


def calibrate(trace: np.ndarray, fs: float, method: str = "circulant",
              nperseg: Optional[int] = None, detrend: str = "mean") -> SurrogateGenerator:
    """Estimate second-order statistics from a measured trace and return a surrogate generator.

    Parameters
    ----------
    trace : ndarray, shape ``(n_time,)``, ``(n_channels, n_time)``, or ``(n_traj, n_channels, n_time)``
        Measured charge-sensor trace(s).
    fs : float
        Sample rate [Hz].
    method : str
        Reserved; currently only ``"circulant"`` (model-free) is implemented.
    nperseg : int, optional
        Welch segment length for the PSD estimate.
    detrend : str
        Detrend mode for the estimate (``"mean"`` or ``"linear"``).

    Returns
    -------
    gen : SurrogateGenerator
        Call ``gen.sample(n_traj, n_points, seed)`` to emit surrogate trajectories.

    Notes
    -----
    Parametric fitting to a named model is explicitly out of scope (TODO/hook only).
    """
    if method != "circulant":
        raise ValueError(f"only method='circulant' is implemented (got {method!r}); "
                         "parametric fitting is out of scope (TODO/hook).")
    return SurrogateGenerator(trace, fs=fs, method=method, nperseg=nperseg, detrend=detrend)