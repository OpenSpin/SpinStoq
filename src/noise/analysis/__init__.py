"""Analysis quantities for ``openspin.noise`` (NumPy, FFT-based, vectorized).

All estimators are ensemble-averaged over the leading trajectory axis and share
the one-sided PSD convention (spec §5): ``var(x) = integral_0^{fs/2} S(f) df``.
"""
from .psd import welch, periodogram, multitaper, psd
from .acf import autocorrelation, acf
from .allan import allan_variance, allan_deviation, allan
from .xcorr import cross_correlation, xcorr
from .xspec import cross_spectrum, coherence, cross_spectral_density

__all__ = [
    "welch", "periodogram", "multitaper", "psd",
    "autocorrelation", "acf",
    "allan_variance", "allan_deviation", "allan",
    "cross_correlation", "xcorr",
    "cross_spectrum", "coherence", "cross_spectral_density",
]