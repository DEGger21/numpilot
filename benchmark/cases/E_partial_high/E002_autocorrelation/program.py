"""E002_autocorrelation: FFT autocorrelation and lag selection.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
The compute path combines at least three missing APIs with a supported anchor.
Named intermediate outputs expose indexing, shape and numerical behavior.
"""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(502)
    x = rng.standard_normal(96, dtype=np.float32) * 0.1
    pulse = np.zeros(12, dtype=np.float32)
    pulse[2:5] = [0.5, 1.0, 0.5]
    x += np.tile(pulse, 8)
    return {
        'x': x,
    }


def forward(x):
    mean_1 = np.mean(x)
    centered = np.subtract(x, mean_1)
    padded = np.pad(centered, (0, x.size))
    spectrum = np.fft.fft(padded)
    conjugate = np.conjugate(spectrum)
    correlation = np.fft.ifft(spectrum * conjugate).real
    lags = np.arange(x.size)
    positive = np.take(correlation, lags)
    normalized = positive / positive[0]
    peaks = np.argpartition(normalized[1:], -4)[-4:] + 1
    sorted_peaks = np.sort(peaks)
    envelope_quantiles = np.quantile(normalized, [0.25, 0.5, 0.75])
    zero_crossings = np.flatnonzero(np.diff(normalized > 0))
    return {
        "envelope_quantiles": envelope_quantiles,
        "zero_crossings": zero_crossings,
        'correlation': normalized,
        'peaks': sorted_peaks,
        'lags': lags,
    }
