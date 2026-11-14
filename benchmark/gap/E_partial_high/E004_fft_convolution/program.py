"""E004_fft_convolution: Spectral convolution compared with direct FIR.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
The compute path combines at least three missing APIs with a supported anchor.
Named intermediate outputs expose indexing, shape and numerical behavior.
"""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(504)
    x = rng.standard_normal(96, dtype=np.float32) * 0.1
    x[0] = 1.0
    x[47] = -2.0
    x[-1] = 0.5
    return {
        'x': x,
    }


def forward(x):
    kernel = np.hamming(9)
    kernel = kernel / np.sum(kernel)
    padded_x = np.pad(x, (0, 8))
    padded_k = np.pad(kernel, (0, x.size - 1))
    sx = np.fft.rfft(padded_x)
    sk = np.fft.rfft(padded_k)
    spectral = np.fft.irfft(sx * sk, n=padded_x.size)
    direct = np.convolve(x, kernel, mode='full')
    error = np.absolute(spectral - direct)
    quantiles = np.quantile(error, [0.5, 0.9, 1.0])
    peak_indices = np.argpartition(spectral, -3)[-3:]
    peak_values = np.take(spectral, peak_indices)
    return {
        "peak_indices": peak_indices,
        "peak_values": peak_values,
        'out': spectral,
        'direct': direct,
        'error_quantiles': quantiles,
    }
