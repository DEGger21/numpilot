"""E001_spectral_peaks: Windowed spectral peak extraction.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
The compute path combines at least three missing APIs with a supported anchor.
Named intermediate outputs expose indexing, shape and numerical behavior.
"""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(501)
    t = np.arange(96, dtype=np.float32) / 96
    x = rng.standard_normal(96, dtype=np.float32) * 0.02
    x += np.sin(2 * np.pi * 7 * t)
    x += 0.4 * np.cos(2 * np.pi * 19 * t)
    return {
        'x': x,
    }


def forward(x):
    window = np.hanning(x.size)
    weighted = np.multiply(x, window)
    spectrum = np.fft.rfft(weighted)
    frequency = np.fft.rfftfreq(x.size, d=0.01)
    power = np.absolute(spectrum) ** 2
    order = np.argsort(power)
    peaks = np.take(frequency, order[-5:])
    strength = np.take(power, order[-5:])
    phase = np.angle(spectrum)
    phase = np.unwrap(phase)
    peak_order = np.argsort(peaks)
    ordered_peaks = np.take(peaks, peak_order)
    return {
        "ordered_peaks": ordered_peaks,
        "peak_order": peak_order,
        'peaks': peaks,
        'strength': strength,
        'phase': phase,
        'power': power,
    }
