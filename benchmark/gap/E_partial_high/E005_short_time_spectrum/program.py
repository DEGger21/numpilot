"""E005_short_time_spectrum: Overlapping short-time spectral frames.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
The compute path combines at least three missing APIs with a supported anchor.
Named intermediate outputs expose indexing, shape and numerical behavior.
"""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(505)
    t = np.arange(96, dtype=np.float32)
    x = np.sin(t * 0.3)
    x[48:] = np.cos(t[48:] * 1.1)
    x += 0.05 * rng.standard_normal(96, dtype=np.float32)
    return {
        'x': x,
    }


def forward(x):
    frames = np.lib.stride_tricks.sliding_window_view(x, 16)[::4]
    window = np.hanning(16)
    weighted = np.multiply(frames, window)
    spectrum = np.fft.rfft(weighted, axis=1)
    power = np.absolute(spectrum) ** 2
    peak = np.argmax(power, axis=1)
    selected = np.take_along_axis(power, peak[:, None], axis=1)
    differences = np.diff(peak)
    unique = np.unique(peak)
    band_quantiles = np.quantile(power, [0.25, 0.75], axis=1)
    change_positions = np.flatnonzero(differences)
    return {
        "band_quantiles": band_quantiles,
        "change_positions": change_positions,
        'power': power,
        'peak': peak,
        'selected': selected,
        'changes': differences,
        'unique_bins': unique,
    }
