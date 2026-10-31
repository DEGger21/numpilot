"""K001_signal_workbench: Multi-stage signal conditioning and spectral diagnostics.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
All compute API names are outside the pinned snapshot, excluding output boxing.
Every stage contributes named outputs; at least 30 static gap call sites.
"""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(1101)
    t = np.linspace(0, 4, 257, dtype=np.float64)
    signal = np.sin(7 * t) + 0.3 * np.cos(19 * t)
    signal += rng.normal(0, 0.03, t.size)
    return {
        'x': signal,
        'kernel': np.array([1, 2, 3, 2, 1], dtype=np.float64) / 9,
    }


def condition(x, kernel):
    """Build a reflected FIR halo and finite-difference diagnostics."""
    padded = np.pad(x, (2, 2), mode="reflect")
    smooth = np.convolve(padded, kernel, mode="valid")
    slope = np.gradient(smooth)
    curvature = np.diff(slope)
    left = np.roll(smooth, 1)
    right = np.roll(smooth, -1)
    filtered = np.median(np.stack((left, smooth, right)), axis=0)
    return filtered, slope, curvature, padded


def spectral(x):
    """Retain complex, magnitude and unwrapped phase observables."""
    window = np.hanning(x.size)
    spectrum = np.fft.rfft(x * window)
    power = spectrum.real ** 2 + spectrum.imag ** 2
    phase = np.unwrap(np.angle(spectrum))
    phase_rate = np.gradient(phase)
    return spectrum, power, phase, phase_rate


def distribution(x):
    """Expose edges, insertion indices and weighted bin sums."""
    histogram, edges = np.histogram(x, bins=17)
    bins = np.digitize(x, edges[1:-1])
    counts = np.bincount(bins, minlength=17)
    weighted = np.bincount(bins, weights=x, minlength=17)
    lookup = np.searchsorted(edges, x, side="right")
    return histogram, edges, bins, counts, weighted, lookup


def order_statistics(power):
    """Select the five strongest bins without a full supported sort."""
    indices = np.argpartition(power, -5)[-5:]
    values = np.take(power, indices)
    order = np.argsort(values)
    peaks = np.take(indices, order)
    ranked = np.take(values, order)
    median = np.asarray(np.median(power))
    quantiles = np.quantile(power, [0.1, 0.5, 0.9])
    return peaks, ranked, median, quantiles


def resample(x):
    """Interpolate to a new grid and retain integral and derivatives."""
    knots = np.arange(x.size)
    nodes = np.arange(0, x.size - 1, 0.75)
    resampled = np.interp(nodes, knots, x)
    derivative = np.gradient(resampled, nodes)
    integral = np.asarray(np.trapezoid(resampled, nodes))
    return resampled, derivative, integral, nodes


def correlate(x):
    """Positive-lag correlation and a small top-k lag set."""
    correlation = np.correlate(x, x, mode="full")[x.size - 1:]
    normalized = correlation / correlation[0]
    peak_indices = np.argpartition(normalized[1:], -4)[-4:] + 1
    peak_values = np.take(normalized, peak_indices)
    return normalized, peak_indices, peak_values


def forward(x, kernel):
    smooth, slope, curvature, padded = condition(x, kernel)
    spectrum, power, phase, phase_rate = spectral(smooth)
    histogram, edges, bins, counts, weighted, lookup = distribution(smooth)
    peaks, peak_values, median, quantiles = order_statistics(power)
    resampled, derivative, integral, nodes = resample(smooth)
    autocorrelation, lag_peaks, lag_values = correlate(smooth)
    joined = np.concatenate((slope, curvature))
    unique_bins = np.unique(bins)
    cross = np.correlate(resampled, kernel, mode="valid")
    return {
        'smooth': smooth,
        'slope': slope,
        'curvature': curvature,
        'spectrum': spectrum,
        'power': power,
        'phase': phase,
        'phase_rate': phase_rate,
        'histogram': histogram,
        'edges': edges,
        'counts': counts,
        'weighted_histogram': weighted,
        'edge_insertions': lookup,
        'peaks': peaks,
        'peak_values': peak_values,
        'median': median,
        'quantiles': quantiles,
        'resampled': resampled,
        'derivative': derivative,
        'integral': integral,
        'nodes': nodes,
        'autocorrelation': autocorrelation,
        'lag_peaks': lag_peaks,
        'lag_values': lag_values,
        'joined': joined,
        'unique_bins': unique_bins,
        'cross': cross,
    }
