"""E003_phase_velocity: Unwrapped spectral phase derivatives.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
The compute path combines at least three missing APIs with a supported anchor.
Named intermediate outputs expose indexing, shape and numerical behavior.
"""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(503)
    t = np.arange(96, dtype=np.float32) / 96
    phase = 2 * np.pi * (3 * t + 7 * t * t)
    x = np.sin(phase)
    x += 0.01 * rng.standard_normal(96, dtype=np.float32)
    return {
        'x': x,
    }


def forward(x):
    analytic = np.fft.fft(x)
    shifted = np.fft.fftshift(analytic)
    angle = np.angle(shifted)
    phase = np.unwrap(angle)
    velocity = np.gradient(phase)
    acceleration = np.diff(velocity)
    absolute_1 = np.absolute(acceleration)
    threshold = np.percentile(absolute_1, 80)
    events = np.flatnonzero(np.absolute(acceleration) > threshold)
    samples = np.take(acceleration, events)
    slope_quantiles = np.quantile(velocity, [0.1, 0.5, 0.9])
    ranked_events = np.argsort(samples)
    return {
        "slope_quantiles": slope_quantiles,
        "ranked_events": ranked_events,
        'phase': phase,
        'velocity': velocity,
        'events': events,
        'samples': samples,
    }
