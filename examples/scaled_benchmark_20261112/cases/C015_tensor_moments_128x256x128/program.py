"""C015_tensor_moments: Three-dimensional channel moment feedback.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
"""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(315)
    return {
        "x": rng.standard_normal((128, 256, 128), dtype=np.float32),
        "drive": rng.standard_normal((128, 1, 128), dtype=np.float32) * 0.1,
        "gain": np.linspace(0.8, 1.2, 128, dtype=np.float32),
    }


def channel_moments(x):
    mean = np.mean(x, axis=1, keepdims=True)
    centered = np.subtract(x, mean)
    variance = np.mean(np.square(centered), axis=1, keepdims=True)
    return mean, variance


def standardize(x, mean, variance):
    centered = np.subtract(x, mean)
    scale = np.sqrt(np.add(variance, 1e-4))
    return np.divide(centered, scale)


def drive_step(x, drive, gain):
    gained = np.multiply(x, gain)
    driven = np.add(gained, drive)
    return np.clip(driven, -2.0, 2.0)


def forward(x, drive, gain):
    initial_mean, initial_variance = channel_moments(x)
    current = x
    step = 0
    while step < 5:
        mean, variance = channel_moments(current)
        normalized = standardize(current, mean, variance)
        if step % 2 == 0:
            current = drive_step(normalized, drive, gain)
        else:
            current = np.tanh(normalized)
        step += 1
    final_mean, final_variance = channel_moments(current)
    return {
        'out': current,
        'mean_change': np.subtract(final_mean, initial_mean),
        'variance_change': np.subtract(final_variance, initial_variance),
    }
