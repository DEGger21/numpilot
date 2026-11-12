"""C003_layer_normalization: Shared row moments feed normalization and diagnostics.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Source: ../../../../examples/custom_operator_layernorm.py
Adaptation: bounded inputs, named outputs, explicit computational stages.
"""
# Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
# SPDX-License-Identifier: Apache-2.0
import numpy as np


def build_inputs():
    rng = np.random.default_rng(303)
    x = rng.standard_normal((16384, 256), dtype=np.float32)
    x[0] = 4.0
    return {
        "x": x,
        "gain": np.linspace(0.5, 1.5, 256, dtype=np.float32),
        "bias": np.linspace(-0.2, 0.2, 256, dtype=np.float32),
    }


def moments(x):
    mean = np.mean(x, axis=1, keepdims=True)
    centered = np.subtract(x, mean)
    squared = np.multiply(centered, centered)
    variance = np.mean(squared, axis=1, keepdims=True)
    return centered, variance


def normalize(centered, variance):
    regularized = np.add(variance, 1e-5)
    scale = np.sqrt(regularized)
    return np.divide(centered, scale)


def affine(normalized, gain, bias):
    scaled = np.multiply(normalized, gain)
    return np.add(scaled, bias)


def forward(x, gain, bias):
    centered, variance = moments(x)
    normalized = normalize(centered, variance)
    output = affine(normalized, gain, bias)
    for iteration in range(2):
        if iteration == 0:
            output = np.clip(output, -3.0, 3.0)
        else:
            output = np.tanh(output)
    row_mean = np.mean(normalized, axis=1)
    row_power = np.mean(np.square(normalized), axis=1)
    return {
        'out': output,
        'mean': row_mean,
        'power': row_power,
    }
