"""C012_calibrated_projection: Matrix inference with iterative calibration.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Source: ../../../../examples/ops_nn_matmul_inference.py
Adaptation: bounded inputs, named outputs, explicit computational stages.
"""
# Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
# SPDX-License-Identifier: Apache-2.0
import numpy as np


def build_inputs():
    rng = np.random.default_rng(312)
    return {
        'features': rng.standard_normal((8, 16), dtype=np.float32) * 0.01,
        'weights': rng.standard_normal((16, 4), dtype=np.float32) * 0.02,
        'target': np.full((8, 4), 0.25, dtype=np.float32),
    }


def project(features, weights):
    raw = np.matmul(features, weights)
    return np.clip(raw, -4.0, 4.0)


def normalize(logits):
    peak = np.max(logits, axis=1, keepdims=True)
    weights = np.exp(np.subtract(logits, peak))
    total = np.sum(weights, axis=1, keepdims=True)
    return np.divide(weights, total)


def correct(logits, residual, rate):
    bounded = np.clip(residual, -1.0, 1.0)
    adjustment = np.multiply(bounded, rate)
    result = np.subtract(logits, adjustment)
    return result


def forward(features, weights, target):
    logits = project(features, weights)
    current = logits
    counter = 0
    while counter < 4:
        probabilities = normalize(current)
        residual = np.subtract(probabilities, target)
        if counter < 2:
            current = correct(current, residual, 0.5)
        else:
            current = correct(current, residual, 0.1)
        counter += 1
    final = normalize(current)
    return {
        'logits': logits,
        'out': final,
        'residual': np.subtract(final, target),
    }
