"""C005_distance_refinement: Iterative pointwise residual contraction.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Source: ../../../../examples/custom_operator_euclidean_distance.py
Adaptation: bounded inputs, named outputs, explicit computational stages.
"""
# Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
# SPDX-License-Identifier: Apache-2.0
import numpy as np


def build_inputs():
    rng = np.random.default_rng(305)
    return {
        "x": rng.standard_normal((16384, 256), dtype=np.float32),
        "target": rng.standard_normal((16384, 256), dtype=np.float32),
        "weight": np.linspace(0.5, 1.0, 256, dtype=np.float32),
    }


def distance(x, y):
    delta = np.subtract(x, y)
    squared = np.square(delta)
    return np.sqrt(squared)


def contract(current, target, rate):
    error = np.subtract(current, target)
    correction = np.multiply(error, rate)
    updated = np.subtract(current, correction)
    return updated


def weighted_rows(error, weight):
    weighted = np.multiply(error, weight)
    total = np.sum(weighted, axis=1)
    return total


def forward(x, target, weight):
    original = distance(x, target)
    current = x
    for iteration in range(5):
        if iteration < 3:
            rate = 0.2
        else:
            rate = 0.1
        current = contract(current, target, rate)
    final = distance(current, target)
    weighted_before = weighted_rows(original, weight)
    weighted_after = weighted_rows(final, weight)
    improvement = np.subtract(weighted_before, weighted_after)
    return {
        'out': current,
        'distance': final,
        'improvement': improvement,
    }
