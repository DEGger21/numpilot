"""C004_cosine_rows: Cosine similarities with zero-row safeguards.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Source: ../../../../examples/custom_operator_cosine_similarity.py
Adaptation: bounded inputs, named outputs, explicit computational stages.
"""
# Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
# SPDX-License-Identifier: Apache-2.0
import numpy as np


def build_inputs():
    rng = np.random.default_rng(304)
    x = rng.standard_normal((12, 7), dtype=np.float32)
    y = rng.standard_normal((12, 7), dtype=np.float32)
    x[0] = 0
    y[1] = 0
    return {
        'x': x,
        'y': y,
    }


def row_norm(x):
    squared = np.square(x)
    power = np.sum(squared, axis=1)
    return np.sqrt(power)


def safe_denominator(nx, ny):
    product = np.multiply(nx, ny)
    return np.maximum(product, 1e-6)


def similarity(x, y, nx, ny):
    products = np.multiply(x, y)
    numerator = np.sum(products, axis=1)
    denominator = safe_denominator(nx, ny)
    return np.divide(numerator, denominator)


def forward(x, y):
    nx = row_norm(x)
    ny = row_norm(y)
    cosine = similarity(x, y, nx, ny)
    distance = np.subtract(1.0, cosine)
    score = distance
    iteration = 0
    while iteration < 3:
        if iteration == 0:
            score = np.clip(score, 0.0, 2.0)
        else:
            score = np.sqrt(score)
        iteration += 1
    return {
        'cosine': cosine,
        'distance': distance,
        'score': score,
    }
