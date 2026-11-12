"""C011_residual_gemm: Residual matrix network with shared projection.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Source: ../../../../examples/big_gemm.py
Adaptation: bounded inputs, named outputs, explicit computational stages.
"""
# Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
# SPDX-License-Identifier: Apache-2.0
import numpy as np


def build_inputs():
    rng = np.random.default_rng(311)
    return {
        'x': rng.standard_normal((512, 512), dtype=np.float32),
        'w1': rng.standard_normal((512, 512), dtype=np.float32) * 0.1,
        'w2': rng.standard_normal((512, 512), dtype=np.float32) * 0.1,
        'w3': rng.standard_normal((512, 64), dtype=np.float32) * 0.1,
    }


def activation(x, weight):
    projected = np.matmul(x, weight)
    return np.maximum(projected, 0.0)


def residual_block(x, weight):
    branch = np.matmul(x, weight)
    activated = np.maximum(branch, 0.0)
    scaled = np.multiply(activated, 0.25)
    return np.add(x, scaled)


def row_summary(x):
    squared = np.square(x)
    mean_square = np.mean(squared, axis=1)
    rms = np.sqrt(mean_square)
    return rms


def forward(x, w1, w2, w3):
    hidden = activation(x, w1)
    for depth in range(3):
        shared = residual_block(hidden, w2)
        if depth < 2:
            hidden = np.maximum(shared, 0.0)
        else:
            hidden = np.tanh(shared)
    output = np.matmul(hidden, w3)
    summary = row_summary(hidden)
    row_logits = np.sum(output, axis=1)
    return {
        'out': output,
        'hidden': hidden,
        'summary': summary,
        'row_logits': row_logits,
    }
