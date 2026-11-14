"""C002_mlp_inference: Two layer classifier with stable probabilities.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Source: ../../../../examples/mlp_forward.py
Adaptation: bounded inputs, named outputs, explicit computational stages.
"""
# Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
# SPDX-License-Identifier: Apache-2.0
import numpy as np


def build_inputs():
    rng = np.random.default_rng(302)
    return {
        'x': rng.standard_normal((9, 12), dtype=np.float32),
        'w1': rng.standard_normal((12, 8), dtype=np.float32) * 0.1,
        'b1': np.linspace(-0.1, 0.1, 8, dtype=np.float32),
        'w2': rng.standard_normal((8, 4), dtype=np.float32) * 0.2,
        'b2': np.zeros(4, dtype=np.float32),
    }


def dense(x, weights, bias):
    projection = np.matmul(x, weights)
    return np.add(projection, bias)


def shifted_exponential(logits):
    peak = np.max(logits, axis=1, keepdims=True)
    shifted = np.subtract(logits, peak)
    return np.exp(shifted)


def stable_softmax(logits):
    numerator = shifted_exponential(logits)
    denominator = np.sum(numerator, axis=1, keepdims=True)
    probabilities = np.divide(numerator, denominator)
    return probabilities


def forward(x, w1, b1, w2, b2):
    hidden = dense(x, w1, b1)
    for iteration in range(2):
        if iteration == 0:
            hidden = np.maximum(hidden, 0.0)
        else:
            hidden = np.minimum(hidden, 6.0)
    logits = dense(hidden, w2, b2)
    probabilities = stable_softmax(logits)
    confidence = np.max(probabilities, axis=1)
    mass = np.sum(probabilities, axis=1)
    return {
        'probabilities': probabilities,
        'confidence': confidence,
        'mass': mass,
    }
