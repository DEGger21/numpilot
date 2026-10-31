"""C006_softplus_network: Stable softplus and logistic gating.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Source: ../../../../examples/custom_operator_softplus.py
Adaptation: bounded inputs, named outputs, explicit computational stages.
"""
# Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
# SPDX-License-Identifier: Apache-2.0
import numpy as np


def build_inputs():
    x = np.linspace(-80, 80, 129, dtype=np.float32).reshape(3, 43)
    return {
        'x': x,
        'bias': np.linspace(-1, 1, 43, dtype=np.float32),
        'residual': np.full((3, 43), 0.1, dtype=np.float32),
    }


def softplus(x):
    magnitude = np.absolute(x)
    tail = np.exp(np.negative(magnitude))
    correction = np.log1p(tail)
    positive = np.maximum(x, 0.0)
    return np.add(correction, positive)


def sigmoid(x):
    bounded = np.clip(x, -40.0, 40.0)
    denominator = np.add(1.0, np.exp(np.negative(bounded)))
    return np.reciprocal(denominator)


def row_peak(x):
    return np.max(x, axis=1)


def forward(x, bias, residual):
    first = softplus(x)
    gate = sigmoid(x)
    hidden = np.multiply(first, gate)
    for iteration in range(3):
        shifted = np.add(hidden, bias)
        if iteration < 2:
            hidden = softplus(shifted)
        else:
            hidden = np.tanh(shifted)
        hidden = np.add(hidden, residual)
    activity = np.mean(hidden, axis=1)
    peak = row_peak(hidden)
    return {
        'out': hidden,
        'softplus': first,
        'gate': gate,
        'activity': activity,
        'peak': peak,
    }
