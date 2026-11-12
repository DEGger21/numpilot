"""C007_least_squares_steps: Bounded gradient descent on squared error.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Source: ../../../../examples/custom_operator_scaled_l2.py
Adaptation: bounded inputs, named outputs, explicit computational stages.
"""
# Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
# SPDX-License-Identifier: Apache-2.0
import numpy as np


def build_inputs():
    rng = np.random.default_rng(307)
    return {
        "predict": rng.standard_normal((16384, 256), dtype=np.float32),
        "label": rng.standard_normal((16384, 256), dtype=np.float32),
        "penalty": np.full((16384, 256), 0.05, dtype=np.float32),
    }


def row_loss(predict, label):
    error = np.subtract(predict, label)
    squared = np.multiply(error, error)
    return np.mean(squared, axis=1)


def gradient_of_loss(predict, label, penalty):
    residual = np.subtract(predict, label)
    data_gradient = np.multiply(residual, 2.0)
    regularizer = np.multiply(predict, penalty)
    return np.add(data_gradient, regularizer)


def update(predict, gradient, rate):
    bounded = np.clip(gradient, -4.0, 4.0)
    step = np.multiply(bounded, rate)
    return np.subtract(predict, step)


def forward(predict, label, penalty):
    initial_loss = row_loss(predict, label)
    current = predict
    step = 0
    while step < 6:
        gradient = gradient_of_loss(current, label, penalty)
        if step < 4:
            current = update(current, gradient, 0.1)
        else:
            current = update(current, gradient, 0.025)
        step += 1
    final_loss = row_loss(current, label)
    return {
        'out': current,
        'initial_loss': initial_loss,
        'final_loss': final_loss,
    }
