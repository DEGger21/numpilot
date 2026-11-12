"""C009_poisson_robust_map: Poisson source and robust residual penalties.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Source: ../../../../examples/poisson_source_smooth_l1_map.py
Adaptation: bounded inputs, named outputs, explicit computational stages.
"""
# Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
# SPDX-License-Identifier: Apache-2.0
import numpy as np


def build_inputs():
    return {
        'x': np.linspace(0.04, 0.96, 81, dtype=np.float32).reshape(9, 9),
        'y': np.linspace(0.96, 0.04, 81, dtype=np.float32).reshape(9, 9),
        'scale': np.full((9, 9), 0.92, dtype=np.float32),
    }


def forcing(x, y):
    phase_x = np.multiply(x, 3.1415927)
    phase_y = np.multiply(y, 3.1415927)
    mode_x = np.sin(phase_x)
    mode_y = np.sin(phase_y)
    return np.multiply(np.multiply(mode_x, mode_y), 19.739208)


def penalty(absolute_error):
    bounded = np.minimum(absolute_error, 1.0)
    quadratic = np.multiply(np.square(bounded), 0.5)
    linear = np.subtract(absolute_error, bounded)
    return np.add(quadratic, linear)


def smooth_l1(prediction, source):
    residual = np.subtract(prediction, source)
    return penalty(np.absolute(residual))


def forward(x, y, scale):
    source = forcing(x, y)
    prediction = np.multiply(source, scale)
    first = smooth_l1(prediction, source)
    current = prediction
    for step in range(4):
        error = np.subtract(current, source)
        if step < 2:
            gradient = np.clip(error, -1.0, 1.0)
        else:
            gradient = np.tanh(error)
        current = np.subtract(current, np.multiply(gradient, 0.1))
    final = smooth_l1(current, source)
    return {
        'source': source,
        'loss_before': first,
        'loss_after': final,
        'out': current,
    }
