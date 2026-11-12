"""C010_energy_feedback: Energy network with damped feedback.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Source: ../../../../examples/elementwise_forward.py
Adaptation: bounded inputs, named outputs, explicit computational stages.
"""
# Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
# SPDX-License-Identifier: Apache-2.0
import numpy as np


def build_inputs():
    return {
        "x": np.linspace(-2, 2, 4_194_304, dtype=np.float32).reshape(2048, 2048),
        "y": np.linspace(2, -2, 4_194_304, dtype=np.float32).reshape(2048, 2048),
        "drive": np.full((2048, 2048), 0.05, dtype=np.float32),
    }


def amplitude(x, y):
    x_energy = np.square(x)
    y_energy = np.square(y)
    energy = np.add(x_energy, y_energy)
    magnitude = np.sqrt(energy)
    return np.tanh(magnitude)


def feedback(state, shared, drive):
    damping = np.multiply(state, 0.8)
    forcing = np.multiply(shared, drive)
    updated = np.add(damping, forcing)
    return updated


def compare_rows(current, previous):
    delta = np.subtract(current, previous)
    return np.mean(np.square(delta), axis=1)


def forward(x, y, drive):
    baseline = amplitude(x, y)
    left, right = x, y
    for iteration in range(4):
        shared = amplitude(left, right)
        if iteration % 2 == 0:
            left = feedback(left, shared, drive)
        else:
            right = feedback(right, shared, drive)
    final = amplitude(left, right)
    row_change = compare_rows(final, baseline)
    return {
        'out': final,
        'left': left,
        'right': right,
        'change': row_change,
    }
