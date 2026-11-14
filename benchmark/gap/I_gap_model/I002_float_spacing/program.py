"""I002_float_spacing: Signed spacing including subnormal values.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
I1: one unary elementwise output.
The target API is absent from the pinned public compute exports.
Inputs include the contract boundary rather than only nominal values.
Source: ../../../../examples/gap_spacing.py
Adaptation: bounded inputs, named outputs, explicit computational stages.
"""
# Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
# SPDX-License-Identifier: Apache-2.0
import numpy as np


def build_inputs():
    x = np.linspace(-4, 4, 257, dtype=np.float32)
    x[128] = np.nextafter(np.float32(0), np.float32(1))
    return {
        'x': x,
    }


def forward(x):
    return {
        'out': np.spacing(x),
    }
