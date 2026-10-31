"""I001_cube_root: Signed cube root including signed zero.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
I1: one unary elementwise output.
The target API is absent from the pinned public compute exports.
Inputs include the contract boundary rather than only nominal values.
Source: ../../../../examples/gap_cbrt.py
Adaptation: bounded inputs, named outputs, explicit computational stages.
"""
# Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
# SPDX-License-Identifier: Apache-2.0
import numpy as np


def build_inputs():
    x = np.linspace(-27, 27, 257, dtype=np.float32)
    x[128] = -0.0
    return {
        'x': x,
    }


def forward(x):
    return {
        'out': np.cbrt(x),
    }
