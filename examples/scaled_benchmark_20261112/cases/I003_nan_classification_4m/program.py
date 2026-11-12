"""I003_nan_classification: NaN classification with infinities and signed zero.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
I1: one unary elementwise bool output.
The target API is absent from the pinned public compute exports.
Inputs include the contract boundary rather than only nominal values.
Source: ../../../../examples/gap_isnan.py
Adaptation: bounded inputs, named outputs, explicit computational stages.
"""
# Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
# SPDX-License-Identifier: Apache-2.0
import numpy as np


def build_inputs():
    return {
        "x": np.resize(np.array([0, -0.0, np.nan, np.inf, -np.inf, -1, 1], dtype=np.float32), (4_000_000,)),
    }


def forward(x):
    return {
        'out': np.isnan(x),
    }
