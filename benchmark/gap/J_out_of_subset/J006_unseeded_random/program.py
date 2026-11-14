"""J006_unseeded_random: Unseeded randomness in the computational entry point.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Analysis-only negative case: harness_runnable=False.
Do not execute as part of the deterministic NumPy case sweep.
"""
import numpy as np


def build_inputs():
    return {
        'x': np.ones(16, dtype=np.float32),
    }


def forward(x):
    noise = np.random.rand(x.size)
    return {
        'out': np.add(x, noise),
    }
