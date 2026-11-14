"""J002_dynamic_execution: Evaluation of a runtime expression.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Analysis-only negative case: harness_runnable=False.
Do not execute as part of the deterministic NumPy case sweep.
"""
import numpy as np


def build_inputs():
    return {
        'x': np.arange(6, dtype=np.float32),
        'expression': np.array('np.square(x) + 1'),
    }


def forward(x, expression):
    out = eval(expression.item(), {"np": np, "x": x})
    return {
        'out': out,
    }
