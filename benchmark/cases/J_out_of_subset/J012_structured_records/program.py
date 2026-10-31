"""J012_structured_records: Named fields in a structured ndarray.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Analysis-only negative case: harness_runnable=False.
Do not execute as part of the deterministic NumPy case sweep.
"""
import numpy as np


def build_inputs():
    return {
        'records': np.array([(1, 2.5), (2, -1.0), (3, 4.0)], dtype=[('id', 'i4'), ('value', 'f8')]),
    }


def forward(records):
    out = records.copy()
    out["value"] = np.square(records["value"])
    return {
        'out': out,
    }
