"""J003_file_io: NumPy file serialization in a temporary file.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Analysis-only negative case: harness_runnable=False.
Do not execute as part of the deterministic NumPy case sweep.
"""
import numpy as np


def build_inputs():
    return {
        'x': np.arange(12, dtype=np.float32).reshape(3, 4),
    }


def forward(x):
    import tempfile
    with tempfile.TemporaryFile() as handle:
        np.save(handle, x)
        handle.seek(0)
        out = np.load(handle, allow_pickle=False)
    return {
        'out': out,
    }
