"""J015_memory_mapping: Memory-mapped temporary array storage.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Analysis-only negative case: harness_runnable=False.
Do not execute as part of the deterministic NumPy case sweep.
"""
import numpy as np


def build_inputs():
    return {
        'x': np.arange(24, dtype=np.float32).reshape(4, 6),
    }


def forward(x):
    import tempfile
    with tempfile.TemporaryFile() as handle:
        handle.truncate(x.nbytes)
        mapped = np.memmap(handle, dtype=x.dtype, mode="r+", shape=x.shape)
        mapped[:] = x
        mapped.flush()
        out = np.square(mapped).copy()
        del mapped
    return {
        'out': out,
    }
