"""J007_threaded_compute: Concurrent Python execution of NumPy chunks.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Analysis-only negative case: harness_runnable=False.
Do not execute as part of the deterministic NumPy case sweep.
"""
import numpy as np


def build_inputs():
    return {
        'x': np.arange(32, dtype=np.float32).reshape(4, 8),
    }


def forward(x):
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as pool:
        chunks = list(pool.map(np.square, x))
    return {
        'out': np.stack(chunks),
    }
