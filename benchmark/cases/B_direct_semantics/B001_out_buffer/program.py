"""B001_out_buffer: Caller-owned output buffer.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Trap: out=. Pinned add rejects out; reference semantics only.
"""
import numpy as np


def build_inputs():
    return {
        'x': np.array([1, 2, 3], dtype=np.float32),
        'y': np.array([4, 5, 6], dtype=np.float32),
        'buffer': np.full(3, -99, dtype=np.float32),
    }


def forward(x, y, buffer):
    result = np.add(x, y, out=buffer)
    return {
        'out': result,
        'buffer': buffer,
    }
