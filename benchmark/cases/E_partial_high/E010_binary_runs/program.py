"""E010_binary_runs: Bit packing and transitions in a boolean signal.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
The compute path combines at least three missing APIs with a supported anchor.
Named intermediate outputs expose indexing, shape and numerical behavior.
"""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(510)
    x = rng.standard_normal(96, dtype=np.float32)
    x[:4] = 1.0
    x[30:45] = -1.0
    x[-7:] = 2.0
    return {
        'x': x,
    }


def forward(x):
    mask = np.greater(x, 0)
    packed = np.packbits(mask, bitorder='little')
    restored = np.unpackbits(packed, count=x.size, bitorder='little')
    pad_1 = np.pad(restored.astype(np.int32), (1, 1))
    boundaries = np.diff(pad_1)
    starts = np.flatnonzero(boundaries == 1)
    stops = np.flatnonzero(boundaries == -1)
    lengths = stops - starts
    histogram = np.bincount(lengths)
    order = np.argsort(lengths)
    return {
        'packed': packed,
        'restored': restored,
        'starts': starts,
        'stops': stops,
        'lengths': lengths,
        'histogram': histogram,
        'order': order,
    }
