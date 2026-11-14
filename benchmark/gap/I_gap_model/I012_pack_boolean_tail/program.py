"""I012_pack_boolean_tail: Packing a non-byte-aligned boolean axis.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
I3: bool input, uint8 output, and three trailing bits.
The target API is absent from the pinned public compute exports.
Inputs include the contract boundary rather than only nominal values.
"""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(912)
    bits = rng.integers(0, 2, (32, 1027), dtype=np.uint8).astype(bool)
    bits[:, -3:] = True
    return {
        'bits': bits,
    }


def forward(bits):
    return {
        'out': np.packbits(bits, axis=1, bitorder='little'),
    }
