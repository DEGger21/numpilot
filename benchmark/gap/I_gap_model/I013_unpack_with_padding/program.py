"""I013_unpack_with_padding: Unpacking beyond the byte extent adds zeros.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
I3: five zero padding bits beyond 129 bytes.
The target API is absent from the pinned public compute exports.
Inputs include the contract boundary rather than only nominal values.
"""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(913)
    packed = rng.integers(0, 256, (24, 129), dtype=np.uint8)
    return {
        'packed': packed,
    }


def forward(packed):
    return {
        'out': np.unpackbits(packed, axis=1, count=1037, bitorder='big'),
    }
