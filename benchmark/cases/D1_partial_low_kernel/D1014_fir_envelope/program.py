"""D1014_fir_envelope: Finite impulse response followed by an energy envelope.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
convolve uses an odd symmetric FIR with same-length output.
"""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(414)
    return {
        'x': rng.standard_normal(97, dtype=np.float32),
        'kernel': np.array([0.1, 0.2, 0.4, 0.2, 0.1], dtype=np.float32),
    }


def forward(x, kernel):
    bounded = np.clip(x, -4.0, 4.0)
    response = np.convolve(bounded, kernel, mode='same')
    power = np.square(response)
    add_1 = np.add(power, 1e-06)
    envelope = np.sqrt(add_1)
    compressed = np.log1p(envelope)
    add_2 = np.add(1.0, compressed)
    output = np.divide(compressed, add_2)
    residual = np.subtract(bounded, response)
    return {
        "residual": residual,
        'filtered': response,
        'out': output,
    }
