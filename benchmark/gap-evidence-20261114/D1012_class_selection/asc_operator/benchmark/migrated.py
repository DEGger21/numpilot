import asnumpy as ap
"""D1012_class_selection: Classification logits and winning indices.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
argmax returns the first index on ties.
"""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(412)
    return {
        'x': rng.standard_normal((1024, 1024), dtype=np.float32),
    }

def forward(x):
    if isinstance(x, np.ndarray):
        x = ap.ndarray.from_numpy(x)
    peak = ap.max(x, axis=1, keepdims=True)
    shifted = ap.subtract(x, peak)
    exp = ap.exp(shifted)
    total = ap.sum(exp, axis=1, keepdims=True)
    probabilities = ap.divide(exp, total)
    winner = ap.argmax(probabilities, axis=1)
    confidence = ap.max(probabilities, axis=1)
    log_probabilities = ap.log(probabilities)
    return {
        "log_probabilities": log_probabilities,
        'probabilities': probabilities,
        'winner': winner,
        'confidence': confidence,
    }
