"""D1012_class_selection: Classification logits and winning indices.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
argmax returns the first index on ties.
"""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(412)
    return {
        'x': rng.standard_normal((11, 7), dtype=np.float32),
    }


def forward(x):
    peak = np.max(x, axis=1, keepdims=True)
    shifted = np.subtract(x, peak)
    exp = np.exp(shifted)
    total = np.sum(exp, axis=1, keepdims=True)
    probabilities = np.divide(exp, total)
    winner = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)
    log_probabilities = np.log(probabilities)
    return {
        "log_probabilities": log_probabilities,
        'probabilities': probabilities,
        'winner': winner,
        'confidence': confidence,
    }
