"""C013_markov_relaxation: Finite-state mass propagation and relaxation.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
"""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(313)
    matrix = rng.uniform(0.1, 1, (7, 7)).astype(np.float32)
    matrix /= matrix.sum(axis=1, keepdims=True)
    return {
        'transition': matrix,
        'state': np.array([[1, 0, 0, 0, 0, 0, 0]], dtype=np.float32),
        'prior': np.full((1, 7), 1 / 7, dtype=np.float32),
    }


def propagate(state, transition):
    moved = np.matmul(state, transition)
    return np.maximum(moved, 0.0)


def relax(state, prior, factor):
    retained = np.multiply(state, 1.0 - factor)
    injected = np.multiply(prior, factor)
    mixed = np.add(retained, injected)
    return mixed


def unit_mass(state):
    total = np.sum(state, axis=1, keepdims=True)
    safe = np.maximum(total, 1e-7)
    normalized = np.divide(state, safe)
    return normalized


def forward(transition, state, prior):
    current = state
    for iteration in range(8):
        moved = propagate(current, transition)
        if iteration % 2 == 0:
            current = relax(moved, prior, 0.1)
        else:
            current = relax(moved, prior, 0.2)
        current = unit_mass(current)
    mass = np.sum(current, axis=1)
    distance = np.sum(np.absolute(np.subtract(current, prior)), axis=1)
    return {
        'out': current,
        'mass': mass,
        'distance': distance,
    }
