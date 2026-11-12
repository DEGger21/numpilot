"""C014_logistic_training: Fixed-step logistic regression.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
"""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(314)
    x = rng.standard_normal((20, 6), dtype=np.float32)
    return {
        'x': x,
        'xt': np.ascontiguousarray(x.T),
        'y': rng.integers(0, 2, (20, 1)).astype(np.float32),
        'weights': np.zeros((6, 1), dtype=np.float32),
    }


def predict(x, weights):
    logits = np.matmul(x, weights)
    bounded = np.clip(logits, -30.0, 30.0)
    return np.reciprocal(np.add(1.0, np.exp(np.negative(bounded))))


def gradient_step(xt, probability, target, weights):
    residual = np.subtract(probability, target)
    data = np.divide(np.matmul(xt, residual), 20.0)
    penalty = np.multiply(weights, 0.01)
    return np.add(data, penalty)


def loss_rows(probability, target):
    bounded = np.clip(probability, 1e-6, 1.0 - 1e-6)
    positive = np.multiply(target, np.log(bounded))
    negative = np.multiply(np.subtract(1.0, target), np.log1p(np.negative(bounded)))
    return np.negative(np.add(positive, negative))


def forward(x, xt, y, weights):
    current = weights
    for epoch in range(7):
        probability = predict(x, current)
        gradient = gradient_step(xt, probability, y, current)
        if epoch < 4:
            rate = 0.2
        else:
            rate = 0.05
        current = np.subtract(current, np.multiply(gradient, rate))
    probability = predict(x, current)
    loss = loss_rows(probability, y)
    return {
        'weights': current,
        'probability': probability,
        'loss': loss,
    }
