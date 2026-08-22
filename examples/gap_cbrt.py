"""Operator-gap example: np.cbrt is intentionally outside the static snapshot."""

import numpy as np


def build_inputs():
    return (np.linspace(-8.0, 8.0, 33, dtype=np.float32),)


def forward(x):
    return np.cbrt(x)


if __name__ == "__main__":
    print(forward(*build_inputs()))
