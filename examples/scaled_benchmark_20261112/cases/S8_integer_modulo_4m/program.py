"""S8_integer_modulo_4m: Signed integer remainders, 4M-element scaled variant."""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(11)
    return {
        "x": rng.integers(-1_000_000, 1_000_001, size=(4_000_000,), dtype=np.int32),
        "y": rng.integers(1, 1000, size=(4_000_000,), dtype=np.int32),
    }


def forward(x, y):
    return {"out": np.remainder(x, y)}
