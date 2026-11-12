"""A001_add_sqrt: Magnitude of a vector sum."""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(1)
    return {
        "x": rng.standard_normal((4_000_000,), dtype=np.float32),
        "y": rng.standard_normal((4_000_000,), dtype=np.float32),
    }


def forward(x, y):
    z = np.add(x, y)
    return {"out": np.sqrt(np.square(z))}
