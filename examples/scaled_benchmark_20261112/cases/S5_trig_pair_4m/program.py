"""A003_trig_pair: Sine and cosine of independent angles."""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(3)
    return {
        "x": rng.standard_normal((4_000_000,), dtype=np.float32),
    }


def forward(x):
    return {"sin": np.sin(x), "cos": np.cos(x)}
