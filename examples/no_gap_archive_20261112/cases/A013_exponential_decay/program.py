"""A013_exponential_decay: Elementwise exponential decay."""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(13)
    return {
        "x": rng.standard_normal((96,), dtype=np.float32),
    }


def forward(x):
    return {"out": np.exp(np.negative(np.square(x)))}
