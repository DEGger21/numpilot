"""S10_boolean_logic_4m: Two boolean masks, 4M-element scaled variant."""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(10)
    return {"x": rng.random(4_000_000) < 0.5, "y": rng.random(4_000_000) < 0.5}


def forward(x, y):
    return {"both": np.logical_and(x, y), "either": np.logical_or(x, y)}
