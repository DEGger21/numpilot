"""A010_boolean_logic: Two boolean masks."""
import numpy as np


def build_inputs():
    return {"x": np.array([True, False, True, False]),
            "y": np.array([False, True, True, False])}


def forward(x, y):
    return {"both": np.logical_and(x, y), "either": np.logical_or(x, y)}
