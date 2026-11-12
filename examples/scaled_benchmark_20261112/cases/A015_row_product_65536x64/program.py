"""A015_row_product: Products of bounded positive entries."""
import numpy as np


def build_inputs():
    return {
        "x": np.linspace(0.8, 1.2, 4_194_304, dtype=np.float32).reshape(65536, 64),
    }


def forward(x):
    return {"out": np.prod(x, axis=1)}
