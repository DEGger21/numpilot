import numpy as np


def build_inputs():
    rng = np.random.default_rng(20261112)
    return {'a': rng.standard_normal((32, 16), dtype=np.float32)}


def forward(a):
    r = np.argpartition(a, 3, axis=1)
    return {'r': r}
