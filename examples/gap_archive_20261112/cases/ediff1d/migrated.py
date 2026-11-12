import numpy as np
import asnumpy as ap

def build_inputs():
    rng = np.random.default_rng(20261112)
    return {'a': rng.standard_normal((2048, 2048), dtype=np.float32)}

def forward(a):
    if isinstance(a, np.ndarray):
        a = ap.ndarray.from_numpy(a)
    r = ap.ediff1d(a)
    return {'r': r}