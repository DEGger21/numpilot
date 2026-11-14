import asnumpy as ap
import numpy as np

def build_inputs():
    rng = np.random.default_rng(402)
    return {
        'x': rng.standard_normal((1024, 1024), dtype=np.float32),
        'y': rng.standard_normal((1024, 1024), dtype=np.float32),
    }
def forward(x, y):
    if isinstance(x, np.ndarray):
        x = ap.ndarray.from_numpy(x)
    if isinstance(y, np.ndarray):
        y = ap.ndarray.from_numpy(y)
    absolute_1 = ap.absolute(x)
    left = ap.add(absolute_1, ap.full_like(absolute_1, 1.0))
    absolute_2 = ap.absolute(y)
    right = ap.add(absolute_2, ap.full_like(absolute_2, 1.0))
    left_ulp = ap.spacing(left)
    right_ulp = ap.spacing(right)
    absolute_3 = ap.absolute(left_ulp)
    left_budget = ap.multiply(absolute_3, ap.full_like(absolute_3, 8.0))
    absolute_4 = ap.absolute(right_ulp)
    right_budget = ap.multiply(absolute_4, ap.full_like(absolute_4, 12.0))
    budget = ap.add(left_budget, right_budget)
    scale = ap.maximum(left, right)
    relative = ap.divide(budget, scale)
    return {'absolute': budget, 'relative': relative}
