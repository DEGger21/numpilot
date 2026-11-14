import numpy as np

def build_inputs():
    rng = np.random.default_rng(402)
    return {
        'x': rng.standard_normal((1024, 1024), dtype=np.float32),
        'y': rng.standard_normal((1024, 1024), dtype=np.float32),
    }

def forward(x, y):
    absolute_1 = np.absolute(x)
    left = np.add(absolute_1, np.full_like(absolute_1, 1.0))
    absolute_2 = np.absolute(y)
    right = np.add(absolute_2, np.full_like(absolute_2, 1.0))
    left_ulp = np.spacing(left)
    right_ulp = np.spacing(right)
    absolute_3 = np.absolute(left_ulp)
    left_budget = np.multiply(absolute_3, np.full_like(absolute_3, 8.0))
    absolute_4 = np.absolute(right_ulp)
    right_budget = np.multiply(absolute_4, np.full_like(absolute_4, 12.0))
    budget = np.add(left_budget, right_budget)
    scale = np.maximum(left, right)
    relative = np.divide(budget, scale)
    return {'absolute': budget, 'relative': relative}
