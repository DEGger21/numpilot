import asnumpy as ap
import numpy as np

def build_inputs():
    rng = np.random.default_rng(404)
    return {
        'x': rng.standard_normal((1024, 1024), dtype=np.float32),
        'y': rng.standard_normal((1024, 1024), dtype=np.float32),
    }
def forward(x, y):
    if isinstance(x, np.ndarray):
        x = ap.ndarray.from_numpy(x)
    if isinstance(y, np.ndarray):
        y = ap.ndarray.from_numpy(y)
    multiply_1 = ap.multiply(x, ap.full_like(x, 0.5))
    left = ap.add(multiply_1, ap.full_like(multiply_1, 0.25))
    multiply_2 = ap.multiply(y, ap.full_like(y, 0.25))
    right = ap.subtract(multiply_2, ap.full_like(multiply_2, 0.1))
    left_velocity = ap.diff(left, axis=1)
    right_velocity = ap.diff(right, axis=1)
    residual = ap.subtract(left_velocity, right_velocity)
    squared = ap.square(residual)
    mean_3 = ap.mean(squared, axis=1)
    rmse = ap.sqrt(mean_3)
    bounded = ap.clip(residual, ap.full_like(residual, -1.0), ap.full_like(residual, 1.0))
    return {'residual': residual, 'bounded': bounded, 'rmse': rmse}
