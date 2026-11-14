import asnumpy as ap
import numpy as np

def build_inputs():
    rng = np.random.default_rng(409)
    return {'x': rng.standard_normal((1024, 1024), dtype=np.float32)}
def forward(x):
    if isinstance(x, np.ndarray):
        x = ap.ndarray.from_numpy(x)
    bounded = ap.clip(x, ap.full_like(x, -2.0), ap.full_like(x, 2.0))
    upstream = ap.roll(bounded, 1, axis=1)
    gradient = ap.subtract(bounded, upstream)
    flux = ap.multiply(gradient, ap.full_like(gradient, 0.2))
    updated = ap.subtract(bounded, flux)
    square_1 = ap.square(updated)
    energy = ap.sum(square_1, axis=1)
    subtract_2 = ap.subtract(updated, bounded)
    absolute_3 = ap.absolute(subtract_2)
    change = ap.sum(absolute_3, axis=1)
    return {'out': updated, 'energy': energy, 'change': change}
