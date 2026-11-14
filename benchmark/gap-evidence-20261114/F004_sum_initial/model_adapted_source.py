import numpy as np

def build_inputs():
    return {'x': np.full((1024, 1024), 0.5, dtype=np.float32)}

def forward(x):
    reduced = np.sum(x, axis=1)
    out = np.add(reduced, np.full_like(reduced, 10.0, dtype=reduced.dtype))
    return {'out': out, 'squared': np.square(out)}
