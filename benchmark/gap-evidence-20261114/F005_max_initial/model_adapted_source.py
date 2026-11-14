import numpy as np

def build_inputs():
    return {'x': np.full((1024, 1024), -0.5, dtype=np.float32)}

def forward(x):
    row_max = np.amax(x, axis=1)
    zero = np.full_like(row_max, 0.0, dtype=row_max.dtype)
    out = np.maximum(row_max, zero)
    one = np.full_like(out, 1.0, dtype=out.dtype)
    shifted = np.add(out, one)
    return {'out': out, 'shifted': shifted}
