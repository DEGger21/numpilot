import numpy as np

def build_inputs():
    return {'x': np.full((1024, 1024), 0.5, dtype=np.float32)}

def forward(x):
    out = np.sum(x, axis=1, initial=10.0)
    return {'out': out, 'squared': np.square(out)}
