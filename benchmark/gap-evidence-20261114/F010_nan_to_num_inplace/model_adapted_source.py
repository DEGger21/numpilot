import numpy as np

def build_inputs():
    return {'x': np.full((1024, 1024), 0.5, dtype=np.float32)}

def forward(x):
    out = np.nan_to_num(x, nan=3.0, posinf=4.0, neginf=-4.0)
    x = out
    return {'out': out, 'input_after': x, 'squared': np.square(x)}
