import numpy as np

def build_inputs():
    return {'x': np.full((1024, 1024), -0.5, dtype=np.float32)}

def forward(x):
    out = np.max(x, axis=1, initial=0.0)
    return {'out': out, 'shifted': np.add(out, 1.0)}
