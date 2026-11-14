import numpy as np

def build_inputs():
    return {
        'x': np.full((1024, 1024), 0.5, dtype=np.float32),
        'y': np.full((1024, 1024), 0.25, dtype=np.float32),
        'buffer': np.full((1024, 1024), -9.0, dtype=np.float32),
    }

def forward(x, y, buffer):
    out = np.add(x, y)
    return {'out': out, 'norm': np.square(out)}
