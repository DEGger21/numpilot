import numpy as np

def build_inputs():
    return {'offset': np.full((1048576,), 0.5, dtype=np.float32)}

def forward(offset):
    n = 1048576
    grid = np.linspace(0.0, 1.0 - 1.0 / n, n, dtype=np.float32)
    return {'out': np.add(grid, offset)}
