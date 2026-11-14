import numpy as np

def build_inputs():
    return {'offset': np.full((1048576,), 0.5, dtype=np.float32)}

def forward(offset):
    grid = np.linspace(0.0, 1.0, 1048576, endpoint=False, dtype=np.float32)
    return {'out': np.add(grid, offset)}
