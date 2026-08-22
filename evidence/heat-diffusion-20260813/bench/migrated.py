"""One explicit-Euler step of a periodic 2-D heat-diffusion network.

``build_inputs`` prepares a Gaussian temperature field and its four periodic
neighbours on the CPU.  ``forward`` is the scientific tensor network that
NumPilot migrates: a five-point finite-difference Laplacian followed by the
residual time update ``u_next = u + alpha_dt * laplacian(u)``.
"""
import numpy as np
import asnumpy as ap

def build_inputs():
    side = 1024
    axis = np.linspace(-1.0, 1.0, side, dtype=np.float32)
    (xx, yy) = np.meshgrid(axis, axis, indexing='xy')
    radius_squared = np.add(np.square(xx), np.square(yy))
    temperature = np.exp(np.multiply(radius_squared, np.float32(-18.0)))
    temperature = np.ascontiguousarray(temperature, dtype=np.float32)
    north = np.ascontiguousarray(np.roll(temperature, 1, axis=0))
    south = np.ascontiguousarray(np.roll(temperature, -1, axis=0))
    west = np.ascontiguousarray(np.roll(temperature, 1, axis=1))
    east = np.ascontiguousarray(np.roll(temperature, -1, axis=1))
    four = np.full_like(temperature, 4.0, dtype=np.float32)
    alpha_dt = np.full_like(temperature, 0.2, dtype=np.float32)
    return (temperature, north, south, west, east, four, alpha_dt)

def forward(temperature, north, south, west, east, four, alpha_dt):
    if isinstance(temperature, np.ndarray):
        temperature = ap.ndarray.from_numpy(temperature)
    if isinstance(north, np.ndarray):
        north = ap.ndarray.from_numpy(north)
    if isinstance(south, np.ndarray):
        south = ap.ndarray.from_numpy(south)
    if isinstance(west, np.ndarray):
        west = ap.ndarray.from_numpy(west)
    if isinstance(east, np.ndarray):
        east = ap.ndarray.from_numpy(east)
    if isinstance(four, np.ndarray):
        four = ap.ndarray.from_numpy(four)
    if isinstance(alpha_dt, np.ndarray):
        alpha_dt = ap.ndarray.from_numpy(alpha_dt)
    vertical = ap.add(north, south)
    horizontal = ap.add(west, east)
    neighbours = ap.add(vertical, horizontal)
    centre = ap.multiply(temperature, four)
    laplacian = ap.subtract(neighbours, centre)
    diffusion = ap.multiply(alpha_dt, laplacian)
    return ap.add(temperature, diffusion)
if __name__ == '__main__':
    output = forward(*build_inputs())
    if hasattr(output, 'to_numpy'):
        output = output.to_numpy()
    print('shape:', output.shape, 'dtype:', output.dtype, 'finite:', bool(np.isfinite(output).all()), 'sum:', float(output.sum()))