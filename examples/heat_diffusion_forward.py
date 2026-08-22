"""One explicit-Euler step of a periodic 2-D heat-diffusion network.

``build_inputs`` prepares a Gaussian temperature field and its four periodic
neighbours on the CPU.  ``forward`` is the scientific tensor network that
NumPilot migrates: a five-point finite-difference Laplacian followed by the
residual time update ``u_next = u + alpha_dt * laplacian(u)``.
"""

import numpy as np


def build_inputs():
    side = 1024
    axis = np.linspace(-1.0, 1.0, side, dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    radius_squared = np.add(np.square(xx), np.square(yy))
    temperature = np.exp(np.multiply(radius_squared, np.float32(-18.0)))
    temperature = np.ascontiguousarray(temperature, dtype=np.float32)
    north = np.ascontiguousarray(np.roll(temperature, 1, axis=0))
    south = np.ascontiguousarray(np.roll(temperature, -1, axis=0))
    west = np.ascontiguousarray(np.roll(temperature, 1, axis=1))
    east = np.ascontiguousarray(np.roll(temperature, -1, axis=1))
    four = np.full_like(temperature, 4.0, dtype=np.float32)
    alpha_dt = np.full_like(temperature, 0.20, dtype=np.float32)
    return temperature, north, south, west, east, four, alpha_dt


def forward(temperature, north, south, west, east, four, alpha_dt):
    vertical = np.add(north, south)
    horizontal = np.add(west, east)
    neighbours = np.add(vertical, horizontal)
    centre = np.multiply(temperature, four)
    laplacian = np.subtract(neighbours, centre)
    diffusion = np.multiply(alpha_dt, laplacian)
    return np.add(temperature, diffusion)


if __name__ == "__main__":
    output = forward(*build_inputs())
    if hasattr(output, "to_numpy"):
        output = output.to_numpy()
    print(
        "shape:",
        output.shape,
        "dtype:",
        output.dtype,
        "finite:",
        bool(np.isfinite(output).all()),
        "sum:",
        float(output.sum()),
    )
