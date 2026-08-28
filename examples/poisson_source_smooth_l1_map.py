# Copyright 2026 NumPilot contributors
# SPDX-License-Identifier: Apache-2.0
"""Poisson source-term residual map with a robust SmoothL1 penalty.

This application-shaped stage evaluates the analytical source term for
u(x, y) = sin(pi*x) sin(pi*y), perturbs a candidate forcing field, and emits
the unreduced pointwise SmoothL1(beta=1) loss map.  It is a benchmarkable stage
from a Poisson-Ritz workflow, not a claim that this file is a complete solver.
"""

import numpy as np


x = np.linspace(0.04, 0.96, 9000000, dtype=np.float32)
y = np.linspace(0.96, 0.04, 9000000, dtype=np.float32)
pi = np.full((9000000,), 3.1415927, dtype=np.float32)
two_pi_squared = np.full((9000000,), 19.739208, dtype=np.float32)
prediction_scale = np.full((9000000,), 0.92, dtype=np.float32)
sigma = np.full((9000000,), 1.0, dtype=np.float32)
half = np.full((9000000,), 0.5, dtype=np.float32)
# NUMPILOT_DEVICE_INPUTS_END
phase_x = np.multiply(pi, x)
phase_y = np.multiply(pi, y)
mode_x = np.sin(phase_x)
mode_y = np.sin(phase_y)
source_shape = np.multiply(mode_x, mode_y)
source = np.multiply(two_pi_squared, source_shape)
prediction = np.multiply(source, prediction_scale)
residual = np.subtract(prediction, source)
absolute_residual = np.absolute(residual)
bounded = np.minimum(absolute_residual, sigma)
bounded_squared = np.multiply(bounded, bounded)
quadratic = np.multiply(bounded_squared, half)
linear = np.subtract(absolute_residual, bounded)
result = np.add(quadratic, linear)
