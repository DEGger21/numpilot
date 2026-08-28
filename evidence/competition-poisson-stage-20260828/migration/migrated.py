# Copyright 2026 NumPilot contributors
# SPDX-License-Identifier: Apache-2.0
"""Poisson source-term residual map with a robust SmoothL1 penalty.

This application-shaped stage evaluates the analytical source term for
u(x, y) = sin(pi*x) sin(pi*y), perturbs a candidate forcing field, and emits
the unreduced pointwise SmoothL1(beta=1) loss map.  It is a benchmarkable stage
from a Poisson-Ritz workflow, not a claim that this file is a complete solver.
"""

import asnumpy as ap


x = ap.linspace(0.04, 0.96, 9000000, dtype=ap.float32)
y = ap.linspace(0.96, 0.04, 9000000, dtype=ap.float32)
pi = ap.full((9000000,), 3.1415927, dtype=ap.float32)
two_pi_squared = ap.full((9000000,), 19.739208, dtype=ap.float32)
prediction_scale = ap.full((9000000,), 0.92, dtype=ap.float32)
sigma = ap.full((9000000,), 1.0, dtype=ap.float32)
half = ap.full((9000000,), 0.5, dtype=ap.float32)
# NUMPILOT_DEVICE_INPUTS_END
phase_x = ap.multiply(pi, x)
phase_y = ap.multiply(pi, y)
mode_x = ap.sin(phase_x)
mode_y = ap.sin(phase_y)
source_shape = ap.multiply(mode_x, mode_y)
source = ap.multiply(two_pi_squared, source_shape)
prediction = ap.multiply(source, prediction_scale)
residual = ap.subtract(prediction, source)
absolute_residual = ap.absolute(residual)
bounded = ap.minimum(absolute_residual, sigma)
bounded_squared = ap.multiply(bounded, bounded)
quadratic = ap.multiply(bounded_squared, half)
linear = ap.subtract(absolute_residual, bounded)
result = ap.add(quadratic, linear)
