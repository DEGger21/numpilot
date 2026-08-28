# Copyright 2026 NumPilot contributors
# SPDX-License-Identifier: Apache-2.0
#
# Competition-facing synthetic resident-compute showcase.  The marker is only
# a benchmark boundary; ordinary NumPilot migration itself needs no annotations.
import asnumpy as ap

base = ap.linspace(1.2, 2.4, 9000000, dtype=ap.float32)
exponent = ap.linspace(0.5, 0.9, 9000000, dtype=ap.float32)
scale = ap.linspace(1.5, 2.5, 9000000, dtype=ap.float32)
# NUMPILOT_DEVICE_INPUTS_END
powered = ap.power(base, exponent)
logged = ap.log(powered)
amplified = ap.multiply(logged, scale)
activated = ap.exp(amplified)
ratio = ap.divide(activated, base)
result = ap.tanh(ratio)
