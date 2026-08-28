# Copyright 2026 NumPilot contributors
# SPDX-License-Identifier: Apache-2.0
#
# Competition-facing synthetic resident-compute showcase.  The marker is only
# a benchmark boundary; ordinary NumPilot migration itself needs no annotations.
import numpy as np

base = np.linspace(1.2, 2.4, 9000000, dtype=np.float32)
exponent = np.linspace(0.5, 0.9, 9000000, dtype=np.float32)
scale = np.linspace(1.5, 2.5, 9000000, dtype=np.float32)
# NUMPILOT_DEVICE_INPUTS_END
powered = np.power(base, exponent)
logged = np.log(powered)
amplified = np.multiply(logged, scale)
activated = np.exp(amplified)
ratio = np.divide(activated, base)
result = np.tanh(ratio)
