# This file is part of the NumPilot project at Harbin Institute of Technology (HIT).
#
# Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
# SPDX-License-Identifier: Apache-2.0
#
# Authors (accounts):
# - Su Tonghua <@sutonghua>
# - Li Xing <@li-xingHIT>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Black--Scholes-style portfolio repricing workload.

The file is intentionally an ordinary, single-file NumPy program.  It prices
``N`` independent European call contracts with a smooth normal-CDF
approximation.  Input construction is kept above the benchmark marker; every
operation below it is a straight-line, float32 tensor graph that can be moved
to AsNumpy without changing the scientific story.

The 4,000,000-contract default is large enough to make device residency
visible while staying practical on the validated runtime profile.  ``result``
is the final output consumed by ``numpilot run`` and ``benchmark-script``.
"""

import numpy as np


# Keep the extent literal: the ordinary analyzer must prove the exact shape
# without executing Python or evaluating arbitrary names.  The strike grid
# keeps every contract solidly in the money, so ``result`` stays far away
# from the near-zero cancellation band where float32 CPU/NPU ulp noise
# would sit on top of the strict absolute-tolerance comparison floor.
spot = np.linspace(80.0, 160.0, 4_000_000, dtype=np.float32)
strike = np.linspace(50.0, 68.0, 4_000_000, dtype=np.float32)
time_to_expiry = np.linspace(0.10, 2.00, 4_000_000, dtype=np.float32)
volatility = np.linspace(0.12, 0.45, 4_000_000, dtype=np.float32)
risk_free = np.full(4_000_000, 0.03, dtype=np.float32)
half = np.full(4_000_000, 0.5, dtype=np.float32)
one = np.full(4_000_000, 1.0, dtype=np.float32)
normal_scale = np.full(4_000_000, 0.79788456, dtype=np.float32)
zero = np.full((4_000_000,), 0.0, dtype=np.float32)
# NUMPILOT_DEVICE_INPUTS_END

# Black--Scholes d1/d2 terms.  The tanh-based CDF avoids a dependency on a
# special-function extension while remaining a useful portfolio-pricing
# approximation for this migration example.
variance = np.multiply(volatility, volatility)
sqrt_time = np.sqrt(time_to_expiry)
sigma_sqrt_time = np.multiply(volatility, sqrt_time)
log_moneyness = np.log(np.divide(spot, strike))
carry = np.multiply(half, np.multiply(variance, time_to_expiry))
d1 = np.divide(np.add(log_moneyness, carry), sigma_sqrt_time)
d2 = np.subtract(d1, sigma_sqrt_time)

normal_cdf_d1 = np.multiply(
    half,
    np.add(one, np.tanh(np.multiply(normal_scale, d1))),
)
normal_cdf_d2 = np.multiply(
    half,
    np.add(one, np.tanh(np.multiply(normal_scale, d2))),
)
discount = np.exp(np.negative(np.multiply(risk_free, time_to_expiry)))
discounted_strike = np.multiply(
    np.multiply(strike, discount),
    normal_cdf_d2,
)
intrinsic_and_time_value = np.subtract(
    np.multiply(spot, normal_cdf_d1),
    discounted_strike,
)
result = np.maximum(intrinsic_and_time_value, zero)
