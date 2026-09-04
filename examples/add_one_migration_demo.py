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
"""Compute-heavy ordinary NumPy workload with one tiny add-one gap.

NumPy accepts ``add(array, 1.0)`` while the pinned AsNumpy binary binding only
qualifies two-array ``add``.  NumPilot recognizes this exact float32 case as
the registered ``add_one`` gap.  The remaining supported power/log/exp chain
keeps the missing operator simple while giving the performance gate enough
device-resident arithmetic to measure a meaningful NPU advantage.
"""

import numpy as np


base = np.linspace(1.2, 2.4, 9_000_000, dtype=np.float32)
exponent = np.linspace(0.5, 0.9, 9_000_000, dtype=np.float32)
scale = np.linspace(1.5, 2.5, 9_000_000, dtype=np.float32)
# NUMPILOT_DEVICE_INPUTS_END
shifted = np.add(base, 1.0)
powered = np.power(shifted, exponent)
logged = np.log(powered)
amplified = np.multiply(logged, scale)
activated = np.exp(amplified)
ratio = np.divide(activated, shifted)
result = np.tanh(ratio)
