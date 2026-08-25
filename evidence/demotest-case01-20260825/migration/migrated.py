# *****************************************************************************
# Copyright (c) 2025 AISS Group at Harbin Institute of Technology. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# *****************************************************************************

"""
GLM-generated 2048x2048 float32 performance variant of Case 01 (numpy):
matrix multiplication followed by row-wise Softmax and row-sum validation.
"""

import asnumpy as ap


# Construct inputs (2048x2048) and weights (2048x2048); matmul yields (2048x2048) logits
x = ap.full((2048, 2048), 0.01, dtype=ap.float32)
w = ap.full((2048, 2048), 0.01, dtype=ap.float32)
logits = ap.matmul(x, w)

# Softmax: convert each row into a probability distribution
exp_logits = ap.exp(logits)
probs = ap.divide(exp_logits, ap.sum(exp_logits, axis=1, keepdims=True))

# Sum each row to validate probabilities sum to 1, then aggregate a checksum
row_sums = ap.sum(probs, axis=1, keepdims=True)
checksum = ap.sum(row_sums, axis=0, keepdims=True)

print(f"row_sum_checksum = {checksum.to_numpy()}")
