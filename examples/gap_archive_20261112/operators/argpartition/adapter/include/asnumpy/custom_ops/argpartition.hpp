/******************************************************************************
 * Copyright (c) 2026 AISS Group at Harbin Institute of Technology. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *****************************************************************************/

#pragma once

#include <asnumpy/utils/npu_array.hpp>

#include <cstdint>

namespace asnumpy {

/**
 * @brief Compute the NumPy-compatible ``argpartition`` result on the NPU.
 *
 * Admitted profile: rank 2 float32 input, axis = 1 (or its rank-2 alias -1), 0 <= kth < a.shape[1] <= 32, int64 output of the same shape.
 * Any other dtype, rank, shape, axis or keyword is rejected explicitly by the
 * integration layer. No implicit cast and no CPU fallback are provided.
 */
NPUArray NumpyArgpartition(const NPUArray& a, int64_t kth, int64_t axis = -1);

} // namespace asnumpy
