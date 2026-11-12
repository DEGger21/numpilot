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
#include <vector>

namespace asnumpy {

/**
 * @brief Compute the NumPy-compatible ``pad`` result on the NPU.
 *
 * Admitted profile: rank 2 float32 input, ``mode='constant'`` with constant
 * value 0, non-negative pad widths.  Any other dtype, rank, mode or negative
 * pad width is rejected explicitly by the integration layer. No implicit cast
 * and no CPU fallback are provided.
 */
NPUArray Pad(const NPUArray& a, const std::vector<int64_t>& padWidth);

} // namespace asnumpy
