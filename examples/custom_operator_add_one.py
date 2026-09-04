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

"""最小自定义算子缺口：对连续 float32 输入逐元素加一。

这个案例刻意把语义缩到一个一元 ``y = x + 1`` 操作，便于观察
``resolve-gaps → CANNBot Skill → 边界模型草稿`` 的真实路由。生成候选仍然
只是 ``generated_unverified``，必须经过独立的编译、数值和 NPU 门禁后才能
进入后续流程。
"""

import numpy as np

from numpilot import operator_spec


@operator_spec(
    name="add_one",
    version="1.elementwise",
    numpy_oracle="add_one_oracle",
    target_api="asnumpy.add_one",
    provenance="custom_generation_candidate",
    route="ascendc_registry_candidate",
    signature_variants=(
        {"positional_args": 1, "keyword_args": ()},
    ),
    dtype_rules=("float32",),
    shape_rules=("rank>=1", "same_shape", "contiguous"),
    source="numpilot://local-llm-generation/operator/add_one",
    license="CANN-Open-Software-License-Agreement-2.0",
    target_soc="Ascend910B4",
    cann_version="9.0.0",
)
def add_one_oracle(x):
    """逐元素计算 ``y = x + 1.0f``，不改变输入。"""

    return np.add(x, np.float32(1.0))


def build_inputs():
    return (np.array([-2.0, 0.0, 3.5], dtype=np.float32),)


def forward(x):
    return add_one_oracle(x)


if __name__ == "__main__":
    output = forward(*build_inputs())
    print("shape:", output.shape, "dtype:", output.dtype, "values:", output)
