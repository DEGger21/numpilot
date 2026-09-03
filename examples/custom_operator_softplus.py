"""领域自定义算子案例：稳定的逐元素 softplus。

该案例用于观察模型是否能把一个带数值稳定分支的 NumPy oracle 映射到
Ascend C。契约只允许 float32、同形状连续输入；生成物仍必须经过完整资格化。
"""

import numpy as np

from numpilot import operator_spec


@operator_spec(
    name="softplus",
    version="1.stable",
    numpy_oracle="softplus_oracle",
    target_api="asnumpy.softplus",
    provenance="custom_generation_candidate",
    route="ascendc_registry_candidate",
    signature_variants=(
        {"positional_args": 1, "keyword_args": ()},
    ),
    dtype_rules=("float32",),
    shape_rules=("rank>=1", "same_shape", "contiguous"),
    source="numpilot://local-llm-generation/operator/softplus",
    license="CANN-Open-Software-License-Agreement-2.0",
    target_soc="Ascend910B4",
    cann_version="9.0.0",
)
def softplus_oracle(x):
    """计算 ``log1p(exp(-abs(x))) + maximum(x, 0)``，避免溢出。"""

    magnitude = np.absolute(x)
    return np.add(np.log1p(np.exp(-magnitude)), np.maximum(x, 0.0))


def build_inputs():
    return (np.array([-80.0, -4.0, 0.0, 4.0, 80.0], dtype=np.float32),)


def forward(x):
    return softplus_oracle(x)


if __name__ == "__main__":
    output = forward(*build_inputs())
    print("shape:", output.shape, "dtype:", output.dtype)
