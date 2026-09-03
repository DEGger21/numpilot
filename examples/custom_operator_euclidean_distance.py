"""简单领域算子：让 LLM 按契约实现逐元素欧氏距离。

相较于带 reduction 的损失，这个 oracle 只有同形状逐元素操作，便于审查
广播、dtype、输出 shape 与 Ascend C kernel 的一一对应关系。
"""

import numpy as np

from numpilot import operator_spec


@operator_spec(
    name="euclidean_distance",
    version="1.elementwise",
    numpy_oracle="euclidean_distance_oracle",
    target_api="asnumpy.euclidean_distance",
    provenance="custom_generation_candidate",
    route="ascendc_registry_candidate",
    signature_variants=(
        {"positional_args": 2, "keyword_args": ()},
    ),
    dtype_rules=("float32",),
    shape_rules=("rank>=1", "same_shape", "contiguous"),
    source="numpilot://local-llm-generation/operator/euclidean_distance",
    license="CANN-Open-Software-License-Agreement-2.0",
    target_soc="Ascend910B4",
    cann_version="9.0.0",
)
def euclidean_distance_oracle(x, y):
    """逐元素 ``sqrt((x-y)^2)`` NumPy oracle，不改变输入。"""

    delta = np.subtract(x, y)
    return np.sqrt(np.square(delta))


def build_inputs():
    x = np.linspace(-1.0, 1.0, 257, dtype=np.float32)
    y = np.linspace(1.0, -1.0, 257, dtype=np.float32)
    return x, y


def forward(x, y):
    return euclidean_distance_oracle(x, y)


if __name__ == "__main__":
    output = forward(*build_inputs())
    print("shape:", output.shape, "dtype:", output.dtype, "max:", float(output.max()))
