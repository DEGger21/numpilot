"""领域自定义算子案例：最后一维的 inference-only layer normalization。

该 oracle 固定 ``epsilon=1e-5``，保留输入 shape，覆盖 mean/variance、广播和
数值稳定性；它是比逐元素案例更严格的手写 kernel 试验，但不声称已部署。
"""

import numpy as np

from numpilot import operator_spec


@operator_spec(
    name="layernorm",
    version="1.lastdim_eps1e-5",
    numpy_oracle="layernorm_oracle",
    target_api="asnumpy.layernorm",
    provenance="custom_generation_candidate",
    route="ascendc_registry_candidate",
    signature_variants=(
        {"positional_args": 1, "keyword_args": ()},
    ),
    dtype_rules=("float32",),
    shape_rules=("rank=2", "normalize_last_dimension", "contiguous"),
    source="numpilot://local-llm-generation/operator/layernorm",
    license="CANN-Open-Software-License-Agreement-2.0",
    target_soc="Ascend910B4",
    cann_version="9.0.0",
)
def layernorm_oracle(x):
    """Normalize each row with a fixed epsilon and no trainable parameters."""

    mean = np.mean(x, axis=1, keepdims=True)
    centered = np.subtract(x, mean)
    variance = np.mean(np.multiply(centered, centered), axis=1, keepdims=True)
    scale = np.sqrt(np.add(variance, np.float32(1e-5)))
    return np.divide(centered, scale)


def build_inputs():
    return (np.array([[1.0, 2.0, 3.0], [4.0, 4.0, 4.0]], dtype=np.float32),)


def forward(x):
    return layernorm_oracle(x)


if __name__ == "__main__":
    output = forward(*build_inputs())
    print("shape:", output.shape, "dtype:", output.dtype)
