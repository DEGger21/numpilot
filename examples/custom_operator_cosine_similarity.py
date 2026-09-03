"""领域自定义算子案例：按行计算 cosine similarity。

这是一个带 reduction 的二维算子，输出 shape 从 ``(batch, features)`` 变为
``(batch,)``，用于测试模型对 shape/infer-shape、epsilon 和零范数语义的处理。
"""

import numpy as np

from numpilot import operator_spec


@operator_spec(
    name="cosine_similarity",
    version="1.rows_eps1e-6",
    numpy_oracle="cosine_similarity_oracle",
    target_api="asnumpy.cosine_similarity",
    provenance="custom_generation_candidate",
    route="ascendc_registry_candidate",
    signature_variants=(
        {"positional_args": 2, "keyword_args": ()},
    ),
    dtype_rules=("float32",),
    shape_rules=("rank=2", "same_batch", "contiguous"),
    source="numpilot://local-llm-generation/operator/cosine_similarity",
    license="CANN-Open-Software-License-Agreement-2.0",
    target_soc="Ascend910B4",
    cann_version="9.0.0",
)
def cosine_similarity_oracle(x, y):
    """Return a row-wise dot product divided by norms, clamped by epsilon."""

    dot = np.sum(np.multiply(x, y), axis=1)
    norm_x = np.sqrt(np.sum(np.multiply(x, x), axis=1))
    norm_y = np.sqrt(np.sum(np.multiply(y, y), axis=1))
    denominator = np.maximum(np.multiply(norm_x, norm_y), np.float32(1e-6))
    return np.divide(dot, denominator)


def build_inputs():
    x = np.array([[1.0, 0.0, 0.0], [1.0, 2.0, 2.0]], dtype=np.float32)
    y = np.array([[0.0, 1.0, 0.0], [2.0, 0.0, 1.0]], dtype=np.float32)
    return x, y


def forward(x, y):
    return cosine_similarity_oracle(x, y)


if __name__ == "__main__":
    output = forward(*build_inputs())
    print("shape:", output.shape, "dtype:", output.dtype)
