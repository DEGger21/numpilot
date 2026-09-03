"""领域自定义算子案例：让 LLM 按 OperatorSpec 手写 ``scaled_l2_loss``。

这里的 NumPy 函数是权威 oracle；装饰器只声明目标 AsNumpy API、签名、dtype、
shape、SoC/CANN 身份和生成路由。NumPilot 不会导入或执行源码来决定路由，而是
从 AST 提取契约，再把候选草稿留在隔离目录中等待人工/自动六门资格化。
"""

import numpy as np

from numpilot import operator_spec


@operator_spec(
    name="scaled_l2_loss",
    version="1.mean",
    numpy_oracle="scaled_l2_loss_oracle",
    target_api="asnumpy.scaled_l2_loss",
    provenance="custom_generation_candidate",
    route="ascendc_registry_candidate",
    signature_variants=(
        {"positional_args": 2, "keyword_args": ()},
    ),
    dtype_rules=("float32",),
    shape_rules=("rank>=1", "same_shape", "contiguous"),
    source="numpilot://local-llm-generation/operator/scaled_l2_loss",
    license="CANN-Open-Software-License-Agreement-2.0",
    target_soc="Ascend910B4",
    cann_version="9.0.0",
)
def scaled_l2_loss_oracle(predict, label):
    """返回逐元素平方误差的 mean，作为唯一数值 oracle。"""

    error = np.subtract(predict, label)
    squared = np.multiply(error, error)
    return np.mean(squared)


def build_inputs():
    predict = np.linspace(-1.0, 1.0, 257, dtype=np.float32)
    label = np.linspace(1.0, -1.0, 257, dtype=np.float32)
    return predict, label


def forward(predict, label):
    return scaled_l2_loss_oracle(predict, label)


if __name__ == "__main__":
    print("loss:", float(forward(*build_inputs())))
