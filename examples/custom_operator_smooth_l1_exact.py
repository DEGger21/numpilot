"""已被官方 ACLNN 目录审计的领域算子对照案例。

该契约与当前 ``Ascend910B4/CANN 9.0.0`` 的 ``operator.smooth_l1_loss`` 映射
一致。官方优先解析若发现已安装符号，应走 adapter candidate，而不是让 LLM
重复手写 kernel；只有语义资格化和 adapter 集成完成后才可执行。
"""

import numpy as np

from numpilot import operator_spec


@operator_spec(
    name="smooth_l1_loss",
    version="2.mean.beta1",
    numpy_oracle="smooth_l1_loss_oracle",
    target_api="asnumpy.smooth_l1_loss",
    provenance="custom_generation_candidate",
    route="ascendc_registry_candidate",
    signature_variants=(
        {"positional_args": 2, "keyword_args": ()},
    ),
    dtype_rules=("float32",),
    shape_rules=("rank>=1", "same_shape", "contiguous"),
    source="numpilot://local-llm-generation/operator/smooth_l1_loss",
    license="CANN-Open-Software-License-Agreement-2.0",
    target_soc="Ascend910B4",
    cann_version="9.0.0",
)
def smooth_l1_loss_oracle(predict, label):
    error = np.subtract(predict, label)
    absolute = np.absolute(error)
    bounded = np.minimum(absolute, np.float32(1.0))
    quadratic = np.divide(np.multiply(bounded, bounded), np.float32(2.0))
    linear = np.subtract(absolute, bounded)
    return np.mean(np.add(quadratic, linear))


def build_inputs():
    predict = np.linspace(-1.0, 1.0, 257, dtype=np.float32)
    label = np.linspace(1.0, -1.0, 257, dtype=np.float32)
    return predict, label


def forward(predict, label):
    return smooth_l1_loss_oracle(predict, label)


if __name__ == "__main__":
    print("loss:", float(forward(*build_inputs())))
