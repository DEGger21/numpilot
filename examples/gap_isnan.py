"""单缺口案例：``np.isnan`` 的布尔输出算子补全。

``isnan`` 是标准 NumPy 一元 ufunc，但不在当前固定 AsNumpy 覆盖快照中。
它专门测试模型生成时的 bool 输出、NaN 边界和一元签名约束。
"""

import numpy as np


def build_inputs():
    # Include finite, NaN and +Inf values so the bool-output oracle exercises
    # the exceptional-value branch without relying on runtime-generated data.
    return (np.array(
        [0.0, np.nan, np.inf, -1.0, 1.0],
        dtype=np.float32,
    ),)


def forward(x):
    return np.isnan(x)


if __name__ == "__main__":
    output = forward(*build_inputs())
    print("shape:", output.shape, "dtype:", output.dtype, "nan_count:", int(output.sum()))
