"""单缺口案例：``np.nextafter`` 的 IEEE-754 相邻值语义。

两个同形状 ``float32`` 输入让补全流程必须明确 dtype、shape、边界和逐元素
二元 ufunc 签名；原始 NumPy oracle 不执行任何随机或外部 I/O。
"""

import numpy as np


def build_inputs():
    source = np.array(
        [0.0, -0.0, np.inf, np.nan, 1.0],
        dtype=np.float32,
    )
    target = np.array(
        [1.0, 1.0, 0.0, 1.0, np.nan],
        dtype=np.float32,
    )
    return source, target


def forward(source, target):
    return np.nextafter(source, target)


if __name__ == "__main__":
    output = forward(*build_inputs())
    print(
        "shape:", output.shape,
        "dtype:", output.dtype,
        "nan_count:", int(np.isnan(output).sum()),
    )
