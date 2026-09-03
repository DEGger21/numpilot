"""纯 NumPy 直通候选：广播式逐元素能量与稳定激活。

这个案例刻意保持 suite contract 的最小形态：输入只由 ``linspace`` 构造，
``forward`` 只调用当前 AsNumpy 覆盖集中的逐元素算子。它适合作为 DIRECT
基线，与下面的缺口案例和自定义 OperatorSpec 案例对照。
"""

import numpy as np


def build_inputs():
    x = np.linspace(-2.0, 2.0, 4096, dtype=np.float32)
    y = np.linspace(2.0, -2.0, 4096, dtype=np.float32)
    return x, y


def forward(x, y):
    energy = np.add(np.square(x), np.square(y))
    return np.tanh(np.sqrt(energy))


if __name__ == "__main__":
    output = forward(*build_inputs())
    print(
        "shape:", output.shape,
        "dtype:", output.dtype,
        "finite:", bool(np.isfinite(output).all()),
    )
