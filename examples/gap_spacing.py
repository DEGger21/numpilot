"""单缺口案例：``np.spacing`` 的浮点间距语义。

这个案例比普通逐元素算子更强调 IEEE 边界（零点、负数和次正规数），因此
适合观察 LLM 草稿能否把语义约束写进算子契约；生成物仍必须经过独立资格化。
"""

import numpy as np


def build_inputs():
    return (np.linspace(-4.0, 4.0, 257, dtype=np.float32),)


def forward(x):
    return np.spacing(x)


if __name__ == "__main__":
    output = forward(*build_inputs())
    print("shape:", output.shape, "dtype:", output.dtype, "finite:", bool(np.isfinite(output).all()))
