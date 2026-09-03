"""应当 fail-closed 的负例：嵌套命名空间和动态 dispatch 不可生成。

``np.fft.fft`` 不是当前直接 ``numpy.<api>`` 缺口契约，``getattr`` 也无法在
静态阶段证明目标函数。即使同一函数里还出现可覆盖的 ``real/add``，整体案例
也必须保留阻断项，不能把不完整的语义交给 LLM 猜测。
"""

import numpy as np


def build_inputs():
    return (np.linspace(-1.0, 1.0, 64, dtype=np.float32),)


def forward(x):
    spectrum = np.fft.fft(x)
    dynamic_cbrt = getattr(np, "cbrt")(x)
    return np.add(np.real(spectrum), dynamic_cbrt)


if __name__ == "__main__":
    print(forward(*build_inputs()))
