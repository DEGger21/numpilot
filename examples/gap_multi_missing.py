"""多缺口负例：两个独立 NumPy ufunc 不能未经拆分就交给一次生成。

``complete-gaps --provider prepare`` 会为两个调用分别建立有界 GapSpec；suite
准入则明确标记为 blocked，提醒使用者先拆分/审核每个算子的契约，再决定是否
启动昂贵的远端生成。
"""

import numpy as np


def build_inputs():
    return (np.linspace(-4.0, 4.0, 257, dtype=np.float32),)


def forward(x):
    return np.add(np.cbrt(x), np.spacing(x))


if __name__ == "__main__":
    print("finite:", bool(np.isfinite(forward(*build_inputs())).all()))
