# examples/mlp_forward.py
"""待迁移示例:小型 MLP 前向推理。仅使用 AsNumpy 覆盖集算子。

计算主体 forward() 全程使用 np.<op>(覆盖集算子), 便于 NumPilot 的 Analyzer
提取调用契约、Planner 选择直迁路线、Code Migrator 把 np.<op> 改写为 ap.<op>。
验证阶段分别执行原始(np)与迁移后(ap, 由后端提供 asnumpy 模块)源码并比对数值一致性。
"""
import numpy as np


def build_inputs():
    # 用覆盖集内的 linspace/zeros 构造确定性输入(避免不可迁移的 np.random)。
    x = np.linspace(-1.0, 1.0, 64 * 784, dtype=np.float32).reshape(64, 784)
    W1 = np.linspace(-0.02, 0.02, 784 * 256, dtype=np.float32).reshape(784, 256)
    b1 = np.zeros((256,), dtype=np.float32)
    W2 = np.linspace(-0.02, 0.02, 256 * 10, dtype=np.float32).reshape(256, 10)
    b2 = np.zeros((10,), dtype=np.float32)
    return x, W1, b1, W2, b2


def forward(x, W1, b1, W2, b2):
    h = np.maximum(np.add(np.matmul(x, W1), b1), 0)      # 线性层 + relu
    logits = np.add(np.matmul(h, W2), b2)
    m = np.max(logits, axis=-1, keepdims=True)
    e = np.exp(np.subtract(logits, m))
    s = np.sum(e, axis=-1, keepdims=True)
    return np.divide(e, s)                                # softmax(数值稳定)


if __name__ == "__main__":
    out = forward(*build_inputs())
    print("output shape:", out.shape, "row0 sum ~1:", float(out[0].sum()))
