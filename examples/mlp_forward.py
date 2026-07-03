# examples/mlp_forward.py
"""待迁移示例:小型 MLP 前向推理。仅使用 AsNumpy 覆盖集算子。

forward() 接受 xp 参数(numpy 或迁移后的 asnumpy 别名), 以便验证阶段
在同一份计算逻辑上对比双端结果。真实迁移由 NumPilot 的 AST 引擎完成,
本文件的 xp 注入只是为验证提供等价参照。
"""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((64, 784)).astype(np.float32)
    W1 = rng.standard_normal((784, 256)).astype(np.float32) * 0.02
    b1 = np.zeros((256,), dtype=np.float32)
    W2 = rng.standard_normal((256, 10)).astype(np.float32) * 0.02
    b2 = np.zeros((10,), dtype=np.float32)
    return x, W1, b1, W2, b2


def forward(x, W1, b1, W2, b2, *, xp=np):
    h = xp.add(xp.matmul(x, W1), b1)
    h = xp.maximum(h, 0)                     # relu
    logits = xp.add(xp.matmul(h, W2), b2)
    e = xp.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


if __name__ == "__main__":
    out = forward(*build_inputs())
    print("output shape:", out.shape, "row sums ~1:", np.allclose(out.sum(-1), 1, atol=1e-4))
