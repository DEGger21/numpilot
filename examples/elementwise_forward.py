"""strict-NPU smoke：不含当前审计已知 CPU fallback 的逐元素网络。"""
import numpy as np


def build_inputs():
    x = np.linspace(-2.0, 2.0, 4096, dtype=np.float32).reshape(64, 64)
    y = np.linspace(2.0, -2.0, 4096, dtype=np.float32).reshape(64, 64)
    return x, y


def forward(x, y):
    energy = np.add(np.square(x), np.square(y))
    return np.tanh(np.sqrt(energy))


if __name__ == "__main__":
    output = forward(*build_inputs())
    if hasattr(output, "to_numpy"):
        output = output.to_numpy()
    print("output shape:", output.shape, "finite:", bool(np.isfinite(output).all()))
