# examples/big_gemm.py
"""待迁移示例:大规模 GEMM 链(多层全连接 + relu),用于凸显 NPU 相对 numpy@CPU 的优势。

设计要点:
- **规模非常大**:方阵边长 _N(默认 2048),forward 为 depth 层 matmul,计算量以 N^3
  增长 —— 矩阵乘正是 NPU(Cube 单元)最擅长、相对 CPU BLAS 优势最明显的负载。
- **只用覆盖集算子**:matmul + relu。relu 以 numpy 惯用法 np.maximum(x, 0) 表达,
  由 NumPilot 的 AST 引擎自动改写为 ap.relu(真机 asnumpy 的二元算子不接受标量 0)。
- **确定性输入**:build_inputs() 全用 np.linspace 造,可复现;W 采用类 He 初始化
  尺度 1/sqrt(N),保证逐层数值稳定(不溢出、输出量级 O(1)),使精度对比有意义。
"""
import numpy as np

_N = 2048          # 方阵边长;越大 NPU 相对 CPU 优势越明显(数据量 ~ N^2,算力 ~ N^3)


def build_inputs():
    n = _N
    scale = float(1.0 / np.sqrt(n))     # 类 He 初始化,保持逐层数值稳定
    x = np.linspace(-1.0, 1.0, n * n, dtype=np.float32).reshape(n, n)
    W1 = np.linspace(-scale, scale, n * n, dtype=np.float32).reshape(n, n)
    W2 = np.linspace(scale, -scale, n * n, dtype=np.float32).reshape(n, n)
    W3 = np.linspace(-scale, scale, n * n, dtype=np.float32).reshape(n, n)
    W4 = np.linspace(scale, -scale, n * n, dtype=np.float32).reshape(n, n)
    return x, W1, W2, W3, W4


def forward(x, W1, W2, W3, W4):
    h = np.maximum(np.matmul(x, W1), 0)     # 全连接 + relu
    h = np.maximum(np.matmul(h, W2), 0)
    h = np.maximum(np.matmul(h, W3), 0)
    return np.matmul(h, W4)                  # 输出层(不接 relu)


if __name__ == "__main__":
    out = forward(*build_inputs())
    print("output shape:", out.shape,
          "max|out|:", float(np.absolute(out).max()),
          "mean:", float(out.mean()))
