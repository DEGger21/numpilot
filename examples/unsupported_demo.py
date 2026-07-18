# examples/unsupported_demo.py
"""负例:使用缺口算子(np.fft.fft), NumPilot 应标记 unsupported 并终止迁移。"""
import numpy as np


def build_inputs():
    return (np.ones(8, dtype=np.float32),)


def forward(x):
    return np.fft.fft(x)


if __name__ == "__main__":
    print(forward(*build_inputs()))
