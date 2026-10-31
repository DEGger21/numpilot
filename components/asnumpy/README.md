# NumPilot 内部 AsNumpy 组件

这是作者维护的静态 AsNumpy 组件锁，固定本次比赛使用的 MatMul 接入版本。NumPilot
只读取本目录中的 manifest、wheel、LICENSE 和 SPDX SBOM，不从 Git、pip 或 latest
自动更新。后续版本由作者替换完整附件并同步更新所有哈希与 API 清单。

- repository: https://gitcode.com/cann/asnumpy.git
- source commit: 1e7177f12039b2d87e3cdff6693459d895744f68
- source tree: a5fb1f3a70be9fc17ecf0dd928776924d4c78e1e
- version: 0.4.1.dev12+gc9115e8ce.d20261031
- wheel: asnumpy-0.4.1.dev12+gc9115e8ce.d20261031-cp312-cp312-linux_aarch64.whl
- wheel SHA-256: a8c47cfe71994f3bfd4180d7354df3904d2bc96c476c3119083fefac747a2138
- manifest SHA-256: 8bd7893a6a77df35d7c1a1f05d1144b2c49ed04a8a8f93afd9b5d4e53e93a539
- runtime: Linux AArch64 / CPython 3.12 / CANN 9.2.0-beta.2 / NumPy 2.5.4 / Loguru 0.7.3
- license: Apache-2.0

当前公开 matmul 的严格 NPU admission 为：两个位置参数、AsNumpy ndarray、二维、非空、
C 连续、float32、内维匹配。该 profile 调用 ACLNN aclnnMatmul，并使用
cubeMathType=0 (KEEP_DTYPE)。其它 dtype/shape 保留兼容回退，详见
docs/asnumpy-api-support.zh-CN.md。
