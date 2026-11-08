# NumPilot 内部 AsNumpy 组件

这是作者维护的静态 AsNumpy 组件锁。NumPilot 只读取本目录中的 manifest、wheel、源码快照、LICENSE 和 SPDX SBOM，不从 Git、pip 或 latest 自动更新。后续版本由作者替换完整附件并同步更新全部哈希与 API 清单。

- repository: https://gitcode.com/cann/asnumpy.git
- upstream base commit: 60d139c54d9e181b8ae101dcbd1479d6b2a86439
- base source tree: 79e34b217f2e77c7d7b911105a1b56e24fd46898
- bundled content tree: f1c17d947a165c587a1b1d618d11d03fc5f5b183
- version: 0.3.1.dev70+g60d139c54.d20261108
- wheel: asnumpy-0.3.1.dev70+g60d139c54.d20261108-cp312-cp312-linux_aarch64.whl
- wheel SHA-256: d3ea03b2fd8c6eace56f532e2b30a917d63a138abe654c0020c14499898ed682
- source archive: asnumpy-source-60d139c54d9e181b8ae101dcbd1479d6b2a86439.tar.gz
- source archive SHA-256: f8b487cce5706b9c50e4e2492330cdf1e3a0857a4d2fc2e065f21c6bb207d9c1
- source files: 211
- source bytes: 1803983
- runtime: Linux AArch64 / CPython 3.12 / CANN 9.2.0-beta.2 / NumPy 2.5.4 / Loguru 0.7.3
- license: Apache-2.0

## 本次 MatMul 修复

新 AsNumpy 原生 C++/pybind11 接口已经存在，但 Python 公共入口原先把 `matmul` 全部转为主机 NumPy。当前静态组件包含三文件工作树补丁：

- `src/asnumpy/linalg/direct.py`：二维、C 连续、`float32` 且内维匹配时调用 `_core.matmul`；其它 dtype、形状、关键字调用保留兼容回退。
- `csrc/linalg/product.cpp`：ACLNN MatMul 使用 `cubeMathType=0`（`KEEP_DTYPE`），避免 FP32 被错误降精度。
- `tests/asnumpy_tests/linalg_tests/test_product.py`：原生派发、回退、精度和非连续输入测试。

验收：MatMul 定向测试 55 passed；AsNumpy 全量测试 1002 passed、32 xfailed、5 xpassed；wheel 在服务器 AArch64/CANN 9.2.0-beta.2 上构建并隔离安装验证。

源码快照由上游 `60d139c54d9e181b8ae101dcbd1479d6b2a86439` 工作树加上述补丁生成，`source-manifest.json` 记录逐文件哈希和补丁身份。普通安装仍只安装已校验 wheel，不自动编译源码。
