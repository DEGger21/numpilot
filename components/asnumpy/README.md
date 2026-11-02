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
- manifest SHA-256: 9b4e95fea4db4ce897c229f0f8b92e04c0cfc9dad9863275730a3f6edaf39db5
- runtime: Linux AArch64 / CPython 3.12 / CANN 9.2.0-beta.2 / NumPy 2.5.4 / Loguru 0.7.3
- license: Apache-2.0

当前公开 matmul 的严格 NPU admission 为：两个位置参数、AsNumpy ndarray、二维、非空、
C 连续、float32、内维匹配。该 profile 调用 ACLNN aclnnMatmul，并使用
cubeMathType=0 (KEEP_DTYPE)。其它 dtype/shape 保留兼容回退，详见
docs/asnumpy-api-support.zh-CN.md。


## 源码基线

当前目录还提供完整的作者维护源码快照，供缺口修复和显式重建使用：

- archive: `asnumpy-source-1e7177f12039b2d87e3cdff6693459d895744f68.tar.gz`
- source manifest: `source-manifest.json`
- source files: 198
- source bytes: 1691974
- archive SHA-256: `9a484272ad540879a93751e415c8adb10b52edfaab50e050672d3abd9cc6544d`
- source manifest SHA-256: `f697292d223bee0b6030a32ccc5bc0f2ba1d096d59b9e5a7efc148df6b0079b4`

源码只由作者维护流程提取、修改、测试和重建；普通安装仍只安装已校验 wheel，不自动编译源码。
具体流程见 `docs/asnumpy-source-extension.zh-CN.md`。
