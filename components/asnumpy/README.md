# NumPilot 内部 AsNumpy 组件

本目录的 `manifest.json` 是 NumPilot 当前发行线的 AsNumpy 锁。它固定了

- 上游 repository、commit、tree 和版本；
- 目标 Python/平台 wheel 文件名、tag 和 SHA-256；
- 构建关键文件摘要及运行时约束；
- 发布 wheel、上游 LICENSE 与 SPDX SBOM 的逐文件摘要。

> 当前目录已经包含 manifest 指向的 wheel、上游许可证和 SPDX SBOM。wheel 是针对 Linux
> AArch64/CPython 3.10/CANN 9.0.0 的固定二进制；发布前仍须在目标 CodeArts/Ascend 环境复核
> CANN 动态库与设备资格，不能把它当作 x86 或其他 Python ABI 的通用 wheel。

运行时约束包括 CANN 9.0.0、Linux AArch64、CPython 3.10、NumPy 2.2.6 和 Loguru 0.7.3；
安装前会把这些值与当前环境逐项比较。由于 `_core` 是 Linux AArch64/CANN 原生扩展，wheel 本体作为同一 release 的受控附件放在
本目录（或通过 `NUMPILOT_ASNUMPY_COMPONENT_DIR` 指向附件目录）。如果附件缺失，安装必须失败
关闭。安装器不会 `git clone`、`git fetch`、解析
`latest`，也不会执行 `pip install -U asnumpy`。

预期目录：

```text
components/asnumpy/
  manifest.json
  README.md
  LICENSE
  SBOM.spdx.json
  asnumpy-0.3.1.dev4+g1929a3ba4-cp310-cp310-linux_aarch64.whl
```

使用发布附件安装：

```bash
python -m numpilot bootstrap --apply --asnumpy-component \
  components/asnumpy \
  --receipt /secure/private/bootstrap/receipt.json
python -m numpilot bootstrap --verify --json
```

冻结提交前，在仓库根目录运行：

```bash
python tools/check_release_tree.py --require-asnumpy-component --json
```

命令会先读取清单，校验 LICENSE/SBOM，再按清单中的**精确文件名和 SHA-256**校验 wheel，
最后执行安装后的 README smoke 和 native 文件身份核对。版本、tag、元数据、receipt 或运行时
文件任一漂移都会保留 `failed` receipt 并停止；不会退回网络或源码路径。

日常开发可以只运行不带该选项的审计；缺少平台 wheel 时会明确给出 warning，不会伪造已固化状态。

`--build-asnumpy` 仅供维护者在隔离构建机生成下一版候选，不能作为用户安装流程。候选通过
完整测试后，应生成新的 component lock、wheel、SBOM 和 release manifest，而不是覆盖本锁。
