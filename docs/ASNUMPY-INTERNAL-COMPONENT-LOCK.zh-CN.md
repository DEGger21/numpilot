# AsNumpy 内部组件锁定与发布合同

本文定义 NumPilot 发布件如何交付、安装和验证 AsNumpy。当前版本仍处于 Alpha 阶段；该状态
不放宽组件完整性、许可证或运行时资格要求。当前发布件同时保存默认运行 wheel 和已扩展 MatMul 的
完整源码基线，源码只供作者审阅、修改和显式重建。`components/asnumpy/` 中的 manifest、wheel、源码归档、source-manifest、`LICENSE` 和
`SBOM.spdx.json` 共同构成不可分割的组件发布件。

## 1. 交付边界

AsNumpy 是 NumPilot 的锁定运行时组件，不是可由依赖解析器自由升级的浮动依赖。默认安装路径
只读取本地 `components/asnumpy/`，先校验 manifest、wheel、源码归档和附件，再安装其中指定的精确 wheel。
它不访问 PyPI 或上游 Git，也不安装 Ascend 驱动、固件、CANN 或 OPP。

发布路径必须拒绝以下行为：

- 解析 `latest`、版本范围、浮动分支或未登记的 wheel；
- 在组件缺失或哈希不匹配时自动 `git clone`、`git fetch` 或下载 AsNumpy 源码；
- 静默执行 `pip install -U asnumpy`，或允许依赖解析器替换已校验 wheel；
- 接受 wheel、`METADATA`、`RECORD`、Python 模块、native `_core`、CANN、SoC 或 Python ABI 漂移。

维护者可以显式使用 `--build-asnumpy` 或 `install.sh --asnumpy-source` 构建下一版组件；该路径
必须与用户安装隔离，并产生新的不可变身份。默认组件安装若失败，必须阻断，不能退回源码下载或自动从源码编译。

## 2. 锁定身份

`components/asnumpy/manifest.json` 是机器可读的权威来源。当前发布件绑定以下身份：

| 字段 | 锁定值 |
|---|---|
| Component ID | `asnumpy-0.3.1.dev70+g60d139c54.d20261108-cp312-cp312-linux_aarch64` |
| 上游仓库 | `https://gitcode.com/cann/asnumpy.git` |
| Git commit | `60d139c54d9e181b8ae101dcbd1479d6b2a86439` |
| Git tree | `f1c17d947a165c587a1b1d618d11d03fc5f5b183` |
| 包版本 | `0.3.1.dev70+g60d139c54.d20261108` |
| 许可证 | `Apache-2.0` |
| 构建方式 | `source-commit-with-fp32-aclnn-matmul-dispatch-overlay` |
| 构建工具链 | `GCC 11.4.0`、CANN `9.2.0-beta.2`、CPython `3.12.14` |
| 目标 wheel | `asnumpy-0.3.1.dev70+g60d139c54.d20261108-cp312-cp312-linux_aarch64.whl` |
| Wheel tag | `cp312-cp312-linux_aarch64` |
| Wheel SHA-256 | `d3ea03b2fd8c6eace56f532e2b30a917d63a138abe654c0020c14499898ed682` |
| 源码归档 | `asnumpy-source-60d139c54d9e181b8ae101dcbd1479d6b2a86439.tar.gz` |
| 源码归档 SHA-256 | `f8b487cce5706b9c50e4e2492330cdf1e3a0857a4d2fc2e065f21c6bb207d9c1` |
| source-manifest SHA-256 | `408337443482fb9b614a83b8933d640192de15937ce7ccd440bf7b00218a7fa8` |
| 源码清单 | 211 个文件，1,803,983 bytes |
| Wheel `METADATA` SHA-256 | `3dc7649033a9820a14f48372b77e38c5d5c48e7c094645d004b8530a0b4994c5` |
| Wheel `RECORD` SHA-256 | `628ee08761514a6aa7c7eca39184ce66dfa99456b3c60d0278757b9c58556f16` |
| `asnumpy/__init__.py` SHA-256 | `295976c53a4fccc2807f9a3ef81821c063f839e6b32e4150d19288ee7af9224c` |
| `asnumpy/_core.cpython-312-aarch64-linux-gnu.so` SHA-256 | `44c5b349b3f8ce16b4d45157a3c39b19ec4a3edfba3aa1687d43081077b9e511` |
| 组件 LICENSE SHA-256 | `1eb85fc97224598dad1852b5d6483bbcf0aa8608790dcc657a5a2a761ae9c8c6` |
| 组件 SBOM SHA-256 | `1603bed4cf56e23f94e462e8b7fc25343212d60ce62d9a4b5748914900dc534f` |
| 组件 manifest SHA-256 | `ff21a9842bb18897a760a5088e46f7710b3f3f0ed10f4b215079355417a7694b` |

Manifest 的运行约束为 Linux/AArch64、CPython 3.12、CANN 9.2.0-beta.2、NumPy 2.5.4 和 Loguru 0.7.3。
NumPilot 当前严格画像进一步绑定 Ascend 910B3、CPython 3.12.14 和精确 CANN 身份。画像外环境
只能返回 `CAPABILITY_AUDIT_REQUIRED`，不能继承 strict-NPU 或性能结论。

组件还绑定以下构建关键文件摘要：

```text
CMakeLists.txt                  e5da3e46e13e2e5936c4e3e03af526b1638f13ae6c73462829abaff1ade1b956
LICENSE                         1eb85fc97224598dad1852b5d6483bbcf0aa8608790dcc657a5a2a761ae9c8c6
README.md                       3a395536661cc2286937baa50aad68033e77748bb9c5c30ba5a8bd1d110cce9a
bindings/python/CMakeLists.txt  6a39bb565d5a4762daaa48217198b3f40296b6a431fa37e924f9784b31f7d1d5
csrc/CMakeLists.txt             3f5ca526867de88dbefc8912d9563053ad78b6e3bc0b58630baacbc37ae281b6
pyproject.toml                  ed92ce53169bcd819dadc34658be54ff778919f277b9a83392d91110a818d6ac
```

Git commit/tree 与 wheel 字节是两层独立合同。同一源码可能因编译器、CANN、链接器或构建环境
不同而产生不同二进制；发布件必须保存实际 wheel，不能仅凭源码 commit 推断二进制身份。

## 3. 安装与验证合同

推荐通过发布件中的组件目录安装：

```bash
python -m numpilot bootstrap --apply \
  --asnumpy-component /absolute/path/to/components/asnumpy \
  --receipt /absolute/private/bootstrap-receipt.json
python -m numpilot bootstrap --verify --json
```

已有可信 wheel 也可显式使用成对参数 `--asnumpy-wheel PATH` 和
`--asnumpy-wheel-sha256 HEX`。两种路径都必须在调用 pip 前验证 wheel 字节；组件目录路径还必须
验证 manifest、LICENSE 和 SBOM。任何不匹配都必须产生失败回执，不能退回源码下载或自动换包。

安装后必须再次核对 distribution metadata、`asnumpy/__init__.py`、所有 native 模块的哈希，
并执行固定的 `(1000, 1000) float32` AsNumpy smoke。完整身份链为：

```text
manifest + LICENSE + SBOM
  → wheel 字节 SHA-256
  → wheel METADATA/RECORD SHA-256
  → 安装后 Python/native 文件 SHA-256
  → bootstrap/qualification receipt
  → NumPilot release manifest
```

组件缺失、附件不完整、哈希漂移、运行时不匹配或 smoke 失败时，安装和资格验证必须失败关闭。

源码归档和 source-manifest 也必须在发布验收时校验。它们用于维护者重建，不会在普通安装中
自动解包或编译；源码清单逐文件绑定 commit、tree、路径、SHA-256、文件数和总字节数。

## 4. 升级、回滚与安全修复流程

AsNumpy 升级是一次新的组件发布：

1. 从发布件中的源码归档提取当前基线，或在隔离环境获取同一 upstream commit/tag；校验 origin、commit、tree、许可证和逐文件 source-manifest；
2. 在目标 CANN/SoC/Python 矩阵构建 wheel，记录工具链、环境和新的 artifact SHA-256；
3. 运行 AsNumpy 测试、NumPilot API/数值/strict-NPU 回归及适用的算子接入回归；
4. 更新 component manifest、锁定文档、SBOM、许可证清单和资格回执；
5. 在第二个干净环境仅从新 wheel 安装，完成 import、smoke、工作负载重扫和回滚演练；
6. 通过发布评审后创建不可移动的 NumPilot tag，并随发布说明记录兼容性和回滚版本。

安全修复遵循同一流程。回滚只能选择已保存、身份完整且未撤回的旧组件；禁止将“重新获取上游
当前版本”作为回滚，也禁止在用户环境中后台替换 NumPilot 已绑定的 AsNumpy 身份。

## 5. SBOM 与许可证

- AsNumpy 的 `Apache-2.0` 许可证文本必须随 wheel 或组件发布件保留；NumPilot 的许可证不会
  重新授权上游或第三方代码。
- CANN、OPP、ACLNN、驱动、CANNBot 和 ops-nn 分别适用各自许可证，不能因为与组件共同交付
  就视为 NumPilot 自有代码。
- `SBOM.spdx.json` 使用 SPDX-2.3；组件 manifest 必须绑定其文件名和 SHA-256。
- Wheel 若包含工具链生成物或第三方库，SBOM 和许可证清单必须记录来源、版本、路径和摘要，
  并在公开分发前确认再分发权。
- Token、私有 endpoint、内部 checkout、模型响应、运行状态和未筛选构建目录不得进入发布件。

发布前应运行发布树与制品清单检查，确认 manifest、wheel、LICENSE、SBOM 和安装后身份全部
一致；上游仓库或网络不可用时，已承诺的本地组件安装路径仍必须可执行。

## 6. 相关文档

- [安装与验收](installation.zh-CN.md)
- [源码基线与缺口扩展流程](asnumpy-source-extension.zh-CN.md)
- [产物存储策略](artifact-storage.md)
- [第三方软件与材料](../THIRD_PARTY_NOTICES.md)
