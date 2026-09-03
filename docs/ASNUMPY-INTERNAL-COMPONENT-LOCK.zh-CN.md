# AsNumpy 内部组件锁定与发布合同

本文定义 NumPilot 发布时如何消费 AsNumpy。它是**发布合同**，不是把开发机上的二进制或
上游源码 checkout 伪装成已经随仓库发行的文件。当前工作树仍是 Alpha 开发快照（整体仍处于
开发态）；AsNumpy
组件附件已经补齐并接入本地哈希校验，但最终比赛/RC 仍须在干净提交和第二环境复核。

> **当前附件状态（2026-09-03）**：`components/asnumpy/` 已提供精确文件名的 wheel、上游
> `LICENSE`、SPDX-2.3 `SBOM.spdx.json` 和 manifest。当前交付 wheel SHA-256 为
> `8d4dbd08…`，安装后的 native `_core` SHA-256 为 `4166713f…`；旧回执中的
> `9822af6a…`（原 wheel 字节已丢失）和公开 evidence 的 `d8a107c1…` 仅作历史参考，不能混用。
> 因而在匹配的 Linux AArch64/CPython 3.10/CANN 9.0.0 环境中不再需要现场拉取 AsNumpy 源码；
> 仍须从本地附件执行一次受哈希保护的安装，且 CANN/驱动本身不由本组件提供。

## 1. 决策与边界

AsNumpy 是 NumPilot 的运行时内部组件，不是一个可以随意升级的浮动 Python 依赖。一个可发布
的 NumPilot 版本必须选择下面一种、且只能选择一种交付形态：

1. **受控 wheel bundle（推荐）**：在 NumPilot 发行附件旁边提供一份经过哈希校验的 AsNumpy
   wheel；安装器只从该本地文件安装，不访问 PyPI、Git 或 `latest` 标签；或
2. **受控内部制品库**：制品库返回同一份不可变 wheel 和签名/哈希清单；下载前先校验清单，
   下载后再校验文件，不能按版本范围解析。

NumPilot wheel 本身不应偷偷把 Ascend 原生库、CANN、OPP 或未获再分发授权的上游源码塞进
Python 包。若法律或平台限制不允许将 AsNumpy wheel 放入同一个 ZIP，应将它作为同一 release
的独立 `components/asnumpy/` 附件交付，并在主 release manifest 中绑定其文件名、字节数和
SHA-256。无论采用哪种形态，运行时都只接受锁定身份完全匹配的组件。

以下行为在发布模式必须拒绝并返回可审计的非零/阻断结果：

- 解析 `latest`、未固定的版本范围、浮动分支或未经审核的 wheel；
- 因本地缓存缺失而自动 `git clone`、`git fetch` 或从网络获取 AsNumpy 源码；
- 安装后静默执行 `pip install -U asnumpy` 或让依赖解析器替换已校验 wheel；
- wheel、Python 模块、native `_core`、CANN/OPP 或 Python ABI 的哈希/身份漂移。

开发者仍可在隔离工作区使用源码优先构建来准备下一版候选，但那是**构建阶段**，不是用户
安装路径。`bootstrap --build-asnumpy` 和“缓存缺失时获取固定源码”的代码路径已经被标为
维护者显式模式；发布安装器默认选择 component bundle，并在组件缺失/校验失败时阻断，不会
自动切换到源码路径。维护者若显式选择源码模式，仍须把它视为有网络风险的开发操作并单独留存
构建回执。

## 2. 当前锁定记录（参考基线）

下表从开发态 `evidence/competition-poisson-stage-20260828/bootstrap-receipt.json` 读取。它
证明了一次真实的 pinned-source build 身份，但该 receipt 和 wheel 都不是当前仓库内的发布
附件；`d8a107…` 不得在没有重新取得二进制本体的情况下当作“随包可下载”。

| 字段 | 冻结值 | 说明 |
|---|---|---|
| 上游仓库 | `https://gitcode.com/cann/asnumpy.git` | 只作来源记录；发布安装不应访问它 |
| Git commit | `1929a3ba44eb532a9f544f774a249cdf1c4119be` | 必须是 detached、无 dirty tree |
| Git tree | `bf2f353825f60bd96f910c3a90fd5b75d4c3c8d0` | 用于防止 commit/工作树不一致 |
| 包版本 | `0.3.1.dev4+g1929a3ba4` | 与 wheel metadata 和运行时 `importlib.metadata` 一致 |
| 许可证 | `Apache-2.0` | 仍须随组件保留上游 LICENSE 和 NOTICE 要求 |
| 目标 wheel | `asnumpy-0.3.1.dev4+g1929a3ba4-cp310-cp310-linux_aarch64.whl` | 仅适用于 CPython 3.10 / Linux AArch64 |
| 当前交付 wheel SHA-256 | `8d4dbd083b1b1a5033431ec6fb9c0b793c96b1cd89386acad1738926b8138271` | 本仓库附件；由固定源码构建结果与已验证安装 native 重打包，作为新的 artifact 身份 |
| 历史 wheel SHA-256（不可替代） | `9822af6aa4ba49a90f31659966a02b1a628daa683ca762471b2b1b3e6a1e1155` | 2026-09-01 回执记录；原 wheel 字节已丢失，不得声称当前附件等于它 |
| 历史资格化 wheel SHA-256（不可替代） | `d8a107c1ff75f506c6218aead803937aae4a2afd59931f05135bf9e6a39a8d49` | 2026-08-28 参考回执；不是当前附件 |
| 已安装 `asnumpy/__init__.py` SHA-256 | `8761f7f9fe557606df6c6683245f4dfc3b8ac69385689a0673f3c90e33afd28e` | 运行时冒烟回执中的文件身份 |
| 已安装 native `_core` SHA-256 | `4166713fcc87fde10d707a09ea40b8acfaf46496298697925d6b22b180011173` | 当前附件对应的运行时冒烟身份 |
| wheel `METADATA` SHA-256 | `bb1206b150596ee9d5be3930c0477cc902bb09c1f2b2232547116ac7eedaddf9` | 供安装后复核 |
| wheel `RECORD` SHA-256 | `8aa704914cd7b767a1c625041bd2c238ebf15d62bf8c05ec1b41c002c52be7d5` | 当前附件逐文件复核 |
| 资格化运行画像 | Linux/AArch64、Ascend 910B4、CANN 9.0.0、CPython 3.10.20、NumPy 2.2.6、Loguru 0.7.3 | 画像外环境只能返回 `CAPABILITY_AUDIT_REQUIRED`；component manifest 会在 apply 前逐项复核 |
| 组件 LICENSE SHA-256 | `1eb85fc97224598dad1852b5d6483bbcf0aa8608790dcc657a5a2a761ae9c8c6` | `components/asnumpy/LICENSE` |
| 组件 SBOM SHA-256 | `a1ed973ab1f15e2b40d726266ff9c64a466cb312d4a0d7e739dfc8d03f8aa7ed` | `components/asnumpy/SBOM.spdx.json` |
| 当前组件 manifest SHA-256 | `873e9673f2b44a725a100838ad1157ae1ae46955b4566da937cc221df4405fd5` | 与 `components/asnumpy/manifest.json` 当前附件一致；receipt 仍会记录实际摘要 |
| 开发态 receipt SHA-256 | `8a6ab368b102ac7815d4dcbeb573fc988f6e5b22a3ef2f550b1c5aeafaa8f28b` | `evidence/.../bootstrap-receipt.json`；不是 wheel 本体 |

为避免把不同回合的证据拼成一个制品，公开的 2026-08-28 evidence 还保留以下历史关联值：
`d8a107c1ff75f506c6218aead803937aae4a2afd59931f05135bf9e6a39a8d49`（wheel）、
`bb1206b150596ee9d5be3930c0477cc902bb09c1f2b2232547116ac7eedaddf9`（METADATA）、
`96e71b0f682b4bad6a4fc9d09e5779c2de6ac64308645c4c179f60b301852630`（RECORD）和
`bd18375f58a4ad732464fadd568986f8a25a9e99cfb0deb0bf7b996cb393c861`（native）。这些值仅用于
审计旧回执，当前组件以本节“当前交付”行和 `manifest.json` 为唯一安装依据。

源码审计还记录了六个构建关键文件的摘要，供下一次重建时比较：

```text
CMakeLists.txt                 d76544dae90390e791fc89caa4018f9bfd1bcd4617b52ae7c23ae05abe87e7a4
LICENSE                         1eb85fc97224598dad1852b5d6483bbcf0aa8608790dcc657a5a2a761ae9c8c6
README.md                      9634533554d62b8bfd0286da7d0b6a1d6974e7154299b88cb809ea10af3f1aee
bindings/python/CMakeLists.txt 4d1e928a8a4e3f695a2bc009a4dbfb8441339bd3a035c532eee57d8639361e35
csrc/CMakeLists.txt            3f5ca526867de88dbefc8912d9563053ad78b6e3bc0b58630baacbc37ae281b6
pyproject.toml                 517be204bad4626bce5c0050a894c73a602d88e5d9e282015507667de4e67e1e
```

Git tree/commit 身份和 wheel 字节哈希是两层不同的合同：同一源码在不同编译器、CANN、
链接器或构建时间戳下可能产生不同 wheel。因此最终发布必须保存**实际交付的 wheel**，不能
只保存 commit 并声称二进制可复现。

## 3. 发行附件布局与安装合同

推荐的受控附件布局如下（示意路径不应包含本机用户名或私有目录）：

```text
release/<version>/
  numpilot-<version>-py3-none-any.whl
  components/asnumpy/
    asnumpy-0.3.1.dev4+g1929a3ba4-cp310-cp310-linux_aarch64.whl
    manifest.json                      # 与本文表格同源的机器可读清单
    LICENSE
    SBOM.spdx.json
  release-manifest.json               # 同时绑定 NumPilot 与 AsNumpy 文件
  SBOM.spdx.json
  THIRD_PARTY_NOTICES.md
```

安装器的发布模式应等价于下面的显式流程；参数名以最终 CLI 帮助为准：

```bash
python -m numpilot bootstrap --apply \
  --asnumpy-component /secure/release/components/asnumpy \
  --receipt /secure/private/bootstrap-receipt.json
python -m numpilot bootstrap --verify --json
```

兼容的显式本地 wheel 路径仍可使用 `--asnumpy-wheel PATH
--asnumpy-wheel-sha256 HEX`，但它只适用于已有可信发布物的迁移/复核；冻结 release 优先使用上面的
component 目录，让 manifest 同时绑定 wheel、LICENSE 和 SBOM。两种路径都不会把 hash 不匹配的文件
交给 pip，也不会因为校验失败而回退到源码获取。

安装前校验归档安全性、发行清单和 wheel SHA-256；安装后再次核对 distribution metadata、
`asnumpy/__init__.py`、所有 native 模块的哈希，并执行固定的 `(1000, 1000) float32`
README smoke。任何 mismatch 都必须保留失败 receipt，不能退回源码下载或自动换包。

release manifest 至少要闭合以下关系：

```text
交付 wheel 字节 hash
  → wheel METADATA/RECORD hash
  → 安装后 Python/native 文件 hash
  → bootstrap/qualification receipt
  → NumPilot release manifest + SBOM + license inventory
```

`9822af…` 与 `d8a107…` 只能作为历史参考，除非发布负责人能提供同字节文件并通过上述闭合。
本次已将可验证的当前字节登记为新的 `8d4dbd…` artifact；重新编译得到的新字节同样必须获得
新的 hash、构建记录和 receipt，不能覆盖旧 hash 或把多个构建混称为同一制品。

## 4. 升级、回滚与安全修复流程

AsNumpy 升级不是依赖更新，而是一次受评审的组件 release：

1. 在隔离构建机获取明确 commit/tag，校验 origin、commit、tree、许可证和关键文件摘要；
2. 在目标 CANN/SoC/Python 矩阵构建 wheel，记录完整构建命令、工具链、环境和新 artifact hash；
3. 运行 AsNumpy 自带测试、NumPilot API/数值/strict-NPU 回归，以及自定义算子接入回归；
4. 生成新的 component lock、SBOM、逐文件许可证清单和 bootstrap/qualification receipts；
5. 在第二个干净环境只从新 wheel 安装，完成 import、README smoke、原有案例重扫和回滚演练；
6. 通过 `develop → main` 评审后，创建不可移动的 NumPilot tag 和对应组件制品；
7. 发布说明同时列出旧版→新版的兼容性、API/ABI 变化、已重跑证据和回滚版本。

若发现安全问题，先撤下受影响的组件 hash，再按同样流程生成修复版本。回滚只能选择已有
锁定且未被撤回的旧 artifact；禁止把“重新拉取当前上游”当作回滚。旧 NumPilot 版本继续绑定
旧 AsNumpy 身份，不能在用户环境中被后台替换。

## 5. 许可证与比赛交付核对

- AsNumpy 源码和构建 wheel 的 `Apache-2.0` 文本必须随组件或最终交付包保留；NumPilot 的
  Apache-2.0 不会重新授权上游代码。
- CANN、OPP、ACLNN、驱动和 CANNBot/ops-nn 的许可证与 AsNumpy 分开核对；它们不能因同放在
  `components/` 目录就被合并成 NumPilot 自有代码。
- 若 wheel 内含由 CANN/工具链生成的二进制或第三方库，SBOM 必须逐项列出来源、版本、许可证、
  文件路径和 SHA-256，并确认具有再分发权。无法确认的库不得进入公开附件。
- 比赛源码 ZIP 可以只包含本锁定文档和脱敏 receipt；不应把私有 checkout、Token、内部仓库
  地址、模型原始响应或未筛选构建目录提交进去。
- PPT/README 应明确“固定 AsNumpy 内部组件 + 本地哈希校验”，不能说成“自动跟随上游最新版本”。

最终发布前，负责人应在 [比赛 RC 发布清单](RELEASE-CHECKLIST-DEMO-V1.md) 中签字确认：组件
本体可取得、哈希与 manifest 相符、第二环境安装通过、许可证/SBOM 完整，且用户在无网络或
上游仓库不可用时仍能复现已承诺的运行路径。

## 6. 与当前开发态的关系

本仓库保存当前组件的 resolver、manifest、wheel、LICENSE、SBOM 和哈希/回执合同；上游
checkout 与完整构建运行树仍位于仓库外。`integrations/asnumpy/0001-move-private-native-results.patch`
针对另一个 AsNumpy commit，不能混入本锁定组件。组件模式已经接通“只读本地附件、缺失即失败、
不隐式获取源码”的门禁；整体 release 仍须在干净提交、第二环境和最终 release manifest 上完成
人工签字。

相关文档：

- AsNumpy 原理与 API 封装底稿属于开发资料，不随公开发布树提供。
- [安装与验收](installation.zh-CN.md)
- [产物存储策略](artifact-storage.md)
- [第三方软件与材料](../THIRD_PARTY_NOTICES.md)
- [比赛 RC 发布清单](RELEASE-CHECKLIST-DEMO-V1.md)
