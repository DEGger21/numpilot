# AsNumpy 源码基线与缺口扩展流程

NumPilot 当前把已接入 MatMul 的 AsNumpy 源码作为作者维护的基线组件随发行物保存。源码快照用于审阅、
修改和显式重建；普通用户安装、import、migrate 和 verify 仍只使用经过哈希校验的静态 wheel，不会
自动编译或联网拉取源码。

## 当前组件

`components/asnumpy/` 同时包含：

| 文件 | 用途 |
| --- | --- |
| `manifest.json` | wheel、源码、运行时和许可证身份的总锁 |
| `source-manifest.json` | 源码快照逐文件 SHA-256、文件数和总字节数 |
| `asnumpy-source-*.tar.gz` | 当前源码基线的完整 Git tracked 文件快照 |
| `asnumpy-*.whl` | 默认运行时组件，包含已接入 ACLNN MatMul 的 native 扩展 |
| `LICENSE` / `SBOM.spdx.json` | 再分发和软件物料证据 |

当前源码基线：

- commit：`60d139c54d9e181b8ae101dcbd1479d6b2a86439`
- tree：`f1c17d947a165c587a1b1d618d11d03fc5f5b183`
- 源码文件：211 个，源码字节：1,803,983
- 源码归档 SHA-256：`33ebd49da2aa23e906906cd3d57686012464c61ca819b2354eb6fdbe31ef6b62`
- 构建环境基线：CPython 3.12.14、Linux AArch64、CANN 9.2.0-beta.2、GCC 11.4.0、C++17

## 作者补充一个缺口的固定流程

1. 用锁定快照创建维护工作目录。该命令会校验 archive、source-manifest、commit、tree、逐文件
   SHA-256 和路径安全性，不访问 Git 或网络：

   ```bash
   python tools/extract_asnumpy_source.py --out /absolute/private/asnumpy-work
   ```

2. 在提取出的源码目录中修改 Python/C++/CMake，并为缺口添加 AsNumpy 测试。不要直接修改
   `components/asnumpy/` 中的 wheel 或 manifest；它们是发布附件。
3. 在目标 CANN、SoC、Python ABI 和编译器环境中运行 AsNumpy 定向测试及全量测试。对于 native
   算子，必须记录实际调用路径、dtype/rank/layout 边界和是否发生 host fallback。
4. 用 AsNumpy 原生构建链显式构建下一版 wheel。源码快照没有 Git 元数据，构建时要显式绑定版本：

   ```bash
   export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_ASNUMPY=<new-version>
   python -m build --wheel --no-isolation --outdir /absolute/private/wheel-out \
     /absolute/private/asnumpy-work
   ```

   构建依赖包括 scikit-build-core、pybind11、NumPy、setuptools-scm、towncrier、CMake/C++17 和
   目标 CANN SDK；这些依赖由作者维护的构建环境提供，不由 NumPilot 运行时自动安装。
5. 生成新的 wheel SHA-256、METADATA/RECORD 哈希、安装后 Python/native 文件哈希、SBOM 和
   API catalog；同步更新 source-manifest、manifest、runtime identity 和锁定文档。
6. 在干净环境中用新组件执行 AsNumpy smoke、NumPilot 全量测试、strict profile 资格和适用性能
   门，再提交新的不可变组件版本。旧组件保留为可回滚版本。

## 三种情况要区分

- **永久补齐 AsNumpy API**：修改 AsNumpy 源码并发布新的 wheel + 源码快照。这是比赛主线和正式
  capability 的唯一来源。
- **已有 API 的组合适配**：可以在单次 run 中用已经资格化的 AsNumpy API 组合出临时接口，例如
  已登记的 `ones_like + add` 路线。它只属于 run-scoped adapter，不修改静态组件，也不能
  自动升级全局 API catalog。
- **CANNBot Skill**：只提供官方算子、编译和调试知识上下文。它可以帮助生成 custom-op 草稿，
  但不改变 AsNumpy Python/native 接口，也不等于已编译、已接入或已激活。

## 为什么源码和 wheel 要同时保留

wheel 是当前运行时的可复现二进制；源码快照是作者进行缺口修复和下一次构建的完整依据。普通运行
路径不自动从源码构建，是为了避免 CANN、驱动、编译器、Python ABI 或链接库漂移导致同一 commit
产生不可比的 native 结果，也避免用户环境出现隐式编译和网络副作用。

因此，静态打包并没有堵住扩展能力：它把扩展动作收敛到作者可审计的源码发布流程，而不是把修改
责任放到用户运行时。每次永久补缺都产生新的 component_id、wheel/source manifest、API catalog
和 runtime qualification receipt。

## 当前边界

源码快照是作者维护附件，不是 NumPilot 对外承诺的通用 AsNumpy 开发环境。它不包含 CANN、驱动、
固件、OPP、系统 fmt/spdlog 或第三方模型配置；这些仍按安装文档和目标机器的资格门管理。任何
源码修改都必须重新构建、重新测试和重新绑定身份，不能沿用旧 wheel 或旧 strict-NPU 结论。

