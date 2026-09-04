# 第三方软件与材料

NumPilot 自有源码按顶层 Apache-2.0 发布。依赖和随附材料仍受各自许可证约束；顶层许可证
不会替代这些条款。

| 项目 | 用途 | 许可证 / 说明 |
|---|---|---|
| NumPy | CPU 参考实现与 shim | BSD-3-Clause |
| Loguru | AsNumpy 运行时日志依赖 | MIT |
| Rich | 终端界面 | MIT |
| pytest | 开发与测试 | MIT |
| build / scikit-build-core / pybind11 / setuptools-scm / towncrier | 可选 AsNumpy 源码构建工具链 | 各项目许可证；仅 `numpilot[bootstrap]` 可选依赖 |
| threadpoolctl | 可选 BLAS 线程观测 | BSD-3-Clause |
| AsNumpy | Ascend NPU 运行时内部组件 | Apache-2.0；当前仓库已随 `components/asnumpy/` 提供固定 Linux/AArch64 CPython 3.10 wheel、SHA-256、上游 LICENSE 和 SPDX SBOM。正式发布仍须确认再分发权，不得解析浮动版本/`latest` |
| Huawei CANN / CANNBot Skills | NPU 工具链、模板与算子开发参考 | CANN Open Software License Agreement Version 2.0；需遵守其用途和分发限制 |
| Huawei ops-nn | official-first 缺口解析与定向算子来源；不随 NumPilot 默认安装或分发 | CANN Open Software License Agreement Version 2.0；下载、使用、修改或分发前需显式接受并遵守用途和分发限制 |

## 外部 AI 模型与平台服务

NumPilot 不在源码或 wheel 中捆绑大语言模型、模型权重、API Token 或私有网关。普通脚本迁移
使用的是用户在运行时通过 `llm.json` 配置的 OpenAI Chat Completions-compatible 服务；模型名称、
服务商、版本、调用时间和适用许可证应由部署者按实际服务记录。仓库不保存开发环境中的
私有模型配置，也不把配置文件视为模型来源证明。

CodeArts 代码智能体及其内置模型属于外部平台。仓库中的 `.codeartsdoer/` Agent/Skill 文件只是
可加载的工作流上下文，不能替代对应账号中的真实任务记录，也不授予外部服务或模型的使用权。
部署者应遵守所选平台、模型和 API 的实际使用条款。

项目不会把 CANN、ops-nn 或完整 CANNBot 仓库复制进 Python 分发包。AsNumpy 在冻结 release 中
作为显式标记的内部运行时组件交付：源代码树中的附件位于 `components/asnumpy/`，构建出的
NumPilot wheel/sdist 会按 `share/numpilot/components/asnumpy/` 携带同一份 manifest、LICENSE、SBOM
和精确 wheel（这不是把 CANN 原生库或上游源码偷偷混入包内）。若再分发条款不允许随 NumPilot
包携带二进制，可改用同一 release 的外部组件附件，并让安装器显式指向它；组件锁定的 commit/tree、
wheel/module/native 哈希和升级/回滚流程见
[`docs/ASNUMPY-INTERNAL-COMPONENT-LOCK.zh-CN.md`](docs/ASNUMPY-INTERNAL-COMPONENT-LOCK.zh-CN.md)。
发布安装器必须只从该受控 wheel 安装，不得因为文件缺失而从上游拉取源码。`numpilot bootstrap`
的默认 component 模式不访问网络；显式 `--build-asnumpy` 才是维护者的开发/构建路径，不能当作
冻结 release 的用户安装合同；
official-first resolver 只读检查用户显式提供的 CANN 根目录和本地 ops-nn Git 对象，不会由
bootstrap、import、analyze 或 migrate 隐式拉取 CANN/ops-nn 源码。NumPilot 仓内仅保存自有的
resolver、来源/哈希收据和一份 AsNumpy 集成参考 patch，不重新分发 ops-nn 或 CANNBot 生成的
算子源码；AsNumpy wheel 是否可公开再分发仍须由发布负责人按上游条款确认。用户仍应核对各自许可证与
部署要求。发布二进制、wheel 或 OPP 时，还应生成对应版本的完整 SBOM 和逐文件
许可证清单。

## 名称、Logo 与商标

`docs/assets/brand/` 中的 AsNumpy、CANN、Ascend 和 CodeArts 名称及 Logo 仅用于说明文档中的
技术关系。相关权利归各自所有者，不随 NumPilot 的 Apache-2.0 重新授权，也不表示赞助或背书。

仓库中的派生图片仅对用户提供的文档截图进行裁切、缩放或透明区域保留，没有重绘或改变标识本身。
逐资产来源、处理方式、哈希与待核权状态见
[`docs/assets/brand/README.md`](docs/assets/brand/README.md)。截图或商标的公开再分发权不能从本项目
Apache-2.0 许可证推导；正式公开前必须由仓库所有者确认或替换为获准资产。
