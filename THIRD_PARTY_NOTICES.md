# 第三方软件与材料

NumPilot 自有源码按顶层 Apache-2.0 发布。依赖和证据材料仍受各自许可证约束；顶层许可证
不会替代这些条款。

| 项目 | 用途 | 许可证 / 说明 |
|---|---|---|
| NumPy | CPU 参考实现与 shim | BSD-3-Clause |
| Loguru | AsNumpy 运行时日志依赖 | MIT |
| Rich | 终端界面 | MIT |
| pytest | 开发与测试 | MIT |
| build / scikit-build-core / pybind11 / setuptools-scm / towncrier | 可选 AsNumpy 源码构建工具链 | 各项目许可证；仅 `numpilot[bootstrap]` 可选依赖 |
| threadpoolctl | 可选 BLAS 线程观测 | BSD-3-Clause |
| AsNumpy | 可选 Ascend NPU 后端 | Apache-2.0；不随 NumPilot wheel 捆绑 |
| Huawei CANN / CANNBot Skills | NPU 工具链、模板与算子开发参考 | CANN Open Software License Agreement Version 2.0；需遵守其用途和分发限制 |
| Huawei ops-nn | P1 待定的官方缺口算子复用来源；不随 NumPilot 默认安装 | CANN Open Software License Agreement Version 2.0；下载、使用、修改或分发前需显式接受并遵守用途和分发限制 |

`evidence/poisson-ritz-p0-20260821/qualification-evidence-v5/*/artifact/` 中的定向算子材料
源自固定版本 CANNBot Skill 驱动的生成与修复流程，按其 `OperatorSpec` 标记为 CANN Open
Software License Agreement Version 2.0。完整协议副本位于该证据目录的 `LICENSE`。

项目不会把 AsNumpy、CANN、ops-nn 或完整 CANNBot 仓库复制进 Python 分发包。当前
`numpilot bootstrap` 只支持按固定来源身份准备用户选择的 AsNumpy/CANNBot 依赖；ops-nn
resolver 仍是 P1 待定，不得据此声称 bootstrap 会拉取它。用户仍应核对各自许可证与部署
要求。发布二进制、wheel、OPP 或比赛交付包时，还应生成对应版本的完整 SBOM 和逐文件
许可证清单。
