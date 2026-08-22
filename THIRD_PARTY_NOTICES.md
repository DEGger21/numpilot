# 第三方软件与材料

NumPilot 自有源码按顶层 Apache-2.0 发布。依赖和证据材料仍受各自许可证约束；顶层许可证
不会替代这些条款。

| 项目 | 用途 | 许可证 / 说明 |
|---|---|---|
| NumPy | CPU 参考实现与 shim | BSD-3-Clause |
| Rich | 终端界面 | MIT |
| pytest | 开发与测试 | MIT |
| threadpoolctl | 可选 BLAS 线程观测 | BSD-3-Clause |
| AsNumpy | 可选 Ascend NPU 后端 | Apache-2.0；不随 NumPilot wheel 捆绑 |
| Huawei CANN / ops-nn / CANNBot Skills | NPU 工具链、模板与算子开发参考 | CANN Open Software License Agreement Version 2.0；需遵守其用途和分发限制 |

`evidence/poisson-ritz-p0-20260821/qualification-evidence-v5/*/artifact/` 中的定向算子材料
源自固定版本 CANNBot Skill 驱动的生成与修复流程，按其 `OperatorSpec` 标记为 CANN Open
Software License Agreement Version 2.0。完整协议副本位于该证据目录的 `LICENSE`。

项目不会把 AsNumpy、CANN、ops-nn 或完整 CANNBot 仓库复制进 Python 分发包。用户应从各
上游项目获取兼容版本，并核对其最新许可证与部署要求。发布二进制、wheel、OPP 或比赛交付
包时，还应生成对应版本的完整 SBOM 和逐文件许可证清单。
