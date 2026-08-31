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
| Huawei ops-nn | official-first 缺口解析与定向算子来源；不随 NumPilot 默认安装或分发 | CANN Open Software License Agreement Version 2.0；下载、使用、修改或分发前需显式接受并遵守用途和分发限制 |

项目不会把 AsNumpy、CANN、ops-nn 或完整 CANNBot 仓库复制进 Python 分发包。当前
`numpilot bootstrap` 只支持按固定来源身份准备用户选择的 AsNumpy/CANNBot 依赖；
official-first resolver 只读检查用户显式提供的 CANN 根目录和本地 ops-nn Git 对象，不会由
bootstrap、import、analyze 或 migrate 隐式拉取源码。NumPilot 仓内仅保存自有的 resolver、
来源/哈希收据和一份 AsNumpy 集成参考 patch，不重新分发 ops-nn 或 CANNBot 生成的算子源码。用户仍应核对各自许可证与
部署要求。发布二进制、wheel、OPP 或比赛交付包时，还应生成对应版本的完整 SBOM 和逐文件
许可证清单。

## 名称、Logo 与商标

`docs/assets/brand/` 中的 AsNumpy、CANN、Ascend 和 CodeArts 名称及 Logo 仅用于说明文档中的
技术关系。相关权利归各自所有者，不随 NumPilot 的 Apache-2.0 重新授权，也不表示赞助或背书。

仓库中的派生图片仅对用户提供的文档截图进行裁切、缩放或透明区域保留，没有重绘或改变标识本身。
逐资产来源、处理方式、哈希与待核权状态见
[`docs/assets/brand/README.md`](docs/assets/brand/README.md)。截图或商标的公开再分发权不能从本项目
Apache-2.0 许可证推导；正式公开前必须由仓库所有者确认或替换为获准资产。
