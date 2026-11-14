# F005_max_initial 缺口迁移证据（1024×1024 输入）

- 缺口：`max(initial=)`
- 输出：`out,shifted`
- 输入规模：1024×1024 float32，1,048,576 元素
- 状态：`passed`
- 完整端到端：`77.75501585006714` s
- 迁移命令：`11.341966` s
- CPU median：`0.675472` ms
- NPU median：`0.7376415` ms
- 加速比：`0.9157185434930112`
- 模型 tokens（含已知重写尝试）：`20844`

结构：

- `numpy_source.py`：原始 NumPy 程序（1024×1024 输入）；
- `numpy_flat_source.py`：进入 NumPyPilot 交互式工作台的直线计算源；
- `model_adapted_source.py`：模型消解缺口后的源程序；
- `asnumpy_migrated.py`：最终 AsNumpy 程序；
- `reports/`：迁移计划、迁移报告、验证/性能回执、workflow 回执、token 指标。

绝对路径已脱敏为仓库相对路径或 `<home>`；原始运行目录保留在服务器。
