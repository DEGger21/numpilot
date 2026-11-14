# F010_nan_to_num_inplace 缺口迁移证据（1024×1024 输入）

- 缺口：`nan_to_num(copy=)`
- 输出：`out,input_after,squared`
- 输入规模：1024×1024 float32，1,048,576 元素
- 状态：`passed`
- 完整端到端：`67.01509118080139` s
- 迁移命令：`9.134424` s
- CPU median：`11.309616` ms
- NPU median：`0.193215` ms
- 加速比：`58.53384054033072`
- 模型 tokens（含已知重写尝试）：`19378`

结构：

- `numpy_source.py`：原始 NumPy 程序（1024×1024 输入）；
- `numpy_flat_source.py`：进入 NumPyPilot 交互式工作台的直线计算源；
- `model_adapted_source.py`：模型消解缺口后的源程序；
- `asnumpy_migrated.py`：最终 AsNumpy 程序；
- `reports/`：迁移计划、迁移报告、验证/性能回执、workflow 回执、token 指标。

绝对路径已脱敏为仓库相对路径或 `<home>`；原始运行目录保留在服务器。
