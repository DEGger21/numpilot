# F008_linspace_endpoint 缺口迁移证据（1024×1024 输入）

- 缺口：`linspace(endpoint=)`
- 输出：`out`
- 输入规模：1024×1024 float32，1,048,576 元素
- 状态：`passed`
- 完整端到端：`48.74586844444275` s
- 迁移命令：`10.133838` s
- CPU median：`8.714843` ms
- NPU median：`0.2792805` ms
- 加速比：`31.204624024949826`
- 模型 tokens（含已知重写尝试）：`18134`

结构：

- `numpy_source.py`：原始 NumPy 程序（1024×1024 输入）；
- `numpy_flat_source.py`：进入 NumPyPilot 交互式工作台的直线计算源；
- `model_adapted_source.py`：模型消解缺口后的源程序；
- `asnumpy_migrated.py`：最终 AsNumpy 程序；
- `reports/`：迁移计划、迁移报告、验证/性能回执、workflow 回执、token 指标。

绝对路径已脱敏为仓库相对路径或 `<home>`；原始运行目录保留在服务器。
