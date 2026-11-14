# 维生素 C 电子密度网格链：迁移证据

本目录是主科学样例的独立证据目录，结构与 benchmark 缺口样例一致：

- `numpy_source.py`：NumPy 单核 CPU 完整链参考实现（来自 `../numpy_fullchain.py`）；
- `asnumpy_migrated.py`：AsNumpy / Ascend NPU 迁移实现（来自 `../asnumpy_fullchain.py`）；
- `reports/`：迁移计划、结果报告、指标，以及原始证据脚本。

这不是模型生成算子案例：迁移代码为手写，使用 NumPilot 审计运行期
（`model_called=false`），模型 token 消耗为 0。
