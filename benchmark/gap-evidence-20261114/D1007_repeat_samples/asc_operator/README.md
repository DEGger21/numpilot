# D1007_repeat_samples Ascend C 算子证据

- 状态：`case_local_adapter_benchmarked_not_activated`
- 缺口：`numpy.repeat`
- 模型：`deepseek-flash`，17 轮 HTTP 200
- 生成墙钟 / tokens：`914 s` / `717123`

## 结果

| 阶段 | 结果 |
| --- | --- |
| Ascend C 编译 | 通过；`18 s`，尝试 3 个构建目录 |
| OPP 安装 | 通过；`1 s` |
| NPU 原生对拍 | 通过；逐位一致 |
| AsNumpy 公共 API | `asnumpy.repeat`，模块 `asnumpy.custom_ops`，非 NumPy alias |
| 适配层 wheel | 构建通过；`149 s` |
| 完整程序数值对拍 | 通过；`consistent=true` |
| CPU / NPU | `49.392091 / 27.628917 ms`，20 次采样 |
| 加速比 | **1.7877x** |
| 生成后工程耗时 | `441 s` |
| 完整端到端 | `1355 s` |

`complete-gaps` 报告为 `generated_unverified_partially_blocked`：缺口算子已生成，另有 2 个阻断项来自源码里的标量二元操作数，不是缺口算子本身。benchmark 源已改为等价的 `full_like` 数组操作数，记录在 `benchmark/source-adaptation.json`。

## 目录

- `numpy_source.py`、`GAP_TASKBOOK.*`、`gap-spec.json`：生成源和缺口契约；
- `draft/`：模型生成并经确定性编译修复的 Ascend C 源码；
- `repairs/`：修复前后哈希和说明；
- `build/`、`deploy/`、`verify/`：编译、私有 OPP、NPU 对拍与契约审计；
- `adapter/`、`benchmark/`、`metrics.json`：适配层、完整程序 benchmark 和机器可读指标。

## 边界

OPP 和 AsNumpy wheel 只安装在本用例私有目录，没有写入共享 AsNumpy 或系统 CANN，也没有全局激活。
