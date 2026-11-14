# D1012_class_selection Ascend C 算子证据

- 状态：`case_local_adapter_benchmarked_not_activated`
- 缺口：`numpy.argmax`
- 模型：`deepseek-flash`，17 轮 HTTP 200
- 生成墙钟 / tokens：`868 s` / `711065`

## 结果

| 阶段 | 结果 |
| --- | --- |
| Ascend C 编译 | 通过；`20 s`，尝试 3 个构建目录 |
| OPP 安装 | 通过；`1 s` |
| NPU 原生对拍 | 通过；逐位一致 |
| AsNumpy 公共 API | `asnumpy.argmax`，模块 `asnumpy.custom_ops`，非 NumPy alias |
| 适配层 wheel | 构建通过；`147 s` |
| 完整程序数值对拍 | 通过；`consistent=true` |
| CPU / NPU | `30.189417 / 5.230175 ms`，20 次采样 |
| 加速比 | **5.7722x** |
| 生成后工程耗时 | `499 s` |
| 完整端到端 | `1367 s` |

## 目录

- `numpy_source.py`、`GAP_TASKBOOK.*`、`gap-spec.json`：生成源和缺口契约；
- `draft/`：模型生成并经确定性编译修复的 Ascend C 源码；
- `repairs/`：修复前后哈希和说明；
- `build/`、`deploy/`、`verify/`：编译、私有 OPP、NPU 对拍与契约审计；
- `adapter/`、`benchmark/`、`metrics.json`：适配层、完整程序 benchmark 和机器可读指标。

## 边界

OPP 和 AsNumpy wheel 只安装在本用例私有目录，没有写入共享 AsNumpy 或系统 CANN，也没有全局激活。
