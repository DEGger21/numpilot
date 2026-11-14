# D1009_periodic_transport Ascend C 算子证据

- 状态：`case_local_adapter_benchmarked_not_activated`
- 缺口：`numpy.roll`
- 模型：`deepseek-flash`，17 轮 HTTP 200
- 生成墙钟 / tokens：`890 s` / `716941`

## 结果

| 阶段 | 结果 |
| --- | --- |
| Ascend C 编译 | 通过；成功构建 `21 s`，尝试 4 个构建目录 |
| OPP 安装 | 通过；`1 s`，安装到本用例私有 vendor |
| NPU 原生对拍 | 通过；逐位一致（mismatch=0） |
| AsNumpy 公共 API | `asnumpy.roll`，模块 `asnumpy.custom_ops`，非 NumPy alias |
| 适配层 wheel | 构建通过；`99 s` |
| 完整程序数值对拍 | 通过；`consistent=true` |
| CPU / NPU | `26.669054 / 7.455630 ms`，20 次采样 |
| 加速比 | **3.5770x** |
| 生成后工程耗时 | `2222 s` |
| 完整端到端 | `3112 s`（从生成开始到 benchmark 完成） |

## 目录

- `numpy_source.py`：放大的 NumPy 源程序（1024×1024 float32）；
- `GAP_TASKBOOK.md/json`、`gap-spec.json`：缺口任务书与契约；
- `draft/`：模型生成并经确定性编译修复的 Ascend C 源码、host/kernel/op_api、构建脚本和单测；
- `repairs/`：修复前后的哈希清单与原始草稿；
- `build/`、`deploy/`：编译和私有 OPP 安装回执；
- `verify/`：NPU 原生对拍与跨文件契约审计；
- `adapter/`：AsNumpy 接入源码、wheel 和构建日志；
- `benchmark/`：完整程序 `migrated.py`、公共路由校验和 CPU/NPU 结果；
- `metrics.json`：本表数据的机器可读版本。

## 边界

- 模型只生成 Ascend C 草稿；编译通过、NPU 对拍通过、适配层构建通过和完整程序 benchmark 通过是
  确定性验证结果，不是模型自报。
- 适配层和 OPP 仅安装在本用例私有目录，**没有写入共享 AsNumpy 或系统 CANN**，也没有激活为全局能力。
- `diff` 的 NaN 输出 payload 与 CPU 不同，但 NaN 位置和非 NaN 位一致；这是浮点运算结果的语义等价，
  不是逐位搬运算子。

- `verify/contract-audit.json` 通过；其中一条 warning 指出 kernel entry 是非模板函数而 tiling-key header 声明了 `D_T_X`。host 使用常量 TilingKey，该 warning 不影响已构建和已验证的路径。
