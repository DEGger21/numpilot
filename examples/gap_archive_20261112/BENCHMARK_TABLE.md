# 含真实缺口算子的完整程序迁移性能清单（历史 6 例主表 + 后续签名/算子批次）

> 说明：主表和“主表 6 例汇总”对应 2026-11-12 的 6 个真实缺口算子案例；2026-11-14 新增的 5 个签名缺口案例和 3 个真实 Ascend C 算子案例在文末单列，不混入主表统计。

本表覆盖 2026-11-12 批次中的 **6 个完整 NumPy 程序 → 6 个完整 AsNumpy 程序**：
每个 `program.py` 都包含 `build_inputs()` 与 `forward()`，迁移后的 `migrated.py`
保持同一程序结构与全部输入输出，只把 NumPy 调用替换为 AsNumpy 调用。
六个程序各含 1 个真实缺口 API；其中 `tril` 的独立 benchmark 通过，但 NumPilot 完整迁移回执失败。缺口算子本身的生成与补全耗时单列在第二张表，
主表的“完整端到端”列给出从模型生成开始到全程序基准完成的总时间。

```
缺口发现 → LLM 生成 Ascend C 算子草稿 → 构建/修复 → 部署 → 数值验证
         → AsNumpy 适配层构建+单测 → 准入 → NumPilot 6 智能体迁移完整程序
         → strict-NPU 执行 → 全程序 CPU/NPU 基准
```

## 测试环境

| 项目 | 配置 |
| --- | --- |
| CANN | 9.2.0-beta.2 |
| NPU | Ascend 910B3 |
| CPU 基线 | 单核（BLAS/OMP 线程数 = 1），numpy 2.5.4 |
| AsNumpy | 0.3.1（为每个缺口算子构建的 overlay wheel） |
| 生成模型 | `deepseek-flash`（每次生成前通过 `llm.json` 真实远端调用；调用前已获批） |
| 计时 | 墙钟；构建/部署/适配/迁移/基准由 `stage_timer.sh` 记录 |

## 口径

| 字段 | 定义 |
| --- | --- |
| 迁移耗时 | 对完整 `program.py` 执行一次 `python -m numpilot migrate` 的墙钟时间；输出是完整 `migrated.py`，不是只替换缺口调用。本批使用 AST 引擎（确定性），模型调用发生在缺口生成阶段 |
| 生成 | `numpilot complete-gaps --provider llm` 从 preflight 到草稿落盘的墙钟时间（含真实模型调用） |
| 补全 | 草稿落盘后，确定性构建/修复/部署/数值验证到 provenance 生成的墙钟时间。**修复不是模型输出**；模型原始草稿保存在 `operators/<case>/draft/`，最终可构建版本保存在 `operators/<case>/final/` |
| 生成+补全 | 上面两段之和，即从模型调用开始到缺口算子通过数值验证的总耗时 |
| 完整端到端 | 从模型生成开始到全程序 CPU/NPU 基准完成的墙钟时间；包含生成、构建/修复/部署/验证、适配层、迁移命令与基准 |
| 适配层 | 为 AsNumpy 构建 overlay wheel、安装并执行算子单测的耗时（pad 的适配层测试与后续构建修复交错，分段耗时不严格可加） |
| 构建次数 | `build-attempt*.log` 的数量，反映草稿修复轮次 |
| 加速比 | `t_cpu_ms / t_npu_ms`，完整程序在 CPU 单核 vs NPU strict-asnumpy 上的 `forward()` 耗时（输入驻留设备，含 D2H 同步） |

## 主表：完整程序迁移

| # | 源码 NumPy 程序（相对路径） | 迁移后 AsNumpy 程序（相对路径） | 数据规模 | 替换的 API | 迁移耗时 | 完整端到端 | CPU / NPU (ms) | 加速比 |
| ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | `examples/gap_archive_20261112/cases/std/program.py` | `examples/gap_archive_20261112/cases/std/migrated.py` | 2048 × 2048 float32；4,194,304 元素；沿 axis=1 求标准差 | `np.std` | 7 s | 90.9 min | 15.96 / 0.3718 | **42.92x** |
| 2 | `examples/gap_archive_20261112/cases/flip/program.py` | `examples/gap_archive_20261112/cases/flip/migrated.py` | 2048 × 2048 float32；4,194,304 元素；沿 axis=1 翻转 | `np.flip` | 8 s | 75.3 min | 0.007381 / 1.756 | 0.0042x |
| 3 | `examples/gap_archive_20261112/cases/ediff1d/program.py` | `examples/gap_archive_20261112/cases/ediff1d/migrated.py` | 2048 × 2048 float32；4,194,304 元素展平；输出 4,194,303 个相邻差分 | `np.ediff1d` | 6 s | 60.7 min | 4.48 / 1.664 | **2.69x** |
| 4 | `examples/gap_archive_20261112/cases/pad/program.py` | `examples/gap_archive_20261112/cases/pad/migrated.py` | 2048 × 2048 float32；constant 填充 ((1,1),(2,2))；输出 2050 × 2052 | `np.pad` | 7 s | 47.0 min | 4.117 / 1.724 | **2.39x** |
| 5 | `examples/gap_archive_20261112/cases/tril/program.py` | `examples/gap_archive_20261112/cases/tril/migrated.py` | 2048 × 2048 float32；4,194,304 元素；默认 k=0 下三角 | `np.tril` | 8 s | 38.4 min | 14.16 / 1.649 | **8.59x** |
| 6 | `examples/gap_archive_20261112/cases/argpartition/program.py` | `examples/gap_archive_20261112/cases/argpartition/migrated.py` | 32 × 16 float32；512 元素；沿 axis=1、kth=3 分区 | `np.argpartition` | 7 s | 38.2 min | 0.01049 / 0.3489 | 0.03x |

## `tril` 完整迁移回执边界

`evidence/tril/migrate-report.json` 明确为 `status=fail`，异常为 `runtime target snapshot is invalid`；
独立 `benchmark/npu_benchmark.json` 为 `status=pass`、`performance_valid=true`、8.59x。
因此主表的 benchmark 数字不能被解释为 NumPilot 完整迁移已通过。

## 缺口算子生成/补全明细

| 案例 | 缺口 API | 生成 | 补全（构建/部署/验证） | 生成+补全 | 适配层 | 构建次数 | 单测 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `std` | `numpy.std(a, axis=None)` | 15.8 min | 41.4 min | 57.2 min | 2.4 min | 6 | 14 |
| `flip` | `numpy.flip(a, axis=0)` | 14.7 min | 32.7 min | 47.4 min | 2.1 min | 5 | 13 |
| `ediff1d` | `numpy.ediff1d(a)` | 14.7 min | 18.0 min | 32.7 min | 2.4 min | 2 | 10 |
| `pad` | `numpy.pad(a, pad_width, mode='constant')` | 16.9 min | 28.5 min | 45.3 min | 1.9 min | 13 | 15 |
| `tril` | `numpy.tril(a, k=0)` | 14.5 min | 21.9 min | 36.4 min | 1.7 min | 2 | 13 |
| `argpartition` | `numpy.argpartition(a, kth, axis=-1)` | 15.1 min | 17.0 min | 32.1 min | 1.7 min | 1 | 19 |

## 主表 6 例汇总（2026-11-12）

| 指标 | 数值 |
| --- | --- |
| 覆盖完整迁移程序 | 6 |
| 含真实缺口算子的程序 | 6 |
| 完整 `migrated.py` 生成 | 6 / 6 |
| 全程序数值对拍通过 | 5 / 6；`tril` 完整迁移回执失败 |
| 适配层单测 | 84 passed |
| 生成耗时范围 | 14.5–16.9 min（均值 15.3 min） |
| 生成+补全范围 | 32.1–57.2 min（均值 41.9 min） |
| 迁移命令范围 | 6–8 s |
| 端到端范围 | 38.2–90.9 min（均值 58.4 min） |
| 加速比 | 42.92x、0.0042x、2.69x、2.39x、8.59x、0.03x |
| 加速比 > 1x | 4 / 6 |

## 必须如实说明的两个案例

- **`flip`（0.0042x）**：CPU 基线是 `numpy.flip`，它返回 view，耗时是 O(1) 的
  视图构造（约 0.007 ms）；NPU 算子需要真实搬运 4 Mi 元素（约 1.76 ms）。
  两者不是等价工作量，**该比值不能当成“NPU 比 CPU 慢 238 倍”的性能结论**。
- **`argpartition`（0.03x）**：输入只有 512 个元素，NPU 时间由 kernel launch
  和每轮 D2H 同步主导（0.349 ms），CPU 是 0.010 ms。该案例的价值是
  **正确性/补全能力**：最终 kernel 逐行复刻 numpy 2.5.4 的 introselect
  （arg=true），在 5256 组对抗样例上与 `np.argpartition` 完全一致，并通过
  19 个适配层单测。不要把它放进“加速比”宣传口径。

## 可复现路径

- 原始程序：`cases/<case>/program.py`
- 迁移后程序：`cases/<case>/migrated.py`
- 模型生成草稿：`operators/<case>/draft/`
- 最终可构建算子：`operators/<case>/final/{op_host,op_kernel,op_api,asnumpy_adapter}`
- 生成的 AsNumpy 适配层：`operators/<case>/adapter/`
- 证据：`evidence/<case>/{gap-spec,generation-preflight,admission,custom-opp-provenance,migrate-report,migrate.log,benchmark/*}`

## 2026-11-14 benchmark 部分缺口补充批次（5 例，1024×1024 输入）

本批从 150 例 benchmark 的 D/E/F/G 部分缺口用例中按“由易到难”先取 5 个签名缺口，
输入统一放大到 1024×1024（1,048,576 个 float32 元素），通过配置模型生成 `forward`
重写、NumPyPilot 交互式六智能体迁移、CPU/NPU 数值对拍和 benchmark 完成。单批最大并发 3。

这些用例的缺口是 **NumPy 调用签名差异**，不是新增 Ascend C 算子；模型把缺口调用改写为
AsNumpy 已覆盖的等价组合，原始程序仍是数值 oracle。

| # | 源码 NumPy 程序（相对路径） | 迁移后 AsNumpy 程序（相对路径） | 数据规模 | 缺口 | 迁移耗时 | 完整端到端 | CPU / NPU (ms) | 加速比 | 模型 tokens |
| ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `benchmark/cases/F_partial_consent/F001_add_out/numpilot-work/scale-1024/flat-v1/program_flat.py` | `benchmark/cases/F_partial_consent/F001_add_out/numpilot-work/scale-1024/interactive-v1/workflow/migration/migrated.py` | 1024×1024 float32；1,048,576 输入元素 | `np.add(out=)` | 11.2 s | 72.4 s | 4.312869 / 0.246286 | **17.51x** | 16,389 |
| 2 | `benchmark/cases/F_partial_consent/F004_sum_initial/numpilot-work/scale-1024/flat-v1/program_flat.py` | `benchmark/cases/F_partial_consent/F004_sum_initial/numpilot-work/scale-1024/interactive-v1/workflow/migration/migrated.py` | 1024×1024 float32；1,048,576 输入元素，输出 1024 | `np.sum(initial=)` | 10.3 s | 69.2 s | 0.541932 / 0.566367 | 0.9569x | 22,391（另有一次空响应未记录） |
| 3 | `benchmark/cases/F_partial_consent/F005_max_initial/numpilot-work/scale-1024/flat-v1/program_flat.py` | `benchmark/cases/F_partial_consent/F005_max_initial/numpilot-work/scale-1024/interactive-v1/workflow/migration/migrated.py` | 1024×1024 float32；1,048,576 输入元素，输出 1024 | `np.max(initial=)` | 11.3 s | 77.8 s | 0.675472 / 0.737642 | 0.9157x | 20,844 |
| 4 | `benchmark/cases/F_partial_consent/F008_linspace_endpoint/numpilot-work/scale-1024/flat-v1/program_flat.py` | `benchmark/cases/F_partial_consent/F008_linspace_endpoint/numpilot-work/scale-1024/interactive-v1/workflow/migration/migrated.py` | 1D float32；1,048,576 输入元素 | `np.linspace(endpoint=)` | 10.1 s | 48.7 s | 8.714843 / 0.279281 | **31.20x** | 18,134 |
| 5 | `benchmark/cases/F_partial_consent/F010_nan_to_num_inplace/numpilot-work/scale-1024/flat-v1/program_flat.py` | `benchmark/cases/F_partial_consent/F010_nan_to_num_inplace/numpilot-work/scale-1024/interactive-v1/workflow/migration/migrated.py` | 1024×1024 float32；1,048,576 输入元素 | `np.nan_to_num(copy=)` | 9.1 s | 67.0 s | 11.309616 / 0.193215 | **58.53x** | 19,378 |

口径：

- **完整端到端**：从交互会话开始到 CPU/NPU benchmark 完成的墙钟时间，包含源码解读、规划、
  缺口计划、LLM 迁移、数值验证和 benchmark。
- **模型 tokens**：缺口重写模型调用 + 交互式六角色模型调用中可读取的 `usage.total_tokens`
  之和；包含已归档的非成功重写尝试；F004 另有一次空响应未落 usage。
- 五组均 `workflow=passed`、`numerically_consistent=true`、`max_abs_diff=0.0`。
- F004/F005 的输入是 1024×1024，但输出是 1024 元素归约结果，NPU 时间由 launch/归约开销主导，
  因此加速比略低于 1；F001/F008/F010 在同等 1M 输入下体现正加速。
- 成功产物保留在各用例的 `numpilot-work/scale-1024/interactive-v1`；
  脱敏证据见 `benchmark/gap-evidence-20261114/README.md`。

## 2026-11-14 真实 Ascend C 生成 + 构建 + benchmark 批次（3 例）

从 D1 单缺口用例中选择 `spacing`、`diff`、`roll`，输入放大到 1024×1024（1,048,576 个
float32 元素），走 `resolve-gaps → complete-gaps --provider llm` 的真实生成路线，再继续
完成编译、私有 OPP 安装、NPU 原生对拍、AsNumpy 适配层构建、完整程序迁移和 CPU/NPU benchmark。
三组最终状态为 `case_local_adapter_benchmarked_not_activated`：能力仅安装在用例私有目录，
没有写入共享 AsNumpy 或系统 CANN，也没有全局激活。

| 样例 | 缺口 API | 生成墙钟 | 生成 tokens | 编译 | NPU 原生对拍 | CPU / NPU (ms) | 加速比 | 完整端到端 |
| --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: |
| `D1002_ulp_budget` | `np.spacing` | 641 s | 664,991 | 通过 | 逐位一致 | 34.860681 / 4.032468 | **8.6450x** | 2,170 s |
| `D1004_two_track_velocity` | `np.diff` | 955 s | 745,668 | 通过 | 非 NaN 位逐位一致，NaN 位置一致 | 27.339721 / 8.133022 | **3.3616x** | 3,102 s |
| `D1009_periodic_transport` | `np.roll` | 890 s | 716,941 | 通过 | 逐位一致 | 26.669054 / 7.455630 | **3.5770x** | 3,112 s |

口径：

- **完整端到端**：从该样例开始算子生成到 CPU/NPU benchmark 完成的墙钟时间，包含生成、
  编译/修复、部署、NPU 对拍、适配层构建、完整程序迁移和 benchmark。
- **生成 tokens**：三组各 17 次 HTTP 200，可读 `usage.total_tokens` 合计 2,127,600；
  适配层为确定性集成，不调用模型，tokens 为 0。
- 三组完整程序均 `comparison.consistent=true`；`diff` 的 NaN payload 与 CPU 不同，
  但 NaN 位置和非 NaN 位一致，按语义等价记录。
- 生成的 Ascend C 源码、修复哈希、编译/部署/对拍/适配层/benchmark 证据见
  `benchmark/gap-evidence-20261114/<case>/asc_operator/`，汇总见
  `benchmark/gap-evidence-20261114/ASC_OPERATORS.md`。
