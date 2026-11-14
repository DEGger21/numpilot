# Benchmark 部分缺口迁移证据（2026-11-14，1024×1024 输入）

每个样例一个目录，包含 NumPy 源码、模型消解缺口后的源码、AsNumpy 迁移代码和
`reports/`（迁移计划、迁移报告、验证/性能回执、token 指标）。

| 样例 | 缺口 | 完整端到端 | CPU / NPU | 加速比 | 模型 tokens |
| --- | --- | ---: | ---: | ---: | ---: |
| `F001_add_out` | `add(out=)` | 72.4 s | 4.312869 / 0.246285 ms | 17.5117x | 16389 |
| `F004_sum_initial` | `sum(initial=)` | 69.2 s | 0.541932 / 0.566366 ms | 0.9569x | 22391 |
| `F005_max_initial` | `max(initial=)` | 77.8 s | 0.675472 / 0.737641 ms | 0.9157x | 20844 |
| `F008_linspace_endpoint` | `linspace(endpoint=)` | 48.7 s | 8.714843 / 0.279280 ms | 31.2046x | 18134 |
| `F010_nan_to_num_inplace` | `nan_to_num(copy=)` | 67.0 s | 11.309616 / 0.193215 ms | 58.5338x | 19378 |

所有输入均已放大到 1024×1024 float32（1,048,576 元素）。F004/F005 的输出是
1024 元素归约结果，NPU 时间由 launch/归约开销主导，因此加速比略低于 1；
F001/F008/F010 在同等 1M 输入下体现正加速。本批是签名契约缺口，不需要生成
Ascend C 算子；后续真正需要算子生成的样例会在各自目录下增加 `asc_operator/`。
原始运行目录保留在服务器，本目录是脱敏副本。

真实算子生成批次（Ascend C 源码、任务书、生成回执）见 [ASC_OPERATORS.md](ASC_OPERATORS.md)。

## 真实 Ascend C 算子批次（3 例）

`spacing`、`diff`、`roll` 三个真实缺口 API 已完成生成 → 编译 → 私有 OPP 部署 →
NPU 原生对拍 → AsNumpy 适配层 → 完整程序 CPU/NPU benchmark：

| 样例 | 缺口 | CPU / NPU (ms) | 加速比 | 生成 tokens | 完整端到端 |
| --- | --- | ---: | ---: | ---: | ---: |
| `D1002_ulp_budget` | `spacing` | 34.860681 / 4.032468 | 8.6450x | 664991 | 2,170 s |
| `D1004_two_track_velocity` | `diff` | 27.339721 / 8.133022 | 3.3616x | 745668 | 3,102 s |
| `D1009_periodic_transport` | `roll` | 26.669054 / 7.455630 | 3.5770x | 716941 | 3,112 s |

三组合计 **2,127,600 生成 tokens**；适配层为确定性集成，不调用模型，tokens 为 0。三组均为
`case_local_adapter_benchmarked_not_activated`；证据在各自 `asc_operator/` 下，汇总见
[ASC_OPERATORS.md](ASC_OPERATORS.md)。
