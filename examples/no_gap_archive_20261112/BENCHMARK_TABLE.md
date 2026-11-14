# 无缺口迁移性能记录

本文件是无缺口批次的仓库记录入口。2026-11-14 复核：本批没有新增无缺口样例，
原有 71 例数据保持不变；当前大规模性能表仍以
[`examples/scaled_benchmark_20261112/BENCHMARK_TABLE.md`](../scaled_benchmark_20261112/BENCHMARK_TABLE.md)
为准。

| 项目 | 数值 |
| --- | ---: |
| 无缺口完整 NumPy 程序 | 71 |
| 有 CPU/NPU 实测数据 | 66 |
| 加速比中位数 | 11.06x |
| 2026-11-14 新增 | 0 |
| 最近复核日期 | 2026-11-14 |

缺口样例的记录见
[`examples/gap_archive_20261112/BENCHMARK_TABLE.md`](../gap_archive_20261112/BENCHMARK_TABLE.md)；
其中 2026-11-14 的 5 个 benchmark 签名缺口样例使用 1024×1024 输入，证据在
`benchmark/gap-evidence-20261114/`。
