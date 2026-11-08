# 逐样例性能证据

`benchmark_suite` 不会自动重跑性能脚本，也不会把 Phase 9 的全局矩阵映射到某个 benchmark case。若要接入性能资格，需要先由独立的 `script_benchmark` 生成 receipt，再用 manifest 显式绑定 sample。

## 目录格式

```text
performance-evidence/
├── manifest.json
└── receipts/
    └── A001.json
```

`manifest.json` 的最小结构：

```json
{
  "schema": "numpilot.performance-evidence-manifest",
  "schema_version": 1,
  "entries": {
    "A_direct_simple/A001": {
      "path": "receipts/A001.json",
      "sha256": "<receipt-file-sha256>",
      "selector": {
        "profile": "sin-matmul-fp32",
        "size": 1024
      }
    }
  }
}
```

receipt 必须是 `numpilot_ordinary_script_benchmark`、`schema_version=1`、`status=passed`，并同时满足：

- `performance.performance_valid=true` 且没有失败 issue；
- `gates` 的每个值为 `true`；
- 所有 `comparisons.*.consistent=true`；
- `evidence_boundary.steady_state=true`、`device_resident=true`、`end_to_end=false`；
- `workers.npu.runtime_identity_before/after` 都是 `matched`，并且 SHA-256 相同；
- `binding.source.sha256` 与当前 benchmark case 源码一致。

## 运行

```bash
python -m numpilot.benchmark_suite \
  --backend npu \
  --root benchmark/cases \
  --out runs/phase18-with-performance \
  --performance-evidence-root performance-evidence
```

没有 manifest 条目时，case 的 `performance_qualified` 保持 `null`。存在 receipt 但 backend、strict-NPU、数值 gate、source hash 或 runtime identity 不匹配时为 `false`。只有全部条件匹配才为 `true`。shim、Phase 9 `profile_required` 矩阵和没有 source hash 的全局报告不能通过此门禁。

系统只把 receipt 的相对路径、SHA-256、source/runtime hash、范围和摘要指标写入 artifact store，不复制原始模型响应、token、endpoint 或完整 timing samples。
