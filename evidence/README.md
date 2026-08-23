# NumPilot 可公开证据包

此目录是从完整运行工作盘中筛选出的最小可审计集合，供开发者和评委随源码克隆。它不包含
虚拟环境、上游完整克隆、普通 build 目录或重复失败轮次。所有文件由根目录
`MANIFEST.sha256` 绑定。

## 内容

### `heat-diffusion-20260813/`

- `shim/`：离线迁移产物与结构化报告；
- `bench/`：CPU baseline、NPU benchmark、迁移产物和人类可读报告；
- `cpu_baseline.npy` 被保留，因为它是本次 1024×1024 数值对拍的精确基线。

结论、硬件/软件身份、计时边界与复现命令见
[`docs/evidence/heat-diffusion-npu-20260813.md`](../docs/evidence/heat-diffusion-npu-20260813.md)。

### `poisson-ritz-p0-20260821/`

- `qualification-evidence-v5/`：两个定向算子的封存源码、生成/repair provenance、规格、
  构建/测试日志、资格化 JSON、overlay 和 OPP manifest；不包含可重建二进制运行树；
- `runs/poisson-full-iteration-strict-npu-v2/`：117 调用点完整动态一轮；
- `runs/poisson-multiround-strict-npu-v3/`：250 轮、29,250 次公开 API 调用的数值证据；
- `runs/poisson-multiround-strict-npu-v4-move/`：消除包装层深拷贝后重新执行的 250 轮
  strict-NPU 数值与对象生命周期回归；
- `profiler/`：修复前 10 轮 profiler 的配置、导出 CSV、`runtime.db` 和同轮数值证据；
- `profiler-copy-elision/`：已知方向最小探针、修复后 `runtime.db`/`rts_track.db`、新轮
  qualification 身份和机器可审计的修复前后对比。1,170 条 kernel 保持不变，
  `aclrtMemcpy` 从 1,235 降至 35；本证据不发布性能结果。
- `performance-original-scale/`：48×48 内点、49 RBF、3000 轮训练，独立 CPU/NPU worker、
  2 次预热、7 次原始样本、单线程 OpenBLAS 实测、最终数值输出和同规模 profiler 门；
  合格结论为 CPU 约快 `2.67×`。
- `performance-broadcast-optimized/`：同规模、同计时口径的广播/CSE 优化；保存迁移产物、
  3000 轮最终输出、7 次原始样本、profiler DB 和机器审计。每轮调用由 117 降到 101，
  NPU 中位时延下降 `19.71%`，但 CPU 仍约快 `2.879×`。

完整解释见 [`docs/evidence/poisson-ritz-p0-npu-20260821.md`](../docs/evidence/poisson-ritz-p0-npu-20260821.md)
和 [`docs/evidence/poisson-ritz-profiler-p0-20260822.md`](../docs/evidence/poisson-ritz-profiler-p0-20260822.md)。
原始规模性能口径见
[`docs/evidence/poisson-ritz-performance-original-scale-20260823.md`](../docs/evidence/poisson-ritz-performance-original-scale-20260823.md)。
优化前后对照见
[`docs/evidence/poisson-ritz-broadcast-optimization-20260823.md`](../docs/evidence/poisson-ritz-broadcast-optimization-20260823.md)。

## 完整运行树

源码 Git 之外仍保留完整 `numpilot-gap-runs` 与 `numpilot-runs` 冷备份。归档级哈希记录在
`ARCHIVE_MANIFEST.sha256`；这些归档包含约 2 GB 工作盘中的失败轮次、嵌套仓库、venv、
build 与二进制制品，计划作为比赛交付/release 附件单独托管。仅有哈希而没有归档文件时，
不能声称第三方已可下载完整工作盘。

## 校验

在仓库根执行：

```bash
sha256sum -c evidence/MANIFEST.sha256
```

证据报告只证明其明确记录的配置和门禁，不应外推到其他 dtype、shape、硬件、CANN/AsNumpy
版本或完整通用科学计算负载。
