# Poisson-Ritz 源项与鲁棒残差阶段开发态证据

这是 2026-08-28 在 Ascend 910B4 上跑通的第二条 ordinary 产品证据。它从本仓库既有
Poisson-Ritz 负载中抽出一个数学上完整、适合设备驻留复用的阶段：

- 计算 `u(x,y)=sin(πx)sin(πy)` 对应的 Poisson 解析源项
  `f(x,y)=2π²sin(πx)sin(πy)`；
- 对候选源项和解析源项的残差生成 `SmoothL1(beta=1)` 未规约逐配点损失图。

这比纯非线性压力链更接近真实 PDE 工作流，但仍没有外部数据集或明确受益方，且不是完整
Poisson 求解器。不得把本阶段的结果外推到完整 3000 轮 Poisson 案例；后者的公开证据仍诚实
显示 CPU 更快。

## 实际链路

1. 9,000,000 个 `float32` 配点；prefix 只创建解析坐标/常量场，marker 后有
   `multiply×7`、`sin×2`、`subtract×2`、`absolute/minimum/add×1`；
2. 静态分析得到 21/21 调用点 `DIRECT` 且全部 `performance_eligible=true`，没有
   `matmul`、规约、文件 I/O、Host fallback 或中途 D2H；
3. 迁移产物由配置的 `glm-5.3` 实际远程生成：HTTP 200、TLS enabled、
   requested/response model exact，schema-v2 状态 `migrated_strict_npu`；
4. 本地 exact AST 门通过，模型产物未经人工修改；
5. 独立数值验证通过；9,000,000 个输出元素最大绝对误差
   `1.9073486328125e-06`，最大相对误差 `5.6004705584200565e-06`；
6. 产品 `benchmark-script` 16/16 门通过，CPU/NPU 分离进程固定同一逻辑核 0，
   OpenBLAS 实测 1 线程，runtime identity 前后稳定，调用数和 47 次同步精确一致。

## 性能与优势区间

3 次预热、CPU/NPU 各 20 个原始样本：

- CPU 单核 resident 中位：`458.698 ms`；
- NPU resident 中位：`178.658 ms`；
- resident median speedup：`2.567×`；
- 保守 `CPU p10 / NPU p90`：`2.471×`；
- 确定性 bootstrap 95% 区间：`[2.538×, 2.592×]`。

这里的优势来自同一批解析坐标/常量场已经在设备上、残差阶段被迭代复用。单次输入准备 +
一次 resident + 最终一次 D2H 的组件和反而是 CPU `759.784 ms`、NPU `963.263 ms`，
诊断比仅 `0.789×`。该值在回执中明确
`directly_sampled_end_to_end=false`；它既不能作为直接 E2E 时延，也清楚表明首次单次调用
不是 NPU 优势区。当前证据没有直接采样多轮端到端，不发布推导的迭代 break-even 数字。

因此比赛展示应同时说清：

- 大数组、计算密集、重复使用设备输入、最后一次回传，是 AsNumpy 的优势区；
- 冷启动、一次性输入创建、细粒度调用或频繁 CPU↔NPU 往返，可能让 CPU 更快。

## 运行身份

- NumPilot：`0.3.0a1.dev0`；
- AsNumpy：`0.3.1.dev4+g1929a3ba4`，commit
  `1929a3ba44eb532a9f544f774a249cdf1c4119be`；
- Ascend：`Ascend910B4`；CANN：`9.0.0`；
- configured model：`glm-5.3`；
- 捕获时 Git HEAD `374e4da7df827bbf69668eb896e86ab1f95fa755` 且
  `git_dirty=true`。

`bootstrap-receipt.json` 绑定本次重新从固定 clean AsNumpy source 构建的 wheel、
Python/module/native hashes、NumPilot 版本和真实 README smoke；benchmark runtime receipt
再绑定它的 SHA-256。这仍是开发态证据，不是干净 tag 上的最终 RC。

## 离线复核与真机重放

```bash
python tools/audit_script_benchmark_receipt.py \
  evidence/competition-poisson-stage-20260828 --json
```

审计器不执行用户代码、不导入 NumPy/AsNumpy、不联网、不访问 NPU；它会独立验证 case
manifest、bootstrap/runtime/source/report/artifact 哈希链并重算所有计时统计。

匹配的 NPU 环境可用冻结副本重放，并写入不存在的新回执：

```bash
python -m numpilot benchmark-script \
  evidence/competition-poisson-stage-20260828/source.py \
  --migration-report \
    evidence/competition-poisson-stage-20260828/migration/script-migration-report.json \
  --migrated evidence/competition-poisson-stage-20260828/migration/migrated.py \
  --output result --warmup 3 --samples 20 \
  --out $RUN/poisson-stage-benchmark.json
```
