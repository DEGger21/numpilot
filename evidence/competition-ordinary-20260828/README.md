# Ordinary configured-GLM → strict-NPU 开发态证据

这是 2026-08-28 在 Ascend 910B4 上执行的新 ordinary 主路径证据。输入是仓库内
`float32`、9,000,000 元素的 power/log/exp 等逐元素合成算例；它用于展示 NumPilot 最适合的
设备驻留区间，不宣称来自真实行业数据，也不涉及 `matmul`。

## 全链路事实

1. `qualified-analysis.json` 对 9 个 NumPy 调用点作静态资格化；结论为 `DIRECT`，每个调用点
   都是 `performance_eligible=true`，分析阶段没有执行脚本或调用模型。
2. 实际迁移仅由配置文件选中的 `glm-5.3` 完成。schema-v2 迁移回执记录
   `actual_remote_call=true`、HTTP 200、TLS 校验开启，requested/response model 精确一致；
   产物状态为 `migrated_strict_npu`。
3. 独立 `verify-script` 双进程对拍通过：9,000,000 个 `float32` 输出元素最大绝对误差
   `2.384185791015625e-07`，容差为 `rtol=1e-4, atol=1e-5`。该回执只声明数值一致，
   所以自身的 `strict_npu_verified` 与 `performance_valid` 均刻意为 `false`；strict-NPU 和
   性能由下一条更强的 benchmark 门独立证明。
4. `benchmark-script` 的 16/16 门全部通过：CPU/NPU 分离进程并固定同一逻辑核 0；四项线程
   环境和 OpenBLAS 运行时都实测为 1；NPU runtime 前后身份稳定；AsNumpy 调用数和
   47 次设备同步均与协议精确一致；最终 named output 数值一致。

## 性能结论与边界

协议为 3 次预热、CPU/NPU 各 20 个有序样本。headline 只统计 marker 之后的 steady
device-resident 计算，包含输出分配和每次设备完成同步，不包含冷启动、一次输入准备或最终
D2H；它不是 kernel-only profiler 结果，也不是直接采样的端到端时延。

- CPU 单核中位数：`616.333 ms`；
- NPU 中位数：`76.289 ms`；
- 中位加速：`8.079×`；
- 保守 `CPU p10 / NPU p90`：`7.503×`；
- 确定性 bootstrap 95% 区间：`[7.954×, 8.147×]`；
- 输入准备 + resident 中位数 + 最终输出传输的组件和：CPU `1001.431 ms`、
  NPU `820.687 ms`，诊断比 `1.220×`。它在 JSON 中明确标为
  `directly_sampled_end_to_end=false`，不能作为端到端 headline。

公开精简包不保存两份 36 MB 输出数组，只保存其 SHA-256、shape/dtype/finite 元数据和数值
比较回执；因此第三方可以独立复算计时统计与所有结构/身份绑定，但不能只凭本目录重新计算
逐元素误差。需要重放数值或真机性能时，应在匹配的 AsNumpy/CANN/NPU 环境重新运行下面命令。

## 身份与成熟度

- NPU：`Ascend910B4`；CANN：`9.0.0`；
- AsNumpy：`0.3.1.dev4+g1929a3ba4`，commit
  `1929a3ba44eb532a9f544f774a249cdf1c4119be`；
- 迁移模型：`glm-5.3`；
- NumPilot 回执版本：`0.2.0a2`；捕获时 Git commit
  `374e4da7df827bbf69668eb896e86ab1f95fa755`，且 `git_dirty=true`。

最后一点意味着这是一份诚实的**开发态真机证据**，不是不可变 release candidate。正式参赛
RC 仍需在干净、已签 tag 上重跑，并在第二个全新环境复验安装和主路径。

## 只读复核

在仓库根执行标准库审计器；它不执行用户代码、不导入 NumPy/AsNumpy、不访问网络或 NPU：

```bash
python tools/audit_script_benchmark_receipt.py \
  evidence/competition-ordinary-20260828 --json
```

要在匹配真机环境重新采样，使用冻结副本并写入一个不存在的新回执路径：

```bash
python -m numpilot benchmark-script \
  evidence/competition-ordinary-20260828/source.py \
  --migration-report \
    evidence/competition-ordinary-20260828/migration/script-migration-report.json \
  --migrated evidence/competition-ordinary-20260828/migration/migrated.py \
  --output result --warmup 3 --samples 20 \
  --out $RUN/ordinary-benchmark.json
```

目录内 `MANIFEST.sha256` 绑定本案例的全部文件；仓库级
`evidence/MANIFEST.sha256` 再绑定整个公开证据集合。
