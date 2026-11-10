# 本地 Codex 生成 AsNumpy adapter：真实接入记录

## 本批结果

本地 Codex 读取固定 handoff 的上下文和窗口，生成六份响应；NumPilot 完成窗口校验、
RecoveryStore 导入、预检、隔离源码复制、wheel 构建、正式资格验证及原始 I006 自动迁移。全程没有调用
`llm.json`。这条路线复用已安装的官方 `aclnnCat`，不是新生成 Ascend C kernel；
CANNBot 自定义 kernel 路线仍是官方算子不满足契约时的后续分支。

| 检查 | 结果 |
| --- | --- |
| 输入源码 | 组件锁定的 211 文件，含作者 FP32 matmul 补丁 |
| 本地候选 | `adapter-c5162b3deefba3afb4eb` |
| 框架 API gate | 17 项通过；105 次 oracle；新 native 105 次、legacy 0 次 |
| 重复调用 | 100 次，峰值 RSS 起止一致 |
| 生成候选测试 | 38 passed |
| 正式资格六门 | 全部 passed；31 项 concatenate 单测、8 项数值样例、24 次 oracle 比较 |
| 准入范围 | float32、连续布局、rank 1–3；3 种签名与 11 个拒绝用例 |
| 设备执行证据 | 独立 msprof 实测 4 条 ConcatD；本轮 gate 另有 2 次 native 调用 |
| 原始 I006 | AST 自动迁移及 strict NPU 验证通过 |
| RecoveryStore | 仍为 `INTEGRATED_CANDIDATE`；正式资格产品另存 capability overlay |
| 全局 activation / 性能证据 | 未执行；原始 I006 已有正式证据，且未加速（见后续复现） |

静态组件不被生成代码覆盖。代码和工件分开保存：`adapter-v2/candidate/` 是六文件
源码 overlay；`integration-v2/source/` 是固定源码包叠加 overlay 的构建输入；
`integration-v2/dist/` 是 wheel；`integration-v2/runtime/site/` 是隔离 runtime；
RecoveryStore 通过 SHA-256 引用它们。实例保存在作者选择的私有 `RUN_DIR` 中；具体位置记录于仓库外开发日志。

资格 ID 为 `official-aclnn-91a218176530837f7b10`，实例的正式产品保存在
`qualification-v1/`，原始 I006 的自动迁移证据保存在 `i006-mainline-v1/`。
固定源码组件及共享运行时不被替换；仅对显式选择该 overlay 的调用范围准入。

## 接入方式

1. 从已验证的官方 handoff/context gate 和真实 runtime 身份，调用
   `materialize-official-candidate-recovery --profile full-local-recovery` 创建**新**恢复状态。
   旧 receipt-only 状态不可原地升级；作者已授权的本地构建不需要反复询问。
2. Codex 只写 `responses/<path>.response`。`inspect-asnumpy-adapter-handoff` 检查文件完整性，
   `validate-asnumpy-adapter-handoff` 解析并验证内容；已存在的 `__init__.py` 导出允许原样保留。
3. `recover-gaps ingest-codex-adapter` 导入 schema-5 本地候选；
   `preflight-integrate-candidate` 必须输出 `status=ready`、`code=ready_for_integration`。
4. `integrate-candidate` 使用新绝对输出目录，构建、安装到私有 site，再运行独立 gate。
   失败回执留档，修复使用新候选和新事务目录。不要修改已发布的回执。

完整参数见 [Codex CLI 参考](../skill/asnumpy-gap-completion/references/codex-cli.md)。
本机原样命令保存在上述实例的 `materialize-recovery-v2-command.sh`、`run-adapter-v2.sh`。
重新实验需要新的 write-once 输出目录。重启后若旧 handoff 的 `/tmp` 输入消失，
先按 `input-restoration.json` 从 `inputs/` 逐字节核对 SHA-256 后恢复；不要改 locator 内容。

## 独立 I006 数值与计时诊断

```bash
python tools/concatenate_adapter_smoke.py \
  --runtime-site /abs/transaction/runtime/site \
  --out /abs/new-smoke \
  --sizes base,medium,large --warmup 3 --repeats 30
```

使用项目虚拟环境 Python 和已加载的 CANN 环境。脚本固定原始 I006 源码 hash，
复用原 `build_inputs`、`forward` 字节码，只把设备执行的 `np` global 绑定为 AsNumpy。
这是 **direct adapter smoke**，不是 Code Migrator 自动迁移回执，不增加 150 例计数。
CPU 和 NPU worker 均从启动时固定到一个逻辑 CPU，线程数为 1，实际可见 NPU 为 1 张。

| 输出规模 | 单核 CPU 中位数 | NPU 设备驻留 | NPU 暖态上传+计算+下载 |
| --- | ---: | ---: | ---: |
| 原始 I006：3×6×5 | 0.003310 ms | 0.108055 ms | 0.268484 ms |
| 扩展：8×192×128 | 0.087720 ms | 0.108405 ms | 0.579238 ms |
| 扩展：64×1536×512（192 MiB） | 82.759729 ms | 0.436969 ms | 41.387589 ms |

三个规模的输出、dtype 和 shape 与 NumPy 完全一致。大规模暖态端到端诊断比率约
2.00×，设备驻留约 189.39×；这两个口径不可混用。小规模 NPU 更慢。数据包括三次
预热和三十次测量，冷启动另记，不在表内。证据仍标记 `diagnostic_only=true`，
没有 strict-NPU/performance qualification。int32 hstack/vstack/旧 native 接口回归也通过。

## 避免数值正确掩盖错误路由

首次 v1 wrapper 误接 legacy 分支，数值测试通过但不是目标 ACLNN 路由。
`v1-route-exclusion.json` 明确排除该次证据，原构建与回执保留。
v2 新增框架独立路由探针：确认新 native 绑定、禁止 legacy、记录调用次数；
候选测试也验证分发。不能单凭 `allclose` 或 wheel 构建成功宣称算子接入完成。

## 正式资格与原始 I006 自动迁移

正式 producer 原生消费 `official_candidate_gap`，保留原始 handoff/source/resolution 内容，
不会把精简 fixture 改名成 I006。validator 从归档字节重建 GapSpec 和决策；检查隔离
runtime 的解释器、wheel、native core、wrapper 和源码事务身份。

下面命令使用项目虚拟环境 Python，提前加载 CANN，并为每次实验选择**新的输出目录**。
`RUN_DIR` 是保存了前述 integration-v2 与 inputs 的私有实验目录。

```bash
export NUMPILOT_RUNTIME_SITE="$RUN_DIR/integration-v2/runtime/site"
export NUMPILOT_DEPENDENCY_SITE="$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
export PYTHONPATH="$NUMPILOT_RUNTIME_SITE:$NUMPILOT_DEPENDENCY_SITE:$PWD"

python tools/concatenate_kernel_profile.py \
  --runtime-site "$NUMPILOT_RUNTIME_SITE" \
  --transaction "$RUN_DIR/integration-v2/integration-transaction.json" \
  --out "$RUN_DIR/kernel-profile-new"

python -S tools/aclnn_overlay_qualification.py \
  --run-root "$RUN_DIR" --attempt integration-v2 \
  --gap-request "$RUN_DIR/inputs/numpilot-official-concat-handoff-abs.json" \
  --official-resolution "$RUN_DIR/inputs/numpilot-official-concat-resolution-abs.json" \
  --source-file "$PWD/benchmark/fixtures/official_candidate/concatenate.py" \
  --source-logical-path concatenate_fixture.py --entries concatenate \
  --runtime-site "$NUMPILOT_RUNTIME_SITE" \
  --kernel-profile-root "$RUN_DIR/kernel-profile-new" \
  --bundle-dir qualification-new --json

python -S tools/qualified_i006_migration.py \
  --capability-overlay "$RUN_DIR/qualification-new/capability-overlay.json" \
  --evidence-root "$RUN_DIR/qualification-new" \
  --runtime-site "$NUMPILOT_RUNTIME_SITE" --out "$RUN_DIR/i006-mainline-new"
```

profiler 记录与 Python 调用计数分别保存。只有实际 `ConcatD` 设备任务、有正的有限
耗时、原始 CSV hash 及相同事务/runtime 身份才可作为 kernel 证据。历史报告可读，
但新生产任务不会再把 native 调用次数当作 kernel 次数。当前新的 profiler 准入路线
只实现了本次 concatenate；ones/zeros 等需增加对应实测器后才能重新生产资格。

I006 runner 复用 Orchestrator、AST Code Migrator、RuntimeProfileGuard 和原始 NumPy
oracle。它保存原程序、生成代码、调用 profile、三类比较、runtime 前后身份及独立回执。
实测精确重写一个 concatenate；kernel/artifact 两条执行路径各调用一次，输入、计算
结果与生成程序输出均一致。单 ndarray 或 tensor-valued axis 不得借通用 profile 误准入。

这是原始 I006 的范围限定迁移成功。全局 runtime 仍为 `audit_required`；没有更新
150 例全量 suite 计数，也没有全局激活候选。本节的独立 runner 不赋予性能资格；后续正式 suite 与性能生产记录见[范围限定 benchmark](scoped-benchmark.zh-CN.md)。前面的 direct smoke
保留为历史诊断，不能替代本节正式迁移回执。

## 下一批工作

1. 已将范围限定资格接入正式 suite/store/性能，详见[后续复现](scoped-benchmark.zh-CN.md)。已完成首批11个基础API及3个混合程序验证（[范围与复现](baseline-api-qualification.zh-CN.md)）；下一步直接补cbrt，再扩展 single-gap → official-gap →
   multi-gap 队列；保留每例不可变 source hash、拒绝原因和候选失败状态。至少复现一例
   CANNBot 指导生成自定义 Ascend C/ASC 算子、暴露 ACLNN 两段式、补入 AsNumpy
   并验证原程序的完整路线；本次官方 aclnnCat 复用不能替代该项验收。
2. 建立单核 CPU/单卡 NPU 正式性能回执，区分设备驻留与端到端开销，再完成
   GPU4PySCF 科学计算示例的 CuPy→NumPy→NumPilot→AsNumpy 迁移故事。
