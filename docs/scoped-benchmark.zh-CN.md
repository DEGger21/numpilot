# 官方算子资格进入 benchmark 与性能回执

正式 API qualification、样例迁移成功、性能证据有效是三个独立结论。
本流程将一个已资格化的官方 adapter 用于原始 benchmark 样例；保留全局
`audit_required`，只对已证明的调用范围记 `strict_npu_qualified=true`。

## 选择正确运行时

从仓库根目录运行，使用构建候选时的虚拟环境 Python 和 CANN 环境。
`ADAPTER_RUN` 指向保存了 `integration-v2` 和 `qualification-v1` 的私有目录；
`BATCH_RUN` 是新的实验目录。不要修改固定 AsNumpy 源组件或共享安装来切换候选。

```bash
export NUMPILOT_RUNTIME_SITE="$ADAPTER_RUN/integration-v2/runtime/site"
export NUMPILOT_DEPENDENCY_SITE="$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
export PYTHONPATH="$NUMPILOT_RUNTIME_SITE:$NUMPILOT_DEPENDENCY_SITE:$PWD"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

python -S -m numpilot.benchmark_suite \
  --root benchmark/cases --backend npu \
  --capability-overlay "$ADAPTER_RUN/qualification-v1/capability-overlay.json" \
  --qualification-evidence-root "$ADAPTER_RUN/qualification-v1" \
  --cases I_gap_model/I006_concatenate_axis \
  --out "$BATCH_RUN/i006" --json
```

`--cases` 使用完整 `group/name`，未知、重复或逃逸路径不会被接受。和 groups/scope
组合时取交集，空集报错。官方 overlay 的 NPU 分支核对仓库原始样例和 expectations，
不能通过修改源程序、重新命名样例或修改期待来制造通过率。

## 批量运行与审计

```bash
python -S -m numpilot.benchmark_suite \
  --root benchmark/cases --backend npu \
  --capability-overlay "$ADAPTER_RUN/qualification-v1/capability-overlay.json" \
  --qualification-evidence-root "$ADAPTER_RUN/qualification-v1" \
  --scope benchmark/scopes/numpy_compute_only.json --scope-categories single_gap \
  --out "$BATCH_RUN/single-gap" --json

python tools/check_operator_artifacts.py \
  --run-root "$BATCH_RUN/single-gap" --expected-total 35 --json
```

原始语料仍为 150 例，其中计算迁移范围 135 例、静态拒绝 15 例；单缺口视图是
35 例。单个 scope 的通过数不代表整个语料已通过。gap 样例在补全后成功仍标记为
`improved`，其原期待保留。

一个样例只有同时通过以下验证，才能使用范围限定的资格：

- 官方 overlay 的 runtime、source、integration、gate 链全部匹配；
- 原始、分析、生成源码与对应执行证据的 hash 一致；
- 每个所需计算操作都在已资格化范围内，静态及实际调用 profile 一致；
- strict NPU guard、输入/计算/生成程序三个对拍和前后 runtime 身份均通过。

store 会复核完整 scope 证据；单独写 `strict=true` 或某个 scope 字符串不会绕过
原有全局 runtime 门。shim 仍只证明迁移语义，不获得 NPU 资格。

## 性能边界

性能 producer 使用同一原始样例和 NumPilot 生成的 `forward`。CPU 与 NPU worker
分离，线程数为 1，固定一个逻辑 CPU，NPU 仅一张可见卡。原始 timing samples、
预热、重复次数、runtime 和来源绑定全部保存。

- 设备驻留：输入已在 NPU，计入生成 forward、runtime guard、输出分配和设备同步。
- 暖态端到端：计入上传、同一生成 forward、下载；不包含进程及设备首次初始化。

这两种口径均不是单个 kernel 的裸执行时间。`performance_qualified=true` 只表示
证据完整有效；加速比小于 1 表示实际较慢，仍须如实记录。扩大输入规模必须生成
独立 workload 身份，不能把扩展规模成绩记到原始 I006。

先从 suite 回执导出未经修改的 scope 引用，再测量。性能工具需要项目 `npu` 可选
依赖中的 `threadpoolctl`，用于检查实际 BLAS 线程数，不能只相信环境变量。

```bash
python - "$BATCH_RUN/i006/cases/I_gap_model/I006_concatenate_axis/runtime-qualification.json" \
  "$BATCH_RUN/i006-ref.json" <<'PY'
import json, sys
with open(sys.argv[1]) as source:
    ref = json.load(source)["runtime_qualification_scope"]
with open(sys.argv[2], "x") as target:
    json.dump(ref, target, indent=2, sort_keys=True)
PY

python -S tools/qualified_i006_performance.py \
  --scoped-qualification-ref "$BATCH_RUN/i006-ref.json" \
  --runtime-site "$NUMPILOT_RUNTIME_SITE" \
  --out "$BATCH_RUN/performance" --warmup 5 --samples 30

python -S -m numpilot.benchmark_suite \
  --root benchmark/cases --backend npu \
  --capability-overlay "$ADAPTER_RUN/qualification-v1/capability-overlay.json" \
  --qualification-evidence-root "$ADAPTER_RUN/qualification-v1" \
  --cases I_gap_model/I006_concatenate_axis \
  --performance-evidence-root "$BATCH_RUN/performance" \
  --out "$BATCH_RUN/i006-with-performance" --json
```

跨 run 复用性能时，Catalog 比较当前与生产时的原始源码、生成代码、profile、runtime、
overlay 及操作集合；只匹配原始源码和 runtime 不够。store/audit 还会重验原始 worker、
timing samples、输出及来源文件。

## 第一轮实测（Stage81，保留历史）

- 原始 I006：正式 suite、范围限定 strict NPU、性能证据挂接、完整 artifact audit 全部通过。
- 单缺口 35 例：1 例迁移成功、28 例缺口停止、6 例执行前资格拒绝；不能记成 35 例通过。
- 6 例拒绝涉及隔离 runtime 尚未审计的基础 API，记录为 `runtime_qualification_failure`，
  保留 `failed/unexpected`，未通过改变 expectations 来隐藏它们。
- I006 原始输出为 3×6×5：CPU 中位数 0.003095 ms，NPU resident 0.423548 ms，
  warm E2E 0.613563 ms。5 次预热、30 次测量、CPU 线程/affinity 均为 1、单张 NPU。
  两种 NPU 范围都较慢，`accelerated=false`。数据包含 runtime guard 开销。

## 保存与交接

保存整个 benchmark run、performance run、官方 qualification bundle 与对应 adapter
源码/构建事务。scope 的相对文件引用及 hash 允许验证样例工件；官方 evidence root
搬移后必须显式重新定位并重新验链。保留失败尝试和全部日志，不改写既有证据。

当前 store 审计默认读取原引用位置。跨机器恢复时，可用
`validate_scoped_qualification(..., evidence_dir=..., official_evidence_root=...)`
显式重定位 scope 与官方证据包进行完整复核；不要直接修改已经 hash 绑定的 store 回执。
一键跨机路径映射尚未实现。

本流程复用了官方 `aclnnCat`；CANNBot 生成自定义 Ascend C/ASC 算子并接入 AsNumpy
仍是单独的路线验收，不以本次官方接口复用替代。


## 混合程序的基础 API 资格（Stage82）

官方 adapter 与已有 add/mean 等混用时，还需基础 API 的独立资格。现在可通过
成对的 --baseline-api-certificate / --baseline-api-evidence-root 参数组合；官方 overlay
先对原始 manifest 验证，再加入基础证书。全局状态保持 audit_required，样例保存 v2 scope。
精确范围、生成命令和实测结果入口见[基础 API 资格](baseline-api-qualification.zh-CN.md)。
旧 v1 证据和已有 I006 性能回执继续有效；不得把不同证书的性能数据混用。
