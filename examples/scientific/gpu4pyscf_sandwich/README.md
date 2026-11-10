# 科学计算示例：GPU4PySCF 轨道基变换

这个示例展示一条实际迁移链：**科学库中的 CuPy 计算核 → NumPy 单核参考 → NumPilot AST 自动迁移 → AsNumpy / Ascend NPU 验证与测量**。计算为

\[
B=C^TAC,\qquad A\in\mathbb{R}^{N\times N},\quad C\in\mathbb{R}^{N\times M}.
\]

来源是 GPU4PySCF 提交 `5fde981316127cd3506d9d3c7d5633647fcc4508` 的
[`cupy_helper.py:1224–1241` 中的 `sandwich_dot`](https://github.com/pyscf/gpu4pyscf/blob/5fde981316127cd3506d9d3c7d5633647fcc4508/gpu4pyscf/lib/cupy_helper.py#L1224-L1241)。
该操作在周期体系 SCF 的
[`khf.py:219`](https://github.com/pyscf/gpu4pyscf/blob/5fde981316127cd3506d9d3c7d5633647fcc4508/gpu4pyscf/pbc/scf/khf.py#L219)
用于 `canonicalize` 中的 `sandwich_dot(fock[k], c_k)`：把原子轨道基的 Fock 算符变换到轨道系数定义的基中，为后续子空间处理提供矩阵。

本目录保留了[原函数片段](source/sandwich_dot.original.txt)、[来源及改写说明](source/origin.json)和 [Apache-2.0 许可证](source/LICENSE)。版权归 PySCF Developers；本示例保留原版权声明。

## 本次实现范围

[`numpy_program.py`](numpy_program.py) 将原函数特化为**实数 float32、单个二维矩阵**，保持原先乘法顺序：先 `C_T @ A`，再乘 `C`，`forward` 中恰好两个 `np.matmul`。原函数的复数共轭、批次处理和 `out` 接口不在本次范围。

输入采用固定随机种子 `20261110`：合成实对称矩阵 `A` 与稠密矩阵 `C`，并在主机输入准备阶段把 `C.T` 物化为 C 连续的 `C_T`。三组输入均为 float32、C 连续；CPU 和 NPU 使用相同定义的输入。`NUMPILOT_SANDWICH_N` / `NUMPILOT_SANDWICH_M` 明确选择规模，要求 `1 <= M <= N`，程序不会自动缩小输入。

这些输入是科学计算核的可复现负载，并非真实分子积分或收敛波函数。结果只证明此特化计算核的迁移、数值一致性和性能，**不代表完整 SCF/DFT 求解加速或化学精度**，也没有测量原 CuPy 实现的 GPU 性能。

本例使用已接入 AsNumpy 的 `matmul`，由 NumPilot 的 `--engine ast` 生成迁移代码；本次不调用模型、不生成新算子。它证明已有 API 的迁移主链可以工作，**不作为 AI 自动补全缺口算子成功的证据**。

## 已取得的结果与计时边界

2026-11-10 的服务器实测使用单个 CPU 逻辑核、实际 BLAS 线程数 1，对比单卡 Ascend910B3。物理 NPU 6 通过 `ASCEND_VISIBLE_DEVICES=6` 映射为进程内逻辑设备 0，实际可见设备数为 1。环境为 Python 3.12.14、NumPy 2.5.4、CANN 9.2.0-beta.2，以及已通过 bootstrap 的隔离 AsNumpy `0.3.1.dev70+g60d139c54.d20261108`。

| N / M | CPU forward / ms | NPU resident + D2H / ms | NPU 暖 E2E / ms | 主展示加速比：CPU / 暖 E2E |
|---|---:|---:|---:|---:|
| 2048 / 1024 | 461.429377 | 1.624385 | 4.407825 | **104.684×** |
| 4096 / 2048 | 3639.596314 | 3.748148 | 11.180063 | **325.543×** |

两种 NPU 口径需要分别解读：

- **CLI resident + D2H**：输入提前驻留设备，计时包含迁移后的 `forward`、运行时 guard、输出回传与同步，不含每轮 H2D；不是 kernel-only，也不是端到端耗时。`npu_benchmark.json` 自带的加速比属于此口径。
- **暖 E2E（用于主展示）**：每轮从三组主机输入开始，包含 H2D、NumPilot 生成的 `forward`、guard、输出分配、D2H 和设备同步。每轮计时前还执行一次同步，清空此前工作；计时后同步计入耗时。

两者均排除进程/模型/设备冷启动、输入生成、`C_T` 转置物化和结果校验；CPU 基线也排除同样的输入准备。暖 E2E 并非从启动 Python 到获得最终报告的整段墙钟时间。

**统计量透明说明：**现有 `bench-cpu` 和 `bench-npu` 均预热 3 次、采样 10 次，字段名虽为 `median_ms`，实现实际取排序后第 6 个样本，即偶数样本的 *upper median*。补充暖 E2E 运行器同样预热 3 次、采样 10 次，但用 `statistics.median`，即第 5、6 个样本的平均值。上表主展示比值严格沿用回执的 CPU upper median / 暖 E2E 真中位数，没有改写计时实现或追溯替换原始统计量。原始样本均有保留。

两组暖 E2E 结果均通过 `atol=1e-5, rtol=1e-4` 的 NumPy 对照；最大绝对误差分别为 `1.15484e-7`、`9.68575e-8`。这些是本输入范围的数值结果，不把 float32 一致性升级为化学精度结论。

另一次独立 profiler 运行记录了 **2 条 `AI_CORE` / `MatMulV2`**，名称均为 `aclnnMatmul_MatMulCommon_MatMulV2`、`HF32 Eligible=NO`，输入 shape 分别是 `(2048,4096) × (4096,4096)` 和 `(2048,4096) × (4096,2048)`。它用于核实两次 matmul 的实际设备执行路径；**上表不使用 profiler 耗时**。

## 在已准备好的服务器上复现

以下命令复用已 bootstrap 的隔离 runtime 与服务器归档的两个运行器，不安装或更换 AsNumpy，也不覆盖已有 `*-v1` 证据。代码在本仓库；`cli_evidence.py`、`sandwich_e2e.py` 和 bootstrap 回执目前保存在服务器的 `documents/gpu4pyscf-sandwich-demo` 中，复现暖 E2E 时需要该证据目录一起存在。

在无其他 NPU 计算/性能采样任务干扰时，用同一 shell 顺序执行。每次通过 `mktemp` 创建全新复现根目录，每个规模先 CPU 后 NPU；**一对 CPU/NPU 必须使用同一个源文件绝对路径、N/M、Python 环境及输出目录**。改变其中任一项后，应在新目录重新生成整对基线，不能借用旧规模或旧源码的 CPU 结果。

```bash
set -e
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u

repo=/home/ma-user/work/numpilot
demo=/home/ma-user/work/documents/gpu4pyscf-sandwich-demo
science_python="$demo/runtime-venv/bin/python"
science_program="$repo/examples/scientific/gpu4pyscf_sandwich/numpy_program.py"

cd "$repo"
export PYTHONPATH="$repo"
export PYTHONNOUSERSITE=1
export NUMPILOT_BOOTSTRAP_RECEIPT="$demo/bootstrap-receipt.json"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export ASCEND_VISIBLE_DEVICES=6

# 以下文件均应已存在，runtime 的 bootstrap 回执应为 ready。
test -x "$science_python"
test -f "$NUMPILOT_BOOTSTRAP_RECEIPT"
test -f "$demo/cli_evidence.py"
test -f "$demo/sandwich_e2e.py"
science_run=$(mktemp -d "$demo/replay-XXXXXXXX")

for n in 2048 4096; do
  m=$((n / 2))
  export NUMPILOT_SANDWICH_N="$n" NUMPILOT_SANDWICH_M="$m"
  benchmark_out="$science_run/benchmark-$n"
  test ! -e "$benchmark_out"

  taskset -c 0 "$science_python" "$demo/cli_evidence.py" \
    "$science_run/cpu-$n-process.json" \
    bench-cpu "$science_program" --out "$benchmark_out" --cpu-threads 1 \
    > "$science_run/cpu-$n.stdout" 2> "$science_run/cpu-$n.stderr"

  taskset -c 0 "$science_python" "$demo/cli_evidence.py" \
    "$science_run/npu-$n-process.json" \
    bench-npu "$science_program" --out "$benchmark_out" --engine ast --skip-env-check \
    > "$science_run/npu-$n.stdout" 2> "$science_run/npu-$n.stderr"

  taskset -c 0 "$science_python" "$demo/sandwich_e2e.py" \
    --benchmark "$benchmark_out" --program "$science_program" \
    --n "$n" --m "$m" --warmup 3 --samples 10 \
    --out "$science_run/e2e-$n" \
    > "$science_run/e2e-$n.stdout" 2> "$science_run/e2e-$n.stderr"
done
```

`cli_evidence.py` 调用的就是正式 NumPilot CLI，并额外记录同进程 affinity、实际线程池与返回码；两个命令分别等价于 `python -m numpilot bench-cpu ...` 和 `python -m numpilot bench-npu ...`。NPU 命令中的 `--skip-env-check` 仅跳过重复的启动环境检查：这里已有匹配的 bootstrap 回执，运行时身份、strict-NPU 来源与数值门禁仍必须通过。不要加 `--allow-cpu-fallback` 来获得展示结果。

复现后先检查每个命令返回码，再检查 `npu_benchmark.json` 的 `status=pass`、`performance_valid=true`、`baseline_compatible=true`，以及暖 E2E `receipt.json` 的 `status=passed`、`comparison.consistent=true`、实际 affinity `[0]`、线程池为 1、`device_count=1` 和 runtime `status=matched`。以新回执报告新测量结果，不把上表数字当作每台机器或每次运行的保证。

## 原始证据定位

本次固定输入程序 SHA-256 为 `e0c6aa6f364a5c284f4dec8e32377009d53f0625d94077301161f1166b4e5dc2`；仓库程序与实测使用的 `documents/.../numpy_program.py` 字节一致。实测 runtime identity 为 `e01f3baf024ecf51a3903315b07b12b7f08d08ad311f3073a34589eb37dc3fd7`。

服务器证据根目录：`/home/ma-user/work/documents/gpu4pyscf-sandwich-demo/`。

| 证据 | 位置（相对证据根目录） |
|---|---|
| 隔离 runtime 与安装校验 | `bootstrap-receipt.json`、`runtime-venv/` |
| 单核/线程/可见设备记录 | `cpu-2048-process.json`、`npu-2048-process.json`，以及对应 `4096` 文件 |
| CPU 原始样本、输出 | `benchmark-2048-v1/cpu_baseline.json`、`cpu_baseline.npy`，以及对应 `4096` 目录 |
| 自动迁移代码、strict-NPU 与 resident 结果 | 各 `benchmark-*-v1/migrated.py`、`npu_benchmark.json`、`benchmark_report.md` |
| 暖 E2E 原始样本、输入/工件 hash、guard 与比较 | `e2e-2048-v1/receipt.json`、`e2e-4096-v1/receipt.json` 及各自 `output.npy` |
| 独立设备执行 trace | `profiler-4096-v1/PROF_*/mindstudio_profiler_output/op_summary*.csv`、`profile-once-4096-v1/receipt.json` |
| 实际运行器与命令 | `cli_evidence.py`、`sandwich_e2e.py`、`run-scientific-benchmarks.sh` |

此科学示例属于独立工作负载证据，不把其成功率或加速比计入 150 个 benchmark 样例的通过数。
