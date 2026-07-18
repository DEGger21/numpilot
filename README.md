# NumPilot demo

NumPilot 是一个面向科学计算程序的 NumPy → AsNumpy/Ascend NPU 自动迁移原型。当前版本强调“结果与证据可信”：它把顶层 `forward()` 作为一个完整 NPU 岛，在函数边界上传 NumPy 数组，运行迁移产物自己的 `build_inputs + forward`，并在发布性能数字前检查实际 AsNumpy wheel 的 capability/provenance。

完整愿景见 `NumPilot20260615-v1.pdf`；本仓库尚未实现通用数据流图划分和自动生成 Ascend C 缺口算子的完整闭环。

继续开发或交给新的 AI 前，请先读 [`docs/AI-HANDOFF.md`](docs/AI-HANDOFF.md)。它记录当前工作树、已完成门禁、裸金属证据、未完成项和续接命令。

## 当前可信边界

- 只分析和改写顶层 `forward()`；`build_inputs()` 与顶层展示代码保留在 CPU。
- 默认采用 `whole_forward_npu`，暂不宣称支持逐算子 CPU/NPU 混合卸载。
- AST 引擎在 `forward` 入口对 NumPy ndarray 做一次幂等 H2D；设备数组和标量保持原样。
- 验证先检查嵌套结构、shape、dtype，再按实际浮点 dtype 使用 `atol + rtol`；整数/布尔精确比较，显式处理零值、NaN 和 Inf。
- 真机 capability 绑定 AsNumpy 版本和运行环境；CPU fallback、版本未知或 API 缺失时 fail closed。
- strict `forward` 禁止残留 NumPy/host escape、未建模 helper/控制流、全局预计算和丢弃的设备结果；两条验证路径都记录实际 `ap.*` callsite 次数。
- `bench-npu` 仅在 strict-NPU、数值和 CPU 基线一致性门禁全部通过后计算 speedup。
- `--allow-cpu-fallback` 只用于诊断，报告始终为 `performance_valid=false`，不会发布 NPU 加速比。

已知重要事实：当前审计的 AsNumpy `0.3.1.dev4+g1929a3ba4` 中，`matmul` 会回到 host 调用 NumPy。因此 `mlp_forward.py` 和 `big_gemm.py` 可用于迁移/诊断展示，但不能作为 NPU GEMM 加速证据。

## 快速运行

离线 CI 使用 numpy shim，不需要 NPU：

```bash
python -m numpilot migrate examples/mlp_forward.py \
  --backend shim --out /tmp/numpilot-shim
python -m pytest -q
```

裸金属 NPU 上，建议先使用没有已知 fallback 的逐元素 smoke：

```bash
python -m numpilot doctor --backend npu
python -m numpilot migrate examples/elementwise_forward.py \
  --backend npu --out /tmp/numpilot-npu

python -m numpilot bench-cpu examples/elementwise_forward.py \
  --out /tmp/numpilot-bench --cpu-threads 1
python -m numpilot bench-npu examples/elementwise_forward.py \
  --out /tmp/numpilot-bench
```

性能结论只读取：

- `/tmp/numpilot-bench/npu_benchmark.json`
- `/tmp/numpilot-bench/benchmark_report.md`

其中必须满足 `performance_valid=true`，才可以引用 `speedup`。

也可直接运行裸金属演示脚本：

```bash
NUMPILOT_PY=/home/lixing/np310/bin/python ./run_demo.sh
```

脚本使用 `set -euo pipefail`；任何 doctor、迁移或 strict-NPU 门禁失败都会停止，不会继续打印“演示成功”。默认不使用 Docker。

## 示例

| 文件 | 用途 | strict-NPU 预期 |
|---|---|---|
| `examples/elementwise_forward.py` | square/add/sqrt/tanh 的裸金属 smoke | 当前审计无已知 fallback，仍以运行时 manifest 为准 |
| `examples/mlp_forward.py` | 两层 MLP 的迁移与正确性示例 | 因 `matmul` CPU fallback 被拒绝性能发布 |
| `examples/big_gemm.py` | 历史 GEMM 压测输入 | 只允许诊断，不是 NPU GEMM 加速证据 |
| `examples/unsupported_demo.py` | `np.fft.fft` 缺口负例 | 分析阶段 early stop |

## 交互向导

```bash
python -m numpilot wizard
python -m numpilot wizard examples/elementwise_forward.py --engine ast --backend npu
```

向导展示分析、整段 NPU 岛决策、迁移、产物级语义校验和 capability 证据门禁。它不在交互进程内顺手计算加速比，而是给出独立 `bench-cpu`/`bench-npu` 命令。

界面中的“命中 AsNumpy 扩展 API”表示复用了已有 API，不等于本次新生成了自定义算子。新生成算子数必须由独立的 OperatorArtifact、构建和测试证据统计。

## CodeArts 多智能体接入

仓库已按华为云码道项目级格式提供：

```text
.codeartsdoer/
├── agents/
│   ├── numpilot-orchestrator.md
│   ├── numpilot-analyzer.md
│   ├── numpilot-operator-engineer.md
│   └── numpilot-evidence-auditor.md
└── skills/
    └── numpilot-migration-gate/SKILL.md
```

建议让 CodeArts AgentTeam 负责角色协作、任务可视化和人工确认，让 NumPilot CLI/contract/capability registry 负责确定性执行与证据落盘。完整映射、官方资料链接和比赛演示脚本见 [`docs/codearts-integration.md`](docs/codearts-integration.md)。

## 架构

```text
CodeArts AgentTeam / 项目级 Agents / Skill
                      │
                      ▼
Orchestrator
  ├─ AnalysisAgent      只分析 forward 调用点
  ├─ DecisionAgent      whole_forward_npu
  ├─ MigrationAgent     AST（默认）或 strict LLM
  └─ VerificationAgent  artifact execution + strict comparison
                      │
       ExecutionBackend + CapabilityManifest
             ├─ ShimBackend（离线语义仿真）
             └─ NpuBackend（真实 AsNumpy/CANN/NPU）
```

核心结构：

- `numpilot/contracts.py`：智能体之间的数据契约；
- `numpilot/capabilities.py`：可序列化 capability/provenance；
- `numpilot/comparison.py`：统一严格比较规则；
- `numpilot/engines/`：AST/LLM 迁移策略；
- `numpilot/backends/`：shim/NPU 端口适配器；
- `numpilot/bench.py`：独立进程基线与 fail-closed NPU 评测；
- `docs/numpilot-technical-review.md`：完整技术评审和分阶段计划。

当前保留 `--threshold` 仅为 CLI/契约兼容；`whole_forward_npu` 策略不会再用它制造不完整的逐算子混合 placement。

## LLM 引擎

```bash
export NUMPILOT_LLM_BASE_URL="https://your-gateway/v1"
export NUMPILOT_LLM_AUTH_TOKEN="..."
export NUMPILOT_LLM_MODEL="..."
python -m numpilot migrate examples/elementwise_forward.py --engine llm
```

CLI 显式选择 `--engine llm` 时使用 strict 语义：未配置、调用失败或返回非法代码会退出，不回退 AST。TLS 证书默认校验；只有受信内网自签网关才应显式设置 `NUMPILOT_LLM_INSECURE=1`。凭据只通过环境变量注入，不得入库。

## 仍未完成

- callsite/value 级 IR、动态 shape guard 和真正的 CPU/NPU 子图划分；
- OS 级用户代码隔离与完整网络/文件权限收敛；
- H2D/D2H 字节数、kernel trace 和 profiler 自动采集；
- 从 GapSpec 到 Ascend C 生成、构建、UT/ST、安装、注册和回滚的自动闭环；
- 通用 ndarray 结构语义、异步 stream、内存池以及 AsNumpy 插件 ABI。

这些边界不会被 shim 通过率或“替换了多少 API”掩盖。
