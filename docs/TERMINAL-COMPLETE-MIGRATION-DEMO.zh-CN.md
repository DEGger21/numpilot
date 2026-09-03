# NumPilot 完整程序迁移：终端录屏操作手册

本文只描述**命令行/终端形式**的完整迁移录屏，不包含其他交互入口的操作步骤。你可以从头到尾
照着执行一次，保留真实墙钟时间，不需要把视频剪成固定 7 分钟。

## 0. 本次演示用的程序

使用仓库中的 `examples/recording_migration_demo.py`。它是一个普通的 Black--Scholes 风格组合
定价程序：为相互独立的 European call 合约准备 `spot`、`strike`、到期时间、波动率和无风险利率，
再计算 `d1/d2`、平滑正态 CDF 近似、折现执行价和最终报价，输出变量 `result`。这是用于演示迁移
链路的数值程序，不是生产交易定价器。

程序有两个刻意的边界：

- 输入准备在 `# NUMPILOT_DEVICE_INPUTS_END` 之前，只执行一次；
- marker 之后只保留 NumPy 命名空间的逐元素计算，作为公平的设备驻留计算区；没有 Python 循环、
  随机数或文件 I/O。

默认规模是 **4,000,000 个独立合约**，最终 `result` 为一维 `float32` 数组，包含 4,000,000 个
元素（16,000,000 bytes，约 15.26 MiB）。这个规模是录屏可操作性和 resident 评测可见性的折中；
是否有 NPU 优势必须以本次 benchmark 回执为准，不能把一次结果外推成所有输入规模或端到端加速。

如果该文件尚未出现在你的 checkout，请先确认当前工作树包含本手册对应的示例文件；不要在录屏
中临时改写案例源码，否则 source hash 和迁移回执会失去闭合。

## 1. 录屏前一次性准备

以下准备不计入“本次迁移运行”的主计时，默认你已经完成安装；它们只用于确认现场状态。

```bash
cd /absolute/path/to/numpilot
export RUNTIME_ROOT=/absolute/path/to/numpilot-runtime
source "$RUNTIME_ROOT/activate-numpilot.sh"

# 只读确认固定 AsNumpy 组件和发布附件；不联网、不改源码。
python tools/check_release_tree.py --require-asnumpy-component --json
python -m numpilot bootstrap --verify
python -m numpilot doctor --backend npu

# 确认私有模型配置可用；不要打印文件内容或 Token。
python -m numpilot --llm-config \
  "$RUNTIME_ROOT/.numpilot-config/llm.json" configure-llm --check

# 录到案例结构和固定源文件哈希；不修改文件。
sed -n '1,180p' examples/recording_migration_demo.py
sha256sum examples/recording_migration_demo.py
```

现场必须具备：匹配的 Ascend 驱动/固件、CANN、OPP、可见 NPU、固定 AsNumpy wheel 和可用的
`llm.json`。模型配置、代理、运行回执和视频原始文件都放在仓库外的私有目录。

## 2. 从头到尾计时的标准命令

这里的“迁移主链耗时”从第一次执行框架命令开始，到 `numpilot run` 完成为止；随后的人类可读
报告是只读后处理，会单独计时。这样报告能够引用已经落盘的主链计时，同时不会把报告渲染时间
混入模型迁移和 NPU benchmark。先创建一个**私有父目录**；不要预先创建 `RUN_DIR`，因为
`numpilot run --out` 要求输出目录不存在。

```bash
set -u
cd /absolute/path/to/numpilot
SOURCE="$PWD/examples/recording_migration_demo.py"
RUNTIME_ROOT=/absolute/path/to/numpilot-runtime
CAPTURE_ROOT="$(mktemp -d /tmp/numpilot-terminal-demo.XXXXXX)"
chmod 700 "$CAPTURE_ROOT"
RUN_DIR="$CAPTURE_ROOT/run"
TIMING_FILE="$CAPTURE_ROOT/timing.json"

START_UTC="$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)"
START_NS="$(date +%s%N)"

# 这是本次录屏的第一个框架动作；保持终端输出可见。
python -m numpilot bootstrap --verify
BOOTSTRAP_RC=$?

if [ "$BOOTSTRAP_RC" -eq 0 ]; then
  python -m numpilot doctor --backend npu
  DOCTOR_RC=$?
else
  DOCTOR_RC=99
fi

if [ "$BOOTSTRAP_RC" -eq 0 ] && [ "$DOCTOR_RC" -eq 0 ]; then
  python -m numpilot \
    --llm-config "$RUNTIME_ROOT/.numpilot-config/llm.json" \
    run "$SOURCE" \
    --backend npu \
    --output result \
    --benchmark \
    --warmup 3 \
    --samples 20 \
    --out "$RUN_DIR"
  RUN_RC=$?
else
  RUN_RC=99
fi

END_NS="$(date +%s%N)"
END_UTC="$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)"
ELAPSED_MS=$(( (END_NS - START_NS) / 1000000 ))

# timing.json 只写到私有父目录；不要把它提交到 Git。
printf '{"started_at_utc":"%s","finished_at_utc":"%s","elapsed_ms":%s,"bootstrap_rc":%s,"doctor_rc":%s,"run_rc":%s}\n' \
  "$START_UTC" "$END_UTC" "$ELAPSED_MS" "$BOOTSTRAP_RC" "$DOCTOR_RC" "$RUN_RC" \
  > "$TIMING_FILE"

printf '\n[总计时] %s ms (%s s)\n' "$ELAPSED_MS" "$(( ELAPSED_MS / 1000 ))"
printf '[开始 UTC] %s\n' "$START_UTC"
printf '[结束 UTC] %s\n' "$END_UTC"
printf '[运行目录] %s\n' "$RUN_DIR"
printf '[退出码] bootstrap=%s doctor=%s run=%s\n' "$BOOTSTRAP_RC" "$DOCTOR_RC" "$RUN_RC"

# 报告是只读后处理，单独计时；它不会改变迁移回执。
REPORT_START_NS="$(date +%s%N)"
if [ -d "$RUN_DIR" ]; then
  python tools/render_migration_report.py "$RUN_DIR"
  REPORT_RC=$?
else
  REPORT_RC=99
  printf '[报告] RUN_DIR 不存在，无法渲染\n' >&2
fi
REPORT_END_NS="$(date +%s%N)"
REPORT_ELAPSED_MS=$(( (REPORT_END_NS - REPORT_START_NS) / 1000000 ))
printf '[报告计时] %s ms (%s s)\n' "$REPORT_ELAPSED_MS" "$(( REPORT_ELAPSED_MS / 1000 ))"
printf '[报告退出码] %s\n' "$REPORT_RC"
```

说明：`bootstrap --verify` 和 `doctor` 也放进主计时，是为了得到一次完整“框架启动到 `run`
结束”的墙钟时间；模型等待、终端确认和失败退出都算在内。报告渲染有单独的 `[报告计时]`。
`run` 内部还会给出每个阶段自己的耗时；主链墙钟、报告后处理和阶段耗时三种数字都要保留，
不要把它们相加后当成 NPU kernel 时间。

终端没有聊天输入框；本次“发给框架的迁移指令”就是上面的 `numpilot run ...` 命令。模型、
系统约束和服务地址由私有 `llm.json` 提供，命令行不再另写一段自然语言 prompt，也不要在录屏
中回显 Token 或完整请求体。

## 3. 录屏中应该看到什么

### 3.1 启动和环境门禁

终端先显示固定组件复验、CANN/NPU/AsNumpy 状态。这里应能说明：本次使用的是本地锁定 wheel，
没有现场拉取或编译 AsNumpy 源码。

### 3.2 `run` 的四个阶段

`run` 是当前普通脚本的一键入口，顺序固定为：

```text
01-analyze    全文件静态分析，得到五态和逐调用理由
02-migrate    使用 llm.json 中配置的真实模型生成迁移候选和 AST/哈希回执
03-verify     原始 NumPy 与迁移后 AsNumpy 程序的独立进程数值对拍
04-benchmark  strict-NPU resident 性能门（启用 --benchmark 时）
```

每一步都在新建的私有 `RUN_DIR` 中写回执。默认输出会显示阶段状态和耗时；如果模型、运行时、
数值或性能门失败，流程会停止，不能手工把失败改成成功。

成功时终端摘要的形状大致如下（秒数、路径和模型名以现场实际输出为准，不要把下面数字当作
预先录好的结果）：

```text
── NumPilot 一键普通脚本流水线 ──
[源码] .../examples/recording_migration_demo.py
[运行目录] /tmp/.../run
[策略] backend=npu · engine=llm · fallback=reject · benchmark=True
[✓] 01-analyze · status=passed · <实际秒数>s
[✓] 02-migrate · status=passed · <实际秒数>s
[✓] 03-verify · status=passed · <实际秒数>s
[✓] 04-benchmark · status=passed · <实际秒数>s
[事实] model_generated=True · numerically_verified=True · strict_npu_performance_verified=True
[下一步] complete
[回执] .../workflow-report.json
```

若现场没有匹配的运行时身份，`01-analyze` 会显示 `CAPABILITY_AUDIT_REQUIRED` 并关闭后续链路；
这是预期的失败关闭行为，不要把它剪掉或改写成 `DIRECT`。

### 3.3 迁移完成后的文件

成功时重点查看：

```text
$RUN_DIR/
├── qualified-analysis.json
├── migration/
│   ├── migrated.py
│   └── script-migration-report.json
├── script-verification-report.json
├── script-benchmark-report.json
└── workflow-report.json
```

`workflow-report.json` 是阶段索引；真正的模型、AST、数值和性能事实以各阶段 JSON 为准。

要在视频里展示模型确实产出了迁移候选，可只查看候选源码、差异和哈希（不要把完整 JSON 或
模型原始响应全部刷屏）：

```bash
sed -n '1,220p' "$RUN_DIR/migration/migrated.py"
diff -u "$SOURCE" "$RUN_DIR/migration/migrated.py" | sed -n '1,220p' || true
sha256sum "$SOURCE" "$RUN_DIR/migration/migrated.py"
```

## 4. 生成人类可读的迁移报告

框架的机器回执不会自动把“数据规模”和“是否有 NPU 优势”写成一段适合答辩的文字。标准计时
脚本已经调用仓库内“只读读取回执、不修改回执”的汇总器；若你分开执行命令，可在迁移主链结束后运行：

```bash
python tools/render_migration_report.py "$RUN_DIR"
```

它会生成：

```text
$RUN_DIR/migration-report.md
```

也可以直接在终端预览：

```bash
python tools/render_migration_report.py "$RUN_DIR" --stdout
```

报告至少包含：

- 本次总状态、源文件和迁移产物 SHA-256、配置模型；
- 各阶段状态和框架内部墙钟耗时；
- 五态分析结果和迁移调用数；
- 数值一致性、最大误差、输出 shape/dtype/元素数/字节数；
- CPU/NPU resident 中位数、median speedup、保守 speedup、warmup/samples/core；
- AsNumpy/CANN/SoC 运行时身份；
- 明确的口径限制：resident headline 不含冷启动、输入准备和最终 D2H，不是端到端总耗时。

报告中的“优势”只能按下面规则解释：

```text
performance_valid = true 且 median_speedup > 1
  → 在报告列出的实际元素规模上观察到 resident NPU 优势
否则
  → 不发布 NPU 加速比，只保留失败原因
```

例如本案例若报告显示 `4,000,000 个元素`、`median_speedup=2.4×`，可以说：

> 在本次固定运行画像、单核 CPU 对照和 steady device-resident 口径下，4,000,000 个
> `float32` 元素的计算区间观测到约 2.4× 加速。

不能说“4,000,000 以下或以上都更快”，因为单次运行没有测出阈值。若要找阈值，必须复制同一份
已审查程序到仓库外，按 1M/4M/9M 等规模分别重新迁移、验证和 benchmark；每个规模都要有独立
source hash、migration report 和 benchmark receipt，不能改写本次报告。

## 5. 视频录制顺序（按真实时间，不设 7 分钟硬限制）

1. 录到案例文件和一句话说明：它是 Black--Scholes 风格的独立合约组合定价程序，默认 4M
   `float32` 输出元素。
2. 录到 `bootstrap --verify`、`doctor --backend npu` 的门禁结果。
3. 在终端执行上面的计时脚本；保留 `run` 的真实阶段输出和每阶段耗时。
4. 录到模型迁移候选的文件名、Diff 摘要、模型身份和 source/migrated hash；不要显示 Token、私有 endpoint 或完整模型响应。
5. 录到 `verify` 数值对拍通过，以及 `benchmark` 的 `performance_valid`、元素数、warmup、samples 和 speedup。
6. 运行（或在计时脚本中保留）`render_migration_report.py`，打开 `migration-report.md`，朗读
   “实际规模、resident 口径和限制”。
7. 最后展示 `[总计时]`、`[报告计时]`、退出码和回执目录；说明这些回执是本次运行的证据，不是
   任意机器的通用承诺。

视频结束后，从头回放一次，确认命令退出码、报告数字和画面一致。原始视频、私有 `CAPTURE_ROOT`、
模型配置和完整回执留在仓库外；公开提交包只放脱敏成片、封面和 SHA-256 清单。

## 6. 失败时怎么处理

- `bootstrap` 或 `doctor` 失败：停止录制，保留终端输出，先修复 CANN/驱动/组件，不进入迁移。
- `01-analyze` 不是 `DIRECT`：按报告处理 `GAP_REQUIRED`、`CAPABILITY_AUDIT_REQUIRED` 或
  `BLOCKED`，不要强行调用模型。
- `02-migrate` 失败：保留失败回执；不要把 AST 机械替换或手工代码当作模型迁移结果。
- `03-verify` 失败：不能发布迁移正确性；新尝试必须使用全新的 `CAPTURE_ROOT`/`RUN_DIR`。
- `04-benchmark` 失败：可以保留迁移和数值结果，但报告必须写“未发布 NPU 加速比”。

任何重跑都不能复用旧的 `RUN_DIR`，也不能覆盖旧的 `migration-report.md`。失败本身是录屏中
有价值的失败关闭证据。

## 7. 录屏结束后的最小检查

```bash
# 仍在仓库根目录
git diff --check
python tools/check_release_tree.py --require-asnumpy-component --json

# 标准脚本已经生成报告；只有报告缺失时才补生成，避免覆盖已有证据。
if [ ! -s "$RUN_DIR/migration-report.md" ]; then
  python tools/render_migration_report.py "$RUN_DIR"
fi
sha256sum "$RUN_DIR/migration-report.md"
```

最终答辩材料应同时交代：实际总耗时、阶段耗时、输入元素规模、dtype/shape、resident 与端到端
计时的区别、是否有真实 NPU provenance，以及哪些结论尚未在其他规模或第二环境复现。
