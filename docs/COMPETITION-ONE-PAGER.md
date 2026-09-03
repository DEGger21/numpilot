# NumPilot 比赛一页纸

## 30 秒说明

NumPilot 把普通单文件 NumPy 科学计算脚本迁移到华为 Ascend NPU，并把“大模型生成代码”变成
一条可审计、失败关闭的工程流程：静态能力判定决定能否迁移，配置的大模型只生成命名空间迁移
产物，本地 AST、数值、runtime 来源、调用次数、同步和性能门决定该产物能否被采用和宣传。

当前主展示不是“只挑最快数字”：Poisson-Ritz 源项与鲁棒残差阶段的设备驻留中位加速是
`2.567×`，但一次输入准备 + 一次计算 + 最终回传的组件和只有 `0.789×`，即 CPU 更快。
另一个纯逐元素合成压力链的 `8.079×` 只作为 AsNumpy 的设备驻留优势区间，不替代应用形态结果。

它解决的不是“让模型改一行 import”，而是模型生成代码在异构计算中最难的三个问题：

- 目标 API 是否真的存在、是否真正运行在 NPU；
- 生成代码是否悄悄改变语义或引入 Host fallback；
- 加速数字是否绑定同一份源码、产物、环境和公平的 CPU/NPU 计时协议。

## 已跑通的产品主链

```text
ordinary NumPy source
  → 五态静态分析（不执行、不调用模型）
  → configured GLM 真实迁移（HTTP/TLS/model receipt）
  → exact AST 门
  → 双进程 named-output 数值验证
  → 单核 CPU / strict-NPU resident benchmark
  → write-once、SHA-256 绑定证据
```

缺口路径另有 official-first 设计：先查已安装 CANN ACLNN，再查固定版本华为官方算子源码，只有
两层都明确无匹配时才允许模型按 CANNBot Skill 生成候选。候选、构建成功、数值资格化和激活是
四种不同状态，不会把“生成了代码”写成“算子已经可用”。

## 评委可核查的当前结果

两条结果都使用 9,000,000 个 `float32` 元素，不含 `matmul`；协议都是 CPU/NPU 分离进程、
同一逻辑核、CPU OpenBLAS 实测 1 线程、各 3 次预热和 20 个原始样本，并在每个 NPU 样本前后
同步。configured `glm-5.3` 的实际调用均为 HTTP 200、TLS enabled、requested/response model
exact；exact AST、数值和 16/16 性能门全部通过。

| 证据层 | resident CPU / NPU | 中位 / 保守加速 | 一次输入准备 + resident + 最终 D2H 的组件和 |
|---|---:|---:|---:|
| **主结果：Poisson-Ritz 源项与鲁棒残差阶段** | `458.698 / 178.658 ms` | `2.567× / 2.471×` | `0.789×`，CPU 更快 |
| **优势区间：纯逐元素合成压力链** | `616.333 / 76.289 ms` | `8.079× / 7.503×` | `1.220×` |

- Poisson 阶段的 bootstrap 95% CI 是 `[2.538×, 2.592×]`；9,000,000 个输出元素数值验证通过，
  最大绝对误差 `1.9073486e-06`、最大相对误差 `5.6004706e-06`。
- 合成压力链的 bootstrap 95% CI 是 `[7.954×, 8.147×]`；最大绝对误差
  `2.3841858e-07`。它只刻画大数组、输入已在设备、重复复用且最后一次回传的优势区间。
- 两个“组件和”都明确 `directly_sampled_end_to_end=false`，不是直接采样的冷启动或端到端时延，
  不得拿来替换 resident headline。
- Poisson 阶段是从 PDE 工作流抽出的应用形态计算段，**不是完整 Poisson 求解器，不使用公开
  数据集，也没有真实受益方声明**；完整 3000 轮 Poisson 的既有证据仍显示 CPU 更快。

完整冻结文件分别在
[`evidence/competition-poisson-stage-20260828/`](../evidence/competition-poisson-stage-20260828/README.md)
和
[`evidence/competition-ordinary-20260828/`](../evidence/competition-ordinary-20260828/README.md)。
上表数字来自上述日期的冻结回执（3 次预热、20 个样本）；当前 `run_demo.sh --full` 默认使用
3 次预热、30 个样本，提交前应按新协议重新冻结。现场 `./run_demo.sh --quick` 使用更短的协议，
结果会随设备温度、运行时和样本数变化，必须在演示截图中同时展示 `warmup/samples`，不能把
quick 结果改写成冻结数字。
[标准库只读审计器](../tools/audit_script_benchmark_receipt.py)能在不执行用户代码、不导入
NumPy/AsNumpy、不联网、不访问 NPU 的情况下，重算 artifact hash、全部计时统计和协议绑定：

```bash
python tools/audit_script_benchmark_receipt.py \
  evidence/competition-poisson-stage-20260828 --json
python tools/audit_script_benchmark_receipt.py \
  evidence/competition-ordinary-20260828 --json
```

两份证据捕获时均为 `git_dirty=true`，所以只是开发态真机验收，不冒充最终 release candidate。

## 3–5 分钟演示

### 0:00–0:35：问题与源码

打开 `examples/poisson_source_smooth_l1_map.py`。指出它仍是普通 `import numpy as np`，其计算
对应 Poisson 解析源项和 SmoothL1 残差图；marker 只定义公平的 resident benchmark 边界，不是
迁移语法要求。主动说明它是应用形态阶段，不是完整求解器。

### 0:35–1:05：五态静态门

```bash
# The analysis receipt is write-once; keep it in a fresh private directory.
RUN=$(mktemp -d /tmp/numpilot-analysis.XXXXXX)
chmod 700 "$RUN"
python -m numpilot analyze examples/poisson_source_smooth_l1_map.py \
  --require-direct --out "$RUN/qualified-analysis.json"
```

强调分析不执行不可信脚本、不调用模型；只有所有调用点都是当前 hash-bound runtime 下的
`DIRECT` 才进入迁移。

### 1:05–2:05：真实 GLM 与本地反证门

```bash
RUNTIME_ROOT=/absolute/persistent/numpilot-runtime
source "$RUNTIME_ROOT/activate-numpilot.sh"
./run_demo.sh --llm-config "$RUNTIME_ROOT/.numpilot-config/llm.json" --quick
```

现场先从已授权的运行时目录读取私有 `llm.json`；不要把配置文件或 Token 放入源码包。若已
source 运行时 activation，也可只使用 `./run_demo.sh --quick`。

在迁移步骤展示 configured/requested/response model、HTTP/TLS receipt 和迁移产物；指出大模型
负责生成，本地 exact AST 负责否决越权改写，失败时不回退到伪造的 AST 产物。

### 2:05–3:20：数值与性能证据

展示 `verify-script` 的 named-output 结果，以及 `benchmark-script` 的
`performance_valid=true`、原始样本、CPU 单线程实测、runtime identity、精确调用/同步次数、
`2.567×` 中位和 `2.471×` 保守结果。主动说明 `0.789×` 只是非 E2E 组件和，并且诚实显示首次
单次执行 CPU 更快。若时间允许，再用冻结证据说明合成链的 `8.079× / 7.503×` 是优势区间，
其 `1.220×` 组件和也不是端到端 headline。

### 3:20–4:00：创新与失败关闭

打开 [三页架构图](numpilot-architecture.drawio)：第一页讲 ordinary 主链，第二页讲
official-first 缺口恢复，第三页讲安装、许可证和可信边界。用一个缺失/未知 API 的分析结果说明
NumPilot 会返回 `GAP_REQUIRED`、`CAPABILITY_AUDIT_REQUIRED` 或 `BLOCKED`，不会硬迁移。

### 4:00–4:30：无网络/NPU 备份

若现场外部服务不可用，只运行上面的 Poisson 冻结证据审计器，并明确称其为“已捕获证据的离线复核”，
不是本次现场 GLM/NPU 重跑。视频应预先录下完整 live 主链。

## 与评分项的对应

- **完整性**：安装/自检、ordinary 分析、实际模型迁移、数值、性能、证据和失败回执已经进入
  产品 CLI，而不是案例专用 notebook。
- **创新性**：五态决策、模型生成与本地事实分权、official-first 缺口恢复、精确模型/源码/
  runtime identity，以及 speedup fail-closed 共同组成可信 AI 软件工程闭环。
- **真实性**：真实 GLM 调用与 Ascend 910B4 执行已有哈希证据；Poisson 应用形态阶段明确保留
  `0.789×` 一次性组件和及完整求解器 CPU 更快的历史结果，证明系统不会只发布好看的数字。

## 提交前仍必须关闭

1. 选定一个固定来源、许可证、数据哈希和明确受益方的真实负载；当前主结果虽有 Poisson 数学
   语义，仍只是解析生成的应用形态阶段，不是完整求解器或公开数据案例，不能单独吃满真实性分。
2. 在功能冻结后的干净 commit/tag 上重跑主链，构建最终 wheel/sdist/交付 ZIP、SBOM 和逐文件
   manifest，并在第二个全新环境按文档复现。
3. 在实际 CodeArts AgentTeam 项目中加载仓库内四个 agent 与 migration-gate Skill，保存平台
   运行截图/导出记录；仓库文件本身不等于平台验收。
4. 增加直接采样的 cold/E2E、多核 CPU 对照和 exact-artifact profiler，作为 resident headline
   之外的补充解释，不把组件和改写成端到端结果，也不扩大当前 `2.567× / 8.079×` 口径。
5. 让 PPT、演示视频、源码 revision、证据 manifest 和最终 release 指向同一个不可变版本。

评分维度与当前可核查证据见
[`COMPETITION-READINESS.md`](COMPETITION-READINESS.md)，最终发布门见
[`RELEASE-CHECKLIST-DEMO-V1.md`](RELEASE-CHECKLIST-DEMO-V1.md)。原始赛题映射和内部阶段计划属于
本地交付工作底稿，不随对外源码树发布。
