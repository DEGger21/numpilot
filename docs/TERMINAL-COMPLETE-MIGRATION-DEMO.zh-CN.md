# NumPilot 完整程序迁移：交互终端录屏手册

本文描述 NumPilot 固化在框架内的交互迁移流程。录屏入口是持久产品终端，不是一次性演示脚本：
以后换成其他普通 NumPy 程序，仍然使用相同指令、阶段、失败边界和最终报告。

## 两种使用方式

1. **终端 configured lane**：用户在仓库外填写私有 `llm.json`，然后像普通开发工具一样使用
   `numpilot`。配置模型负责源码解读、迁移候选和满足门禁时的算子草稿；NumPilot 负责静态
   分析、哈希绑定、真机资格化、数值对拍、性能测试和报告。
2. **AI/CodeArts-agent lane**：直接让正在与用户交流的 AI 阅读仓库与
   [`skill/SKILL.md`](../skill/SKILL.md)，并明确“不用 `llm.json` 里的内置模型”。此时交互 AI
   自己承担源码解读、缺口补全、算子/接入和迁移候选生成，NumPilot 只执行可接受该产物的确定性
   门禁。两种模型来源不能在同一个 run 目录中静默混用。

本文后续录屏步骤使用第一种。代理仅是具体网络环境可能需要的外部设置；NumPilot、Skill 和公开
回执均不固定代理地址，只检查所选模型是否能返回有效内容。

## 本次最小缺口程序

使用仓库中的 `examples/add_one_migration_demo.py`。它是普通 NumPy 程序，核心调用是：

```python
shifted = np.add(base, 1.0)
```

NumPilot 只把这个精确的直接调用识别为已登记的 scalar-one 签名缺口。终端应说明：当前合格的
AsNumpy 原语采用 array-array 合同，本次需要的是 `asnumpy.add_one(x)`；其范围为 `float32`、shape
保持不变、语义为 `y = x + 1`。这不是对所有 scalar ufunc 的泛化支持。

示例默认使用 9,000,000 个 `float32` 元素，并在这个简单缺口后串联已经支持的
`power/log/multiply/exp/divide/tanh` 计算，把计算区放在 `# NUMPILOT_DEVICE_INPUTS_END` 之后。
这样缺口仍然只有 `add_one`，同时能让真机 resident benchmark 覆盖一段有代表性的计算。最终性能
数字必须来自本次机器和本次报告，不能预先写死。

## 录屏前准备

先完成安装、CANN/AsNumpy/NPU 运行环境激活，并在仓库外准备权限受限的 `llm.json`。不要在录屏
中打印该文件、Token、私有 endpoint 或环境变量全集。

可在正式录制前单独检查：

```bash
cd /absolute/path/to/numpilot
source /absolute/path/to/numpilot-runtime/activate-numpilot.sh
python -m numpilot doctor --backend npu
```

如果现场网络必须经过代理，请在当前 shell 中按该环境的实际要求设置；不要把地址提交到仓库。
模型配置与网络是否真正可用，由下面的 `check llm` 现场验证。

## 录屏指令

启动持久终端：

```bash
numpilot
```

启动后应先看到 NumPilot Logo、版本和简短帮助。然后在 `numpilot ›` 提示符中依次输入：

```text
check llm
migrate
/absolute/path/to/numpilot/examples/add_one_migration_demo.py
y
execute
exit
```

其中源码路径必须替换为本机真实绝对路径。不要把 `y` 或 `execute` 提前管道输入；分别等终端展示
模型解读/流程确认，以及待执行源码路径与哈希后再手动输入，画面更真实，也保留了两个独立授权门。

## 画面与阶段

### 1. `check llm`

该命令向 `llm.json` 选择的模型发送一次很小、有界的真实请求。成功只表示本次检查收到了可用
回复，不表示模型已经能正确完成迁移。画面只显示脱敏的配置/model 标识和 pass/fail，不展开请求
正文、回复正文、Token、endpoint 或代理地址。

失败时先修正私有配置或现场网络，再重新输入 `check llm`；交互会话不应因此退出。

### 2. 源码快照与只读分析

输入 `migrate` 后，框架询问 Python 文件的绝对路径，读取稳定 UTF-8 快照并显示：

- 绝对路径、SHA-256、字节数、行数；
- 检测到的 NumPy API 与调用行；
- 当前 AsNumpy 的五态/能力结论；
- 缺少的精确调用合同与需要的接口语义。

这一阶段不导入、不执行源码，也不调用模型来覆盖静态分析结果。

### 3. 模型解读与 `y` 门

框架通过私有 `llm.json` 发起有界源码解读请求，显示模型等待状态和标记为
`interpreted_unverified` 的解读。它帮助开发者理解程序，但不是迁移代码，也不证明执行、精度或
性能。

随后终端询问是否继续。存在 CANNBot 缺口时，问题还应明确：本次 `y` 授权后续生成与 NPU
资格化，并接受本次固定 CANNBot/CANN 许可证；它不授权执行用户源码。框架展示待执行的原始与
规范化源码路径及哈希后，会在紧邻执行点要求单独输入精确的 `execute`。任一门输入其他字符或
EOF 都写入 declined 回执，并让相应迁移、验证和 benchmark 保持 `not_run`。

### 4. CANNBot-Skill 生成 Ascend C 草稿

对本例的 `add_one` 缺口，框架按以下顺序在终端展示完整过程：

1. 查询已安装 CANN 和固定 `ops-nn` 证据；只有可审计的 clean miss 才进入自定义生成。
2. 校验固定 CANNBot checkout、commit、许可证和必需文件哈希。
3. 完整打印本次真实模型请求实际嵌入的公开 CANNBot Skill 文件。
4. 模型处理期间周期显示累计等待时间，证明流程仍在运行；超时仍按失败处理。
5. 保存一个有界的 Ascend C `Adds` 片段及模型/请求/响应/产物哈希回执。

该文件的状态必须显示为 `generated_unverified`。它只是模型生成的 Ascend C 设计草稿，**未编译、
未安装、未激活，也没有被迁移程序执行**。完整展示 CANNBot 流程不能把这些事实升级。

### 5. 独立的 run-scoped 组合接口资格化

框架随后处理一条不同的证据链：用当前已合格的 `asnumpy.ones_like` 和 `asnumpy.add` 物化仅本次
运行可见的 `asnumpy.add_one(x)` 组合接口，并在隔离真 NPU 进程中验证多个 `float32` 输入、数值
结果、运行时身份、构成 API 调用次数和禁止 fallback 条件。

只有该资格化回执通过，终端才能说“本次运行新增 `asnumpy.add_one` 接口”。必须同时说明：

- `interface_provenance=composite_npu`；
- `scope=run`，上游 AsNumpy 和共享 OPP 未修改；
- `cannbot_draft_activated=false`，上一步的自定义 Ascend C 草稿未参与执行。

资格化失败时流程在这里停止，不能继续全程序迁移。

### 6. 全程序迁移、精度和性能

接口资格化后，框架先证明精确 scalar-one lowering，再迁移 normalized ordinary program，并固定
执行：

```text
01-analyze → 02-migrate → 03-verify → 04-benchmark
```

- `02-migrate` 的 AsNumpy 候选必须来自本次配置模型并通过模型/响应/源码/产物哈希和 AST 门；
- `03-verify` 在两个隔离进程中执行哈希绑定的原始语义与迁移程序，对指定输出做 shape、dtype、
  NaN/Inf 和容差比较；
- `04-benchmark` 只有 strict-NPU runtime、同步、调用计数、禁止 fallback、数值与样本门全部通过时
  才发布 `performance_valid=true` 和 speedup。

resident benchmark 测的是 marker 后的稳定设备驻留计算，不是冷启动端到端，也不包含所有传输或
模型等待时间。若任何性能门失败，终端必须显示 unavailable/not_run，而不是编造加速数字。

### 7. 最终报告

结束画面从实际回执汇总：

- 原始源码和 normalized/migrated 源码路径及哈希；
- CANNBot Skill、Ascend C 草稿、组合接口物化/资格化、lowering 的回执；
- 迁移、数值验证和性能状态；
- 输出 shape、dtype、元素数、字节数；
- 仅在有效时显示的本次规模 speedup；
- 所有生成文件的路径；
- 最终 `migration-report.json` 与 `migration-report.md` 的绝对路径。

报告如果只有一个有效规模，只能说“在该规模和该计时边界下观察到该加速比”，不能给出“超过
某规模必然更快”的阈值。

## 通用程序的失败关闭规则

同一个 `migrate` 交互适用于其他普通 NumPy 文件，但 `add_one` 的窄适配器不会套用到别的 API：

- 无缺口：正常进入 ordinary migration/verify/benchmark；
- 已登记且完整资格化的缺口：按该适配器合同继续；
- 通用 CANNBot 草稿只有 `generated_unverified`：停止，`interface_available=false`；
- 官方候选、构建、资格化、激活或重扫任一证据缺失：停止；
- 模型、NPU、数值或性能门失败：保留失败回执，不升级成成功。

## 自动化复验入口

人工录屏应使用持久终端。CI 或本地重复测试可以显式传参调用同一框架：

```bash
python -m numpilot --llm-config /absolute/private/llm.json \
  interactive-migrate \
  --source /absolute/path/to/source.py \
  --yes \
  --execute-source \
  --cannbot-gap-draft \
  --accept-cannbot-license \
  --show-cannbot-skill full
```

自动化的 `--yes`、`--execute-source` 和 `--accept-cannbot-license` 分别授权流程、执行已展示且
绑定哈希的用户源码、以及本次 CANNBot/CANN OSL 使用；`--non-interactive` 本身不是授权。

## AI/CodeArts-agent 入口

如果不希望使用内置配置模型，可以直接把下面这句话发给 CodeArts 侧边栏或其他交互 AI：

> 阅读 NumPilot 仓库与 `$numpilot-migration-gate`，使用 agent 模式，不用 `llm.json` 里的内置
> 模型进行迁移。由你与我交互完成源码解读、缺口补全、算子生成、接入和迁移候选，再执行框架
> 支持的确定性门禁并报告所有产物与边界。

这句话是硬选项门：AI 在读取、`stat` 或探测 `llm.json` 之前就必须切换到 agent 模式，不运行
`check llm`、`configure-llm` 或 configured-model generation。交互模型自己写出的产物标为
`agent_authored_draft`；当前若没有已登记适配器接受它，应停在 `migrate=draft/stopped`、
`verify=not_run`、`benchmark=not_run`，不能手工伪造 configured-model 回执。
