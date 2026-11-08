# NumPilot 外部模型与算子生成架构

本文记录 NumPilot 当前如何使用 `llm.json` 中配置的模型，以及从
`SLAI-AscendBridge2` 借鉴的调用边界。文档约束的是工程接口和验收责任，
不把某一次模型输出当成稳定实现。

## 结论

NumPilot 不应把 DeepSeek 或其他大模型嵌入运行时。推荐的主线是：

```text
llm.json
  ↓（本地配置加载，token 只在内存中）
ModelGateway / LlmEngine
  ↓（OpenAI Chat Completions 兼容接口）
外部模型服务（当前配置标签：openai-compatible，模型：deepseek-flash）
  ↓（结构化响应 + 脱敏回执）
GapCompletionPipeline / LlmDraftProvider
  ↓
隔离 candidate workspace
  ↓
本地语法、契约、编译、AsNumpy/NPU、数值、性能门禁
  ↓
OperatorArtifactStore（候选、资格化、激活和运行回执）
```

模型是候选生成器，不能直接修改 AsNumpy、全局运行时或 benchmark 结果。
AsNumpy 源码、CANNBot 静态 bundle 和最终激活内容仍由项目作者控制。

## 当前 NumPilot 的调用链

1. `llm.json` 由 `numpilot.llm_config.load_llm_config()` 读取。环境变量
   `NUMPILOT_LLM_*` 可以覆盖保存配置；配置文件和 token 不进入仓库。
2. `ModelGateway` 是统一的类型化边界，给请求附加任务和角色标签，并计算
   请求哈希。生产 transport 使用 `LlmEngine._chat_with_receipt()`。
   当前 `LlmDraftProvider` 的 production gap-generation 路径已经固定使用
   `ModelGateway(task="gap_generation", role="operator-generator")`；注入式
   test-double 仍只用于离线测试。
3. `LlmEngine` 调用 `<base_url>/chat/completions`，请求协议是
   `openai-chat-completions`，带 `Bearer` token、精确模型名和温度 0。当前
   `llm.json` 的模型名是 `deepseek-flash`；这只是配置事实，不代表本轮已经
   访问 DeepSeek。
4. 响应必须通过模型身份、HTTP 状态、TLS、内容哈希、响应实体哈希和 usage
   回执校验。终端和持久化工件不输出 endpoint、token 或原始响应。
5. 普通迁移使用 `LlmEngine`。算子缺口使用 `GapCompletionPipeline`：先解析
   `GapSpec` 和官方/CANNBot 上下文，再让 `LlmDraftProvider` 生成候选。直接
   Ascend C 缺口采用分阶段文件生成，每个阶段有独立回执和重试；默认最多
   3 次尝试。
6. `LlmDraftProvider` 明确不编译、不执行、不激活模型返回的代码。候选写入
   隔离 workspace 后，才交给本地 validator、编译器、AsNumpy qualification
   runtime、数值检查和性能检查。通过后由 `OperatorArtifactStore` 记录
   candidate → qualification → activation 的不可变引用链。

## 能否由 Codex 扮演模型角色

可以分两种情况：

- **离线开发角色**：我可以按相同的 GapSpec、CANNBot 上下文和输出契约，直接
  生成候选、审查提示词、构造失败回执和测试 double。这适合写代码、补测试和
  回放流程，但生成内容不能冒充 DeepSeek 响应，也不能生成虚假的远程回执。
- **真实模型角色**：只有显式执行 NumPilot 的 `llm.json` transport 才算一次
  DeepSeek 调用。调用前应向作者说明入口、provider/model、上下文组成、最大
  输出预算、超时、预期工件和费用影响；调用结果必须经过现有回执和本地门禁。

因此，模型能力可以由外部服务替换，NumPilot 的验收、存储和运行时不依赖
某一个模型。当前开发回合没有调用 `llm.json`。

## 调用次数与时间口径

现有实现已经在每个生成报告中记录 `attempt_count`、`remote_receipts`，回执中的
`utc_start`/`utc_end` 可用于计算请求耗时；模型响应内容只保存哈希。一次普通
生成成功通常是 1 次请求，失败重试最多 3 次。直接 Ascend C 生成拆成固定文件
阶段，阶段数由算子合同决定，最坏情况下是“阶段数 × 3”，修复流程还会产生
额外的有界请求。

当前仓库没有真实 DeepSeek 生成样本，因此不能诚实地给出平均补全时延、成功率
或平均费用。150 例无模型 replay 的耗时也不能作为模型耗时。获得真实平均值时，
应从远程回执计算中位数、P95、每个 gap 的请求次数、通过率和修复次数，再写入
独立的模型评估报告。

## SLAI-AscendBridge2 的做法

`SLAI-AscendBridge2` 使用 Claude Code 作为外部编排器，不在项目代码中内嵌模型：

- Claude Code 通过 `ANTHROPIC_BASE_URL`、`ANTHROPIC_AUTH_TOKEN` 和
  `ANTHROPIC_MODEL` 连接 Anthropic 兼容服务；其文档给出了 DeepSeek 的兼容
  endpoint `https://api.deepseek.com/anthropic` 及 `deepseek-chat`/
  `deepseek-reasoner` 配置示例。
- `.claude/agents` 和 Agent Teams 负责角色分工、上下文组织、任务派发和回报；
  这属于开发期编排，不是 NumPilot 运行时依赖。
- CANNBot 是按缺口触发的外部资产。流程先检查 torch/torch_npu 原生接口，
  再查 CANN/Ascend 社区实现，最后才进入 CANNBot 的 Architect、Design
  Reviewer、Developer、Reviewer 四角色流程。
- 每个适配任务在 adaptation-local 目录产出源码、构建物、测试和审查记录，
  不把 CANNBot 全局注册为 NumPilot 的默认插件。

NumPilot 应借鉴这种“外部模型/工具 + 本地编排与证据”的方式，但保留自己的
`ModelGateway`、`OperatorArtifactStore` 和 AsNumpy qualification gate。两者的
provider 协议可以不同：SLAI 示例使用 Anthropic 兼容协议，NumPilot 当前实现
使用 OpenAI Chat Completions 兼容协议；关键是都不把模型嵌入最终运行时。

## 交付约束

- 没有模型回执时，候选只能是 `generated_unverified`，不得计入 qualified。
- 没有匹配的 AsNumpy native runtime、bootstrap receipt 和数值证据时，不能声明
  `strict_npu_qualified`；shim 结果不能冒充 NPU 结果。
- 性能结论必须绑定单逻辑 CPU 对单卡 NPU、固定 dtype/shape/contiguity、预热与
  冷启动口径，并单独保存 performance receipt。`--performance-evidence-root` 使用
  显式 manifest 将 receipt 绑定到 sample；无条目保持 `performance_qualified=null`，
  shim 或 hash 不匹配不能升级资格。
- 任何真实模型调用都必须在开发日志中记录用途、模型身份、请求次数、回执哈希、
  通过的本地门禁和失败原因；不得记录 token、endpoint 或原始响应。
