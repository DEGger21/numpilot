# AsNumpy 算子接入生成经验

这份记录来自 NumPilot Phase 18.20B-8 的真实生成实测，作为
`asnumpy-gap-completion` 的复用经验。它描述的是已观察到的工程事实和
门禁边界，不把模型输出当成正确算子。

## 已验证事实

- `llm.json` 的 OpenAI-compatible DeepSeek endpoint 可以连通；readiness 和
  generation 请求均取得 HTTP 200，模型身份可由远端回执校验。
- HTTP 200、模型身份正确、token 使用量存在，都不能证明生成了可用的
  AsNumpy overlay。只有本地协议解析通过，才允许进入
  `generated_unverified`。
- 旧的 `NUMPILOT_EXACT_EDIT_V1` 要求模型同时复制 `OLD` 原文、组织多层
  `BEGIN_EDIT/END_*` 标记并保持空行精确。DeepSeek 三轮实测共 17 次请求，
  在 `EDIT_COUNT` 或 `NEW block` 阶段失败，没有发布 overlay。
- 失败时必须封存 request/response hash、远端 receipt、失败原因、上下文
  计划 hash 和所有 false gates；不能把 readiness、失败响应或候选文件当成
  成功。

## 当前协议

官方 existing-module 路由使用 `NUMPILOT_WINDOW_REPLACE_V1`。模型只输出
当前目标文件的 reviewed window：

```text
NUMPILOT_WINDOW_REPLACE_V1
PATH: exact/target/path
BASE_SHA256: full-target-sha256
WINDOW_COUNT: N
BEGIN_WINDOW 1
LABEL: reviewed-window-label
CONTENT:
complete replacement for this reviewed window
END_WINDOW 1
END_WINDOW_REPLACE
```

NumPilot 本地重新计算窗口范围和 hash，要求 label、顺序、数量、整文件
`BASE_SHA256` 全部匹配，并且重建后的差异只能落在审计过的窗口中。旧
exact-edit parser 保留用于历史回执重放和修复兼容；新 generation lane 默认
使用窗口协议。

## 交互和失败处理

1. 终端先显示 official resolution、context gate、AsNumpy context plan 和
   `plan_sha256`，再显示模型身份和本次请求边界。
2. 每个目标文件显示 `file i/n`、尝试次数、解析结果和 receipt 数量；模型
   正在生成时用心跳保持终端可见。
3. 解析失败只进入下一次有界重试；成功前不写正式组件、不 build、不 import、
   不执行 NPU、不推进 `ADAPTER_CANDIDATE`。
4. 所有尝试完成后输出隔离的
   `asnumpy-adapter-generation-failure.json`，并将阶段标记为 `blocked`。
5. 需要换用 Codex 等更强代理时，代理应读取同一份 context plan 和窗口
   协议，生成本地候选后再交给 NumPilot 做同样的机械校验；代理不能绕过
   RecoveryStore、GapSpec、manifest、ACLNN 两段式和后续数值/性能门禁。

## 继续开发的验收条件

- 协议解析正负例通过，窗口外修改、label 漂移、hash 漂移、占位符和 no-op
  均 fail-closed。
- 真实模型只有在 commentary 预告后才可再次请求；请求结束必须有可复核的
  failure/success receipt。
- 结构化候选进入 `generated_unverified` 后，仍必须依次通过 build、import、
  NumPy oracle、真实 NPU execution、performance 和 qualification，才能计入
  benchmark 迁移结果。
