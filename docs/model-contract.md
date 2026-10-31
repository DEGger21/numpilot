# 通用模型与回执合同

NumPilot 的生成路径使用用户自己配置的 OpenAI-compatible endpoint。当前支持范围是一个有意
收窄的合同，不等同于支持所有 OpenAI API：

- `protocol` 固定为 `openai-chat-completions`；
- Bearer auth；
- `POST <base_url>/chat/completions`；
- 文本响应位于 `choices[0].message.content`；
- 请求使用受控的 temperature、`max_tokens=1..1000000` 和 `timeout=10..3600s` profile。

`provider` 只是用户填写的审计标签，不决定某个厂商是否获准。`model` 保存并发送精确原值，
可安全包含 `/`，例如 `org/model`；两端空白不会被 strip 后接受，而是直接拒绝。展示用脱敏文本
从不参与身份比较。

远端调用的 payload 不限于 ordinary migration：`migrate` 发送完整输入源码；gap/operator 生成与
修复发送有界缺口合同、静态语义函数、选定 CANNBot Skill 内容以及适用的草稿/诊断；AsNumpy adapter
生成与修复发送本地目标投影选中的源码/头文件、仓库 integration Skill、overlay 与诊断。CANNBot
提供的是固定工作流上下文，实际草稿由当前入口明确选择并授权的边界模型生成。所有真实生成/修复请求都可能
产生费用；使用非本地 endpoint 前必须确认这些材料允许披露。

## 入口与 no-llm 边界

本合同只约束 **terminal/manual** 或 **hybrid 中明确选择的 configured-model 子运行**。终端
手动使用时，`llm.json` 可以不配置，用户也可直接提供候选并只运行 NumPilot 的确定性门禁。
**CodeArts-agent** 入口先读取仓库与 Skill，由侧边栏交互模型独立完成迁移、缺口补全或算子草稿；
若用户要求使用该模型或明确“不用 `llm.json`/内置模型”，不得读取配置、运行 `configure-llm`、
传入 `--provider llm`、探测 endpoint 或静默回退。候选应标记 `agent_authored_draft`，没有适配器
绑定完整 agent/provider/model/response/source/artifact 证据前，不适用本合同的 configured receipt
或 strict-NPU/性能结论。Hybrid 切换到 configured 子运行必须重新声明、授权并使用新 run 目录。

## 配置 schema 2

仓库根目录提供空白模板 `llm.example.json`。复制为私有、被 `.gitignore` 忽略的 `llm.json`：

```bash
install -m 600 llm.example.json llm.json
# 编辑 llm.json：base_url、auth_token、model
python -m numpilot configure-llm --check
```

模板内容：

```json
{
  "schema_version": 2,
  "base_url": "",
  "auth_token": "",
  "model": "",
  "provider": "openai-compatible",
  "protocol": "openai-chat-completions",
  "response_model_aliases": []
}
```

源码 checkout 中，NumPilot 默认自动发现仓库根的 `llm.json`，与当前工作目录无关。若显式设置了
`NUMPILOT_CONFIG_HOME`、`NUMPILOT_HOME` 或 `NUMPILOT_RUNTIME_ROOT`，该配置目录优先；自动化也可设
`NUMPILOT_DISABLE_REPO_CONFIG=1` 跳过仓库本地文件。`XDG_CONFIG_HOME` 只影响默认用户目录，不会隐藏
仓库本地文件。仍可用
`python -m numpilot configure-llm --init` 在用户配置目录生成同一模板，或用 `--llm-config` 指定文件。

对于仓库外的持久运行根，可设置
`NUMPILOT_RUNTIME_ROOT=/absolute/persistent/numpilot-runtime`；单次命令也可用
`python -m numpilot --llm-config /absolute/persistent/numpilot-runtime/.numpilot-config/llm.json configure-llm --check`
显式选择私有文件（`--llm-config` 放在子命令后也可）。该选项只传递路径，不会输出 Token。

普通用户配置目录为 `0700`、文件为 `0600`。源码 checkout 的仓库根通常为 `0750/0755`；此时
`llm.json` 本身仍必须是 `0600`，且仓库目录不得对其他用户可写。`llm.example.json` 必须保持空白；
Token、endpoint、代理地址和原始响应正文不得提交。环境变量逐字段优先于文件；若环境显式覆盖 `model`，
不会继承旧模型的 alias allowlist。`--check` 使用与生成相同的本地 readiness/profile 校验且
不联网；`--probe` 才执行真实请求。probe 与 gap/repair/adapter 的远端 readiness 都要求响应
包含固定 `NUMPILOT_LLM_READY` marker，不能出现 probe 成功而真正生成门禁拒绝的分叉。

## 强制代理门禁

NumPilot 读取标准 `HTTP_PROXY` / `HTTPS_PROXY` 和对应小写变量，不把代理地址写入
`llm.json`。需要确保模型流量只能经代理时，显式设置：

```bash
proxy=http://127.0.0.1:<port>
export HTTP_PROXY="$proxy" HTTPS_PROXY="$proxy"
export http_proxy="$proxy" https_proxy="$proxy"
export NUMPILOT_LLM_REQUIRE_PROXY=1
python -m numpilot configure-llm --probe
```

门禁按 endpoint 协议要求匹配的 `HTTP_PROXY` 或 `HTTPS_PROXY`，并检查 `NO_PROXY` / `no_proxy`
没有绕过目标。缺少匹配代理或命中 bypass 时，请求会在任何网络传输前 fail closed；启用门禁后
每次请求都从当前环境快照构造独立 proxy opener，避免复用缓存了旧代理环境的全局 opener。
`--check` 不联网：它验证本地模型配置，并在启用强制代理策略时校验当前代理环境和 bypass 门禁。
`--probe` 才验证实际代理传输与远端合同。代理 URL、Token、endpoint 和响应正文都不得进入回执、
日志或 Git。

## 身份规则

1. configured model 必须与请求 body 的 model 精确一致。
2. 响应 model 默认必须与 configured model 精确一致。
3. 网关确实返回固定别名时，用户必须把完整别名显式加入 `response_model_aliases`。
4. 响应 model 缺失、未授权 alias、provider/protocol 快照漂移或内容哈希不一致都 fail closed。
5. 失败尝试可作为诊断证据保留，但只有最后一个通过 exact/explicit-alias 身份和本地内容门禁的
   回执能授权产物发布。

## Receipt schema 2

新回执绑定 protocol、provider、configured/requested/response model、alias policy、identity
decision、identity SHA-256、请求/响应 body SHA-256 和提取内容 SHA-256。Token、Base URL、标准
代理变量的值和响应正文不得进入回执、日志或 Git。

旧 schema 1 只作为已有 `provider=glm`、请求/响应均精确为同一 `glm-*` 身份的只读兼容路径；
新写入、通用 provider、alias 和 repair/adapter 证据一律使用新 schema，旧 schema 不会被静默
升级或用于放宽身份。

GLM 可作为实际配置，例如 `model=glm-5.3`、`provider=glm`，但它与 GPT、DeepSeek 或
`org/model` 一样服从上述合同，不再是运行时硬编码的授权条件。
