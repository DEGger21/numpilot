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

## 配置 schema 2

`python -m numpilot configure-llm --init` 创建私有 `llm.json`：

```json
{
  "schema_version": 2,
  "base_url": "https://gateway.example/v1",
  "auth_token": "由用户在私有文件中填写",
  "model": "org/model",
  "provider": "gateway-audit-label",
  "protocol": "openai-chat-completions",
  "response_model_aliases": []
}
```

目录权限为 `0700`、文件权限为 `0600`。环境变量逐字段优先于文件；若环境显式覆盖 `model`，
不会继承旧模型的 alias allowlist。`--check` 使用与生成相同的本地 readiness/profile 校验且
不联网；`--probe` 才执行真实请求。probe 与 gap/repair/adapter 的远端 readiness 都要求响应
包含固定 `NUMPILOT_LLM_READY` marker，不能出现 probe 成功而真正生成门禁拒绝的分叉。

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
