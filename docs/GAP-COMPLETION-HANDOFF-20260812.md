# NumPilot 缺口算子补全留档（2026-08-12 UTC）

## 结论

本轮完成了 PDF 设想中的第一条安全纵切：

```text
直接 numpy.<api> 缺口
→ qualified callsite 分析
→ GapSpec
→ cannbot/numpyexample 生成契约
→ 真实 LLM Ascend C 隔离草稿
→ 全文件哈希 manifest
```

当前终态是 `generated_unverified`，不是 `qualified` 或 `activated`。框架不会自动执行模型生成的 `build.sh`，不会安装系统 OPP，不会修改 AsNumpy，也不会把草稿加入静态覆盖集。

## 已落地代码

- `numpilot/agents/analysis.py`
  - 只识别真实 NumPy import；
  - 保留 canonical qualified API；
  - 局部变量/参数遮蔽时不误判 NumPy；
  - callsite 只保存参数数量和关键字结构，不向模型发送字面值。
- `numpilot/gaps/models.py`
  - versioned `GapCallsite`、`GapSpec`、`OperatorDraft`；
  - 对 gap ID、SHA-256、目标环境和行列信息做 fail-closed 校验。
- `numpilot/gaps/spec.py`
  - 只允许真正缺失且直接的 `numpy.<api>` 进入生成；
  - 签名缺口、Python dunder、嵌套 namespace 保持 blocked；
  - route 未实际检索时明确记录 `not_evaluated/review_required`。
- `numpilot/gaps/provider.py`
  - 真实 LLM 调用，无 AST/模板回退；
  - 最多 3 次响应/结构修正；
  - 路径穿越、符号链接、额外文件、大小、深度和类型限制；
  - 草稿文件全集、哈希、artifact ID、门禁状态和模型证据复验。
- `numpilot/gaps/pipeline.py`
  - 私有工作目录、原子非覆盖写、部分阻断状态和 `gap-report.json`。
- `numpilot/__main__.py`
  - 新增 `complete-gaps`；
  - `--provider prepare` 不联网；
  - `--provider llm` 必须调用已配置模型，成功输出脱敏证据。
- `examples/gap_cbrt.py`
  - 直接缺口 API 的 smoke case。
- `tests/test_gap_completion.py`、`tests/test_analysis.py`
  - 覆盖规格确定性、真实 import、遮蔽、脱敏、无回退、模型调用、篡改、symlink、path traversal、原子写和部分阻断。

## 真实模型调用证据

旧运行曾通过配置的 `remote_llm` 路径生成一次 `numpy.cbrt` 草稿；其客户端 manifest
记录为：

- provider：`glm`
- model：`glm-5.2`
- mode：`remote_llm`
- attempt count：`3`
- fallback：`disabled`
- 最终有效响应 SHA-256：
  `3d0078e9c7db1dd26c99a062b05d8a81251c769c6813060f6e030d8e72a60bbf`
- artifact ID：`draft-5b0b2e87b24f44ea3513`
- 文件数：`13`

持久产物：

```text
/home/ma-user/work/numpilot-gap-runs/cbrt-20260812/
├── gap-report.json
└── gaps/cbrt-5cc42f1bfbab7d63/
    ├── gap-spec.json
    ├── GENERATION_REQUEST.md
    └── draft/
        ├── operator-artifact.json
        ├── op_host/
        ├── op_kernel/
        ├── op_api/
        ├── asnumpy_adapter/
        └── tests/
```

下机前已在不使用代理联网的情况下重新运行相同命令；它复用了现有草稿并重新验证 manifest、文件全集和全部哈希，退出码为 0。

这里需要区分两层证据：上述旧 manifest 的 `provider/model/mode` 和响应哈希由客户端
记录，旧格式没有逐次 HTTP transport receipt；离线复用也只证明既有产物哈希仍一致，
不能证明复用当次又调用了模型，更不能单靠客户端字段证明私有网关背后的官方服务身份。
最终代码已刻意拒绝这种缺少新回执/分层计数的 legacy 远程草稿重载；该目录只保留作
历史静态审计，不再作为符合当前 schema 的可复用草稿。

## 2026-08-12 全新联网复验

为排除“只是复用了旧草稿”，在一个事先不存在的新输出目录执行了完整
`complete-gaps --provider llm`。该运行确实进入远程调用路径，连续 3 次均收到 HTTP
错误并失败关闭，未创建 draft；因此不能把这次完整生成写成成功。

随后使用同一份私有配置发送一次最小、全新的 OpenAI-compatible
`/chat/completions` 请求，收到：

```text
UTC start:     2026-08-12T10:32:51.722350+00:00
UTC end:       2026-08-12T10:32:52.724602+00:00
HTTP status:   200
requested:     glm-5.2
response model:glm-5.2
response id:   msg_2026081218325129e72141d26f4959
prompt tokens: 16
completion:    2
```

脱敏回执保存在仓库外，权限为 `0600`：

```text
/home/ma-user/work/numpilot-gap-runs/glm-proof-20260812-v1/
live-glm-connectivity-receipt.json
```

回执保存请求体、响应体、内容和当时客户端代码的 SHA-256，不保存 token 或 base URL。
它能证明客户端在该时间向**配置的 endpoint** 发出全新请求并收到一个自报
`glm-5.2`、带响应 ID 和 usage 的 HTTP 200 响应。由于 endpoint 是私有 IP 网关，且没有
网关服务端日志、厂商签名或可验证 attestation，这仍不能独立证明网关背后一定是官方
GLM 服务；最终结论不得越过这条证据边界。

回执实现完成第一轮安全审查后，又通过 NumPilot 自身
`LlmEngine._chat_with_receipt` 发出一次带唯一 nonce 的全新请求，而非手写旁路请求：

```text
UTC start:       2026-08-12T10:55:29.584763Z
UTC end:         2026-08-12T10:55:32.463904Z
HTTP status:     200
requested model: glm-5.2
response model:  glm-5.2
response id:     msg_20260812185529b285f9ddbf2d45eb
nonce present:   true
TLS verify:      enabled
prompt tokens:   48
completion:      23
```

第一份框架回执同样在仓库外、权限 `0600`：

```text
/home/ma-user/work/numpilot-gap-runs/glm-proof-20260812-v1/
framework-live-glm-receipt.json
```

在独立审计继续封住旧草稿回执伪造、计数/哈希不一致、类型混淆和无界响应读取后，使用
**最终源码**再执行一次随机 challenge-response。模型被要求只回显不可预知的 nonce，
响应内容与 nonce 字节完全相等，因此内容 SHA-256 也与 nonce SHA-256 完全相等：

```text
UTC start:       2026-08-12T11:02:39.323976Z
UTC end:         2026-08-12T11:02:42.640084Z
HTTP status:     200
requested model: glm-5.2
response model:  glm-5.2
response id:     msg_202608121902397ad37b223bc641a1
exact nonce:     true
content/nonce SHA-256:
  433248d45dab2d952b785986f16e6723bb8b6fad2090f1df1e6102d7a2ecbb0f
TLS verify:      enabled
```

最终回执同时绑定 `llm.py`、`provider.py`、`__main__.py`、`llm_config.py` 的源码
SHA-256，文件权限 `0600`，文件自身 SHA-256 为
`20b8a21cd2738c3aa052319bdc4294edc8aff2f9e5fd62348d665a9c28c206db`：

```text
/home/ma-user/work/numpilot-gap-runs/glm-proof-20260812-v1/
framework-live-glm-final-receipt.json
```

这些框架回执含请求体/响应体/内容 SHA-256、UTC、响应 ID/created、usage 和 TLS 状态；
严格字段白名单不保存 Token、Base URL 或 endpoint 指纹。最终挑战加强了“最终 NumPilot
代码真的走了全新远程 HTTP 调用”的证据，但私有网关后的上游身份限制仍完全相同。

最后又在确认不存在的目录执行最终 `complete-gaps` 全链路，只允许 1 次远程尝试。该大
请求在配置 endpoint 返回 `HTTP 504`，NumPilot 只记录安全状态码并失败关闭；未创建
draft，也未创建会误导为成功的 `gap-report.json`：

```text
/home/ma-user/work/numpilot-gap-runs/glm-full-final-20260812/
├── full-generation-attempt-receipt.json  # 0600
└── gaps/cbrt-5cc42f1bfbab7d63/
    ├── gap-spec.json
    └── GENERATION_REQUEST.md
```

失败尝试回执的 SHA-256 为
`b8ee38eb698efb06a1c13a95ebba8fc153635ffed69a977f4cc1f69bd6344936`。
因此本轮准确结论是：最终代码的最小 freshness challenge 成功得到自报 `glm-5.2` 的
HTTP 200 且严格回显 nonce；完整算子生成路径也真实联网，但大请求以 HTTP 504 失败，
没有产出一个新的、带当前 transport receipt 的算子草稿。

草稿静态审查发现多个 `TODO_VERIFY_CANN_API`。生成的 README 和 build 脚本也明确说明当前预期不能编译，因此没有把 CPU adapter 测试或源码存在本身当成 NPU 成功证据。

## 环境与持久化

`/home/ma-user/work/.codex-env.sh` 已包含：

```bash
export NUMPILOT_CONFIG_HOME="/home/ma-user/work/.numpilot-config"
export http_proxy="http://127.0.0.1:7897"
export https_proxy="http://127.0.0.1:7897"
export HTTP_PROXY="${http_proxy}"
export HTTPS_PROXY="${https_proxy}"
```

重新登录后的第一条命令：

```bash
source /home/ma-user/work/.codex-env.sh
```

所有本轮源码、配置和模型产物均位于 `/home/ma-user/work`，符合本服务器“关机后只保留 work”的约束。LLM Token 只在私有配置中，未写入仓库、报告或本留档。

## 验证结果

历史下机前执行：

```text
python3 -m pytest -q
144 passed in 10.38s

python3 -m compileall -q numpilot
exit 0
```

本次最终源码执行：

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider
253 passed in 6.70s

python3 -m compileall -q numpilot
bash -n run_demo.sh
python3 -m numpilot --help
git diff --check
全部通过
```

真实 NPU 环境自检：

```text
Python            PASS
numpy             PASS
CANN 9.0.0        PASS
ASCEND_OPP_PATH   PASS
NPU device        FAIL（npu-smi 未返回设备信息）
asnumpy           FAIL（ModuleNotFoundError）
```

因此本机当前不能诚实完成 build → NPU runtime → 数值校验 → 激活闭环。

## 下次续接

```bash
source /home/ma-user/work/.codex-env.sh
cd /home/ma-user/work/numpilot
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider

# 复核最终 freshness 回执；不联网
python3 -m json.tool \
  /home/ma-user/work/numpilot-gap-runs/glm-proof-20260812-v1/\
framework-live-glm-final-receipt.json >/dev/null
```

注意：`cbrt-20260812` 的旧草稿缺少当前逐次 transport receipt，最终代码会按预期拒绝
重载，不能再把它称为“当前 schema 下可离线复用”。若要重新生成，必须选一个全新输出
目录并明确接受会联网；当前最终尝试的已知结果是 HTTP 504。

下一阶段按以下顺序继续：

1. 用本机 CANN 9.0.0 头文件和 cannbot registry-invoke skill 逐项消除 `TODO_VERIFY_CANN_API`。
2. 在 `/home/ma-user/work` 下建立隔离 OPP/AsNumpy 构建环境，不写系统安装。
3. 使用已实现的离线 qualification 准入契约，增加独立可信的 build、UT、NumPy oracle
   数值、AsNumpy public API、NPU runtime evidence。
4. 将通过复验的原子 Registry overlay 接到 profile-aware capability 查询；name-only 查询
   在无法判断 dtype/shape constraints 时继续 fail closed。
5. 只有全部门禁通过后才激活为 `CUSTOM_NPU`；环境或哈希漂移立即 stale/fail closed。

## 工作树说明

当前没有创建 Git commit。工作树原本已有 LLM 配置、wizard、README、run_demo 等未提交改动，本轮全部保留，没有 reset 或覆盖。续接时先运行 `git status --short`，不要用破坏性 reset。
