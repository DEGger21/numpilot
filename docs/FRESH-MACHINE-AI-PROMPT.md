# NumPilot 全新机器源码实验 Prompt

你正在一台全新机器上工作。该机器与开发机没有代码、虚拟环境、缓存、wheel、receipt、
CANNBot checkout 或 LLM 配置重叠。除系统已安装的 Ascend 驱动/CANN、NPU、必要的
Python 和编译环境外，不假设存在任何 NumPilot 相关内容。

本轮目标是只手工克隆 NumPilot；AsNumpy 从 NumPilot 固定的官方源码现场编译。CANNBot
仅在需要 gap/operator lane 时从固定官方提交拉取并校验，direct migration 不依赖它。不要索取、下载或使用 wheelhouse，也不要从旧机器复制
AsNumpy wheel、缓存、`llm.json`、Token、代理配置或历史执行产物。

## 强制约束

1. 只使用发布说明给出的、包含本 Prompt 所述集成合同的不可变 release tag，不得改用浮动
   master。2026-08-27 尚未产生该 release；`v0.2.0a2` 只是基线且不含这些改动。若发布方
   还没有给出新 tag，必须停止新机验收，不能回退到 `v0.2.0a2` 冒充当前版本。
2. 完整阅读仓库 `README.md`、`docs/FRESH-NOTEBOOK-QUICKSTART.md` 和
   `docs/AI-HANDOFF.md` 中与普通纯 NumPy script、bootstrap、configured-model 生成约束相关的内容。
   若历史 handoff 与全新源码 Quickstart 冲突，以本 Prompt 和源码 Quickstart 为准。
3. 输入对象只允许纯 Python + `import numpy as np` 的 ordinary script；不要求
   `build_inputs()` 或 `forward()`。
4. 所有迁移、缺口算子和 AsNumpy adapter 代码都必须由用户配置的 OpenAI-compatible 模型
   生成。本地 AST 只能扫描和审核，不能在模型失败时替代生成。GLM 是可选示例，不是硬锁。
5. 不处理 ops-nn，不自动继续 Case02–10。
6. NumPilot 锁定 AsNumpy source profile 的公开 `matmul` 已知会回到 host NumPy；
   对官方最新提交的独立审计也得到同样结论。用户本轮已选择暂时接受该
   fallback，默认只验证安装与兼容性迁移；只有用户另行授权后才做独立数值实跑。结果必须标记
   `strict_npu=false`、`performance_valid=false`，不得发布或暗示 NPU 加速比，也不得
   复用历史 P0 wheel 的 180× 结果。
7. 任何模型请求若需要代理，只能使用这台新机器真实可用、由用户提供的代理；不得假定
   开发机的 `127.0.0.1:7897` 在新机器也存在。Token、代理值和响应正文不得写入仓库。
8. 在用户明确说“开始迁移”前，不调用模型、不生成迁移文件。

## 阶段 A：只读检查

先执行并报告，不做安装：

```bash
python3 --version
git --version
g++ --version
npu-smi info
```

确认 Python ≥3.10、NPU 健康、CANN 环境可定位。若系统缺 Python 开发头、Git、
GCC/G++ ≥11.2、fmt/spdlog CMake 开发包，明确列出缺项；不得自动 sudo。

## 阶段 B：克隆并确认版本

```bash
NUMPILOT_RELEASE_TAG="${NUMPILOT_RELEASE_TAG:?请填写 Day 5 发布说明中的不可变 tag}"
test "$NUMPILOT_RELEASE_TAG" != v0.2.0a2
git clone --branch "$NUMPILOT_RELEASE_TAG" --depth 1 \
  https://gitcode.com/li-xingHIT/numpilot.git
cd numpilot
git status --short --branch
git rev-parse HEAD
```

工作树必须干净，HEAD 必须与发布说明中该新 tag 的 commit 一致。

## 阶段 C：一键源码安装

direct lane 默认不安装 CANNBot：

```bash
PERSISTENT_ROOT=/absolute/persistent/root
./install.sh --allow-network \
  --persistent-root "$PERSISTENT_ROOT"
source "$PERSISTENT_ROOT/activate-numpilot.sh"
```

只有需要 gap/operator lane 时，才先让用户审阅并接受固定 CANNBot commit
`1ef6372a538f39b0b7ed34666a3e1f465fb4936a` 的 CANN Open Software License v2.0，
并在安装命令中增加 `--with-cannbot --accept-cannbot-license`。标准持久 HOME 可省略
`--persistent-root`，并 source 安装器打印的 activation 路径；临时 HOME/Notebook 必须显式
指定平台会保留的根。activation 不含 Token、endpoint、代理或 pip 索引，source 两次也不会
重复 PATH/CANN 路径。

如 CANN 环境未激活，使用新机器实际存在的 `set_env.sh` 传给 `--cann-env`。不要修改
`.bashrc`。安装失败时保留第一处错误和安装器打印的 receipt 路径，不要绕过门禁。

若新机器默认 Python 私有镜像缺少依赖，不得让安装器自动换源。先向用户报告，经确认后可
显式附加 `--pip-index-url https://pypi.org/simple` 或用户指定的可信索引。不得把带凭据的私有
URL 写入仓库。默认 receipt 位于以下命令输出的 XDG 状态路径，不在仓库内：

```bash
.venv/bin/python -m numpilot paths --get bootstrap_receipt
```

## 阶段 D：离线复验

```bash
python -m numpilot --version
python -m numpilot paths --json
python -m numpilot bootstrap --verify
# 仅当安装时选择了 gap/operator lane：
python -m numpilot bootstrap --verify \
  --with-cannbot --accept-cannbot-license
python -m numpilot doctor --backend npu
```

检查 receipt 为 `ready=true`、AsNumpy 为 `built-from-pinned-source`、README smoke
通过，且 verify 报告
`network_requested_by_numpilot=false`。安装 smoke 不等于每个 API 都在 NPU 上执行。
若选择了 gap/operator lane，还必须看到 CANNBot Skill smoke 通过；direct-only 不要求该项。
如果安装时显式使用了 `--receipt /absolute/path`，后续新 shell 要设置同一绝对路径的
`NUMPILOT_BOOTSTRAP_RECEIPT`；使用生成的 activation 时会自动恢复该路径。

## 阶段 E：新建 configured-model 配置

克隆不会自动生成配置。执行：

```bash
python -m numpilot configure-llm --init
```

告诉用户命令打印的实际 `llm.json` 路径，等待用户在新机器上填写
`base_url`、`auth_token`、精确 `model`、审计标签 `provider`、
`protocol=openai-chat-completions` 并保持 0600 权限。响应 model 默认必须精确一致；只有网关
确实返回固定别名时才显式填写 `response_model_aliases`。例如 GLM 可填写
`model=glm-5.3`、`provider=glm`，但这只是示例。
用户填写完成后：

```bash
python -m numpilot configure-llm --check
python -m numpilot configure-llm --probe
```

不得显示 Token 或响应正文。

## 阶段 F：迁移前停点

用户上传 Case01 的纯 NumPy 文件后，只读取、静态扫描并报告：

- 文件 SHA-256；
- import 是否符合纯 NumPy ordinary script；
- NumPy API 清单；
- DIRECT / FALLBACK_CONSENT / GAP_REQUIRED / CAPABILITY_AUDIT_REQUIRED / BLOCKED 分类；
- 已知 `matmul` fallback 对本轮结论的影响。

同时确认 capability `audit_id` 含 `pinned-source-receipt-v1`，原生 API 不是
`CAPABILITY_AUDIT_REQUIRED`。若 receipt/安装文件/包版本任一漂移，必须停止，不得把
`UNKNOWN` 改写为 direct。

然后停止并告诉用户“环境与源码已经读完，等待你说开始迁移”。不得提前调用模型。

用户明确说“开始迁移”后，才执行一次 Case01：

```bash
python -m numpilot migrate /absolute/path/to/01_numpy.py \
  --backend npu --fallback-policy accept \
  --out /tmp/numpilot-case01
```

保存 configured-model receipt、迁移产物和 AST 审计。`migrate` 不执行用户源码；不得把迁移成功
写成已实跑。如果用户之后明确要求实际运行，先记录 source/artifact SHA-256 并审核 AST，
再在两个独立子进程执行原 NumPy 文件和 configured-model 迁移文件，保存非敏感输出与数值对照。
不得把该兼容性 fallback 结果写成 strict NPU 或性能有效，也不得继续下一个案例。
