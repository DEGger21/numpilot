<div align="center">

<h1><img src="docs/assets/brand/numpilot-wordmark.png" alt="NumPilot" width="560"></h1>

### 面向华为 Ascend NPU 的可验证 NumPy 迁移工具

<p>
  <a href="README.md">English</a> ·
  <a href="docs/installation.zh-CN.md">安装指南</a> ·
  <a href="docs/architecture.md">系统架构</a> ·
  <a href="CONTRIBUTING.md">参与贡献</a> ·
  <a href="LICENSE">许可证</a>
</p>

<p>
  <img alt="版本" src="https://img.shields.io/badge/version-0.3.0a1.dev0-6D5CE7?style=flat-square">
  <img alt="状态" src="https://img.shields.io/badge/status-alpha-F59E0B?style=flat-square">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&amp;logo=python&amp;logoColor=white">
  <img alt="NPU 平台" src="https://img.shields.io/badge/NPU-Linux%20AArch64-008AA3?style=flat-square">
  <img alt="许可证" src="https://img.shields.io/badge/license-Apache--2.0-2EA44F?style=flat-square">
</p>

</div>

NumPilot 分析受支持、可静态理解的单文件 NumPy 程序，生成基于 AsNumpy 的迁移候选，并通过确定性的
能力、来源、结构和数值门禁决定候选能否使用。不支持或语义不明确的行为会明确停止，不会被静默改写
或回退到 CPU。

> 模型负责提出候选；NumPilot 的确定性门禁决定候选能否采用。

## 安装

### 推荐：华为云 Ascend 基础镜像

> **CANN 9.0.0 · Ascend 910B · Ubuntu 22.04 · Linux/Arm64 · Python 3.10 + 3.11** —
> `swr.cn-southwest-2.myhuaweicloud.com/wuzhenqing/cann:9.0.0-910b-ubuntu22.04-py3.11`

该镜像可用于统一华为云上的操作系统、CANN 和 Python 基础环境，减少不同机器之间的环境差异：

```bash
git clone --branch main --depth 1 https://gitcode.com/li-xingHIT/numpilot.git
cd numpilot

export NUMPILOT_IMAGE='swr.cn-southwest-2.myhuaweicloud.com/wuzhenqing/cann:9.0.0-910b-ubuntu22.04-py3.11'
docker pull "$NUMPILOT_IMAGE"

# 进入镜像；NPU 使用还需添加华为云环境提供的设备与宿主机驱动映射参数
docker run --rm -it \
  -v "$PWD":/workspace/numpilot \
  -w /workspace/numpilot \
  "$NUMPILOT_IMAGE" bash

# 进入容器后确认实际解释器，不要只依据镜像标签判断 ABI
uname -m
command -v python3
python3 -c 'import platform; print(platform.python_implementation(), platform.python_version())'
```

当前镜像为 `linux/arm64`，CANN 位于 `/usr/local/Ascend/cann-9.0.0`；镜像同时包含 Python 3.10
与 3.11，当前 `PATH` 优先使用 Miniconda Python 3.10。镜像标签可能更新，因此应以上述容器内检查
结果为准。随仓 AsNumpy wheel 是 `cp310`：选择 CPython 3.10.x 时可以使用下面的安装器，但当前
strict-NPU 结论仍进一步绑定 CPython 3.10.20 和 Ascend 910B4；选择 Python 3.11 时只能先使用核心
CLI、shim 和 Agent Skill，或者重新构建、登记并验证新的 `cp311` AsNumpy 组件。

镜像不等同于已经通过 NumPilot strict-NPU 资格验证，也不会替代宿主机驱动、固件、NPU 设备授权
或容器设备映射。不要通过改名强行安装 `cp310` wheel；正式部署还应固定并复核镜像 digest。

### 核心 CLI

这条路线无需 CANN 或 NPU，可用于静态分析、shim 后端和 Agent Skill 模式：

```bash
git clone --branch main --depth 1 https://gitcode.com/li-xingHIT/numpilot.git
cd numpilot
python3 -m venv .venv
. .venv/bin/activate
python -m pip install .
python -m numpilot doctor --backend shim
```

### Ascend NPU 运行环境

请先安装匹配的 Ascend 驱动和 CANN。随仓组件要求 Linux/AArch64 和 CPython 3.10；当前严格画像
使用 CPython 3.10.20，以及位于 `/usr/local/Ascend/cann-9.0.0` 的 CANN 9.0.0。安装前先检查：

```bash
uname -m
python3 -c 'import platform; print(platform.python_implementation(), platform.python_version())'
mkdir -p "$HOME/.local/venvs"
chmod 700 "$HOME/.local/venvs"

./install.sh --allow-network \
  --python "$(command -v python3)" \
  --cann-env /usr/local/Ascend/cann-9.0.0/set_env.sh \
  --venv "$HOME/.local/venvs/numpilot" \
  --persistent-root "$HOME/.local/share/numpilot"

. "$HOME/.local/share/numpilot/activate-numpilot.sh"
python -m numpilot bootstrap --verify --json
python -m numpilot doctor --backend npu
```

两项检查必须分别输出 `aarch64` 和 `CPython 3.10.x`。如果正确解释器不是默认的
`python3`，请改用 `--python /absolute/path/to/python3.10`；目前只有精确的 3.10.20 画像带有严格验证结论。

这些命令会安装或创建：

| 路线 | 会安装或创建 | 不会安装 |
|---|---|---|
| `pip install .` | NumPilot CLI、NumPy、Loguru、Rich、Skill 包，以及作为包数据保存的锁定 AsNumpy 组件附件 | Ascend 驱动、CANN、OPP、已激活的 AsNumpy 运行时、CANNBot 或大模型 |
| `install.sh` | 私有虚拟环境、NumPilot 及其 Python 依赖、随仓精确 AsNumpy wheel、bootstrap 回执、激活脚本，以及状态/配置/缓存目录 | Ascend 驱动、固件、CANN、OPP、系统原生库、模型凭据，也不会修改 shell 启动文件 |

CANNBot 是可选项。可用 `--with-cannbot --accept-cannbot-license` 启用固定 checkout；也可用
`--cannbot-source /absolute/path/to/cannbot-skills --accept-cannbot-license` 选择已有的精确 checkout。
配置模型终端中的 `add_one` 缺口示例需要选择其中一种方式；已有合格算子的普通映射不需要 CANNBot。
安装器目前以 editable 方式安装 NumPilot，并把激活脚本绑定到当前 checkout，因此请保持仓库路径
不变。更多前置条件与故障排查见[安装指南](docs/installation.zh-CN.md)。

## 选择工作流

| 工作流 | 模型来源 | 是否需要 `llm.json` |
|---|---|---|
| 终端工作流 | 在 NumPilot 中配置的 OpenAI-compatible endpoint | 需要 |
| Agent Skill 工作流 | IDE 或侧边栏中已经运行的 AI | 不需要，而且智能体不得读取或调用该文件 |

### A. 配置 `llm.json` 的终端工作流

先让 NumPilot 打印准确配置路径，再创建私有模板：

```bash
python -m numpilot paths --get llm_config
python -m numpilot configure-llm --init
```

没有路径覆盖时，配置文件位于 `~/.config/numpilot/llm.json`（设置了 `XDG_CONFIG_HOME` 时位于
`$XDG_CONFIG_HOME/numpilot/llm.json`）。激活上面的 NPU 安装后，路径是
`$HOME/.local/share/numpilot/.numpilot-config/llm.json`。应以 `paths --get llm_config` 实际打印的路径
为准。文件内容如下：

```json
{
  "schema_version": 2,
  "base_url": "https://your-openai-compatible-service.example/v1",
  "auth_token": "REPLACE_WITH_A_PRIVATE_TOKEN",
  "model": "exact-model-id",
  "provider": "provider-label",
  "protocol": "openai-chat-completions",
  "response_model_aliases": []
}
```

填写后先做本地检查，再显式发起一次联网探测：

```bash
python -m numpilot configure-llm --check
python -m numpilot configure-llm --probe
numpilot
```

进入终端后依次输入 `check llm`、`migrate`、源码绝对路径、`y`、`execute`、`exit`。
`y` 授权生成，`execute` 另行授权执行已展示且绑定哈希的源码与候选。配置目录权限必须为 `0700`，
`llm.json` 必须为 `0600`，且绝不能提交到仓库。

### B. 完全不配置 `llm.json` 的 Agent Skill 工作流

在能够读取工作区文件并执行经授权命令的 IDE/侧边栏 AI 中打开克隆后的仓库。不要运行
`configure-llm` 或 `check llm`，直接把下面的提示词发给侧边栏 AI：

```text
请阅读 /absolute/path/to/numpilot/skill/SKILL.md，并使用其中的 agent/no-llm 路线。
请用当前侧边栏已经提供的模型迁移 /absolute/path/to/source.py。
不要读取 llm.json，不要运行 configure-llm 或 check llm，也不要调用 NumPilot 配置的模型。
先只读分析源码并向我展示计划；我确认后，由你生成候选，并只把框架支持的制品提交给
NumPilot 的确定性门禁。请如实报告 stopped、unverified、数值、运行时和性能状态，不得升级结论。
```

要体验随仓示例，可把源码替换为 `examples/add_one_migration_demo.py` 的绝对路径。Skill 是操作合同，
不是模型本身：侧边栏 AI 负责生成候选，NumPilot 负责其支持的分析和验证门禁。未安装 NPU 运行环境时，
这条路线可以分析和生成草稿，但不能声称通过 strict-NPU 资格验证；没有登记制品适配器的路线会如实
停在智能体草稿阶段，不会伪造后续回执。

<p align="center">
  <img src="docs/assets/architecture/numpilot-two-routes.zh-CN.png" alt="终端与 Agent Skill 两种使用方式" width="100%">
</p>

## 核心能力

- 面向受支持 NumPy 子集的全文件调用点与有界数据流分析。
- 绑定 AsNumpy、CANN 和真实 Ascend SoC 的运行时能力匹配。
- 带响应、AST、源码和制品完整性检查的模型辅助迁移。
- 官方算子优先的发现、已登记缺口处理流程和受约束的算子草稿生成。
- 面向受支持画像的隔离数值对拍与真实 NPU 资格验证。
- 范围绑定的性能测试及机器可读 JSON/Markdown 报告。
- 交互式终端与 Agent Skill 两种使用方式。

## 系统架构

<p align="center">
  <a href="docs/numpilot-zh-v6.svg">
    <img src="docs/numpilot-zh-v6.svg" alt="NumPilot v6 系统架构" width="100%">
  </a>
</p>

<p align="center"><sub>点击图片可查看完整尺寸的系统架构图。</sub></p>

## 工作原理

```text
分析源码 → 匹配运行时能力 → 生成候选 → 资格验证 → 数值验证 → 输出报告
```

每个阶段都采用失败关闭策略。生成的算子在完成所需的构建、测试、公共 API、数值和真实设备门禁前，
始终只是未验证草稿。

## 兼容性

NumPilot 安装器允许用户指定 CANN、虚拟环境、状态、缓存和模型配置路径，不依赖某一位开发者的工作
目录；但当前 strict-NPU 画像仍绑定下表中的精确 CANN 身份和位置。NPU 二进制兼容范围比源码级 CLI
更窄：

| 模式 | 环境要求 | 支持范围 |
|---|---|---|
| 核心 CLI / shim | Linux、Python 3.10+、NumPy 1.26+ | 无 NPU 的静态分析、CLI 体验和数值诊断；不产生 strict-NPU 结论 |
| 随仓 NPU 组件 | Linux/AArch64、CPython 3.10、CANN 9.0.0、兼容的 Ascend 驱动/固件/OPP | 安装 `components/asnumpy/` 中哈希绑定的 AsNumpy wheel |
| 已严格验证画像 | Ascend 910B4、位于 `/usr/local/Ascend/cann-9.0.0` 的 CANN 9.0.0、CPython 3.10.20、NumPy 2.2.6、Loguru 0.7.3、随仓 AsNumpy 组件 | 真实 NPU 资格验证及范围绑定的性能门禁 |
| 其他机器 | 不同 SoC、CANN 版本或位置、Python ABI、操作系统或原生组件 | 不会自动视为合格；必须建立并评审新的运行时画像，再重新构建和验证组件，才能声明 strict-NPU 支持 |

随仓 wheel 的标签是 `linux_aarch64` 而不是 `manylinux`，目前没有完成跨 Linux 发行版验证。当前
原生构建要求 glibc 2.34+、兼容的 libstdc++、`libfmt.so.8`、`libspdlog.so.1`、
`libpython3.10.so.1.0` 及匹配的 CANN 运行库。因此，包元数据声明 Python 3.10+ 并不等于已经证明
NPU 兼容。每台新机器都应先运行 `bootstrap --verify` 和 `doctor`；这两个命令只验收安装，
每个工作负载仍必须分别通过能力与资格门禁。NumPilot 不负责安装 Ascend 驱动、固件、CANN 或 OPP。

## 安全说明

- 只迁移和执行可信源码；`python -I` 隔离不是操作系统沙箱。
- 配置的远端模型可能接收源码，并可能产生调用费用。
- 生成的 kernel 和接入代码在全部必要门禁通过前始终未验证。
- 运行时身份或制品发生漂移时，系统会明确阻断或要求重新审计。

安全问题请按 [SECURITY.md](SECURITY.md) 报告。

## 文档

- [安装与验收](docs/installation.zh-CN.md)
- [架构与信任边界](docs/architecture.md)
- [模型配置与传输合同](docs/model-contract.md)
- [数值验证](docs/script-verification.md)
- [strict-NPU 性能测试](docs/script-benchmark.md)
- [Agent Skill 合同](skill/SKILL.md)
- [侧边栏工作流](skill/references/codearts-sidebar.md)
- [变更记录](CHANGELOG.md)

## 参与贡献

欢迎贡献。提交 Pull Request 前，请阅读 [CONTRIBUTING.md](CONTRIBUTING.md)，为行为变化补充测试，并运行
相关测试和发布树检查。

## 许可证

NumPilot 使用 [Apache License 2.0](LICENSE)。第三方组件保留各自许可证，详见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。产品名称和 Logo 权利归各自所有者，不表示赞助或背书。
发布二进制发行件前，须确认随附二进制和品牌素材的再分发条款。
