<div align="center">

<h1><img src="docs/assets/brand/numpilot-wordmark.png" alt="NumPilot" width="560"></h1>

### 从 NumPy 程序到经过验证的华为 Ascend NPU 执行

<p>
  <a href="README.md">English</a> ·
  <a href="docs/installation.zh-CN.md">安装指南</a> ·
  <a href="docs/architecture.md">系统架构</a> ·
  <a href="evidence/README.md">证据审计</a> ·
  <a href="CONTRIBUTING.md">参与贡献</a>
</p>

<p>
  <img alt="版本" src="https://img.shields.io/badge/version-0.3.0a1.dev0-6D5CE7?style=flat-square">
  <img alt="状态" src="https://img.shields.io/badge/status-development--snapshot-F59E0B?style=flat-square">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&amp;logo=python&amp;logoColor=white">
  <img alt="发布分支" src="https://img.shields.io/badge/release_branch-main-0F766E?style=flat-square">
  <img alt="许可证" src="https://img.shields.io/badge/license-Apache--2.0-2EA44F?style=flat-square">
</p>

<p>
  <a href="docs/installation.zh-CN.md"><img alt="快速安装" src="https://img.shields.io/badge/快速安装-打开指南-5B4EE8?style=for-the-badge"></a>
  <a href="docs/numpilot-zh-v6.svg"><img alt="查看系统架构" src="https://img.shields.io/badge/系统架构-查看架构图-008AA3?style=for-the-badge"></a>
</p>

</div>

> [!IMPORTANT]
> NumPilot 当前是 Alpha 开发快照，不是已经发布的 RC。安装器或 `doctor` 通过只证明 NPU 基础可用；
> strict-NPU 与性能结论只对生成证据时绑定的精确运行时画像和哈希回执有效。

**NumPilot 的目标是将输入 NumPy 程序中的每个 API 自动迁移为功能等价的 NPU 算子实现。**
在保持程序语义和计算结果一致的前提下，充分发挥 NPU 的大规模并行计算能力，加速大规模数据处理。

当前 Alpha 快照支持可静态分析的普通单文件 NumPy 子集。输入需使用顶层 `import numpy as np`，
并在迁移前通过失败关闭的全程序能力门禁。系统会对已安装的 AsNumpy/CANN/SoC 运行时、模型响应、
迁移后 AST、数值结果及可选性能结论分别建立并核验证据。

> **模型负责提出候选；确定性证据决定候选能否采用。**

<table align="center">
  <tr>
    <td align="center"><img src="docs/assets/brand/asnumpy-mark.png" alt="AsNumpy" height="54"><br><sub>AsNumpy</sub></td>
    <td align="center"><img src="docs/assets/brand/cann-wordmark.png" alt="CANN" height="34"><br><sub>CANN / ACLNN</sub></td>
    <td align="center"><img src="docs/assets/brand/ascend-wordmark.png" alt="Ascend" height="38"><br><sub>Ascend NPU</sub></td>
    <td align="center"><img src="docs/assets/brand/codearts-icon.png" alt="CodeArts" height="54"><br><sub>可选 CodeArts 协作</sub></td>
  </tr>
</table>

## 系统架构：从 NumPy 输入到经过验证的 NPU 执行

<p align="center">
  <a href="docs/numpilot-zh-v6.svg">
    <img src="docs/numpilot-zh-v6.svg" alt="NumPilot 架构：NumPy 分析、AsNumpy 能力分流、缺口补全与 Ascend NPU 验证执行" width="100%">
  </a>
</p>

<p align="center">
  <sub>
    打开<a href="docs/numpilot-zh-v6.svg">中文高清架构图</a> ·
    <a href="docs/numpilot-en-v6.svg">English diagram</a> ·
    <a href="docs/numpilot-zh.pptx">中文 PPT</a> ·
    <a href="docs/assets/architecture/numpilot-architecture-zh.svg">可机器审计的边界图</a>
  </sub>
</p>

v6 架构图使用实线表示当前原语，使用虚线表示目标自动化路径。如果平台无法渲染 SVG 或读取轮廓化文字，
下方文本摘要保留了相同语义。

### 如何阅读这张架构图

| 阶段 | NumPilot 的处理 | 门禁或输出 |
|---|---|---|
| 1. 未分类工作负载 | 解析完整程序，提取 NumPy 调用点、签名和有界数据流 | 入口不预先标注 API 已支持或缺失 |
| 2. 能力检查 | 依据精确的 AsNumpy/CANN/SoC 画像逐调用点匹配，并独立记录判定 | 采用最严格的逐调用结果，生成失败关闭的整份工作负载结论 |
| 3. 全文件迁移 | 全部调用合格后，用户配置模型才改写完整文件；NumPilot 随后执行精确响应与 AST 检查 | 只产生迁移候选，不自动信任模型结果 |
| 4. 缺口补全 | API 缺失时优先查找可复用的华为官方算子；只有官方路径明确无匹配后，才由 CANNBot Skills 约束用户配置模型补写自定义算子 | 两条路线统一汇入 AsNumpy 公共 API 接入边界 |
| 5. 资格化与回流 | 接入候选依次通过制品、构建、单测、数值、公共 API 和真实 NPU 门禁，然后重新扫描原工作负载 | 只有重扫后全部调用合格，才能返回迁移与验证主链 |

<details>
<summary><strong>文本版架构流程与当前路由边界</strong></summary>

图中上半部分表示 API 已经具备合格能力时的普通主链：

```text
未分类 NumPy 源码
  → 全文件 AST + 有界数据流分析
  → 逐调用点、运行时绑定的能力判定
  → 失败关闭的整份工作负载结论
  → 用户配置模型生成候选改写
  → 模型回执 + 精确 AST 门禁
  → 独立数值验证
  → 可选、画像绑定的 strict-NPU 基准测试
  → 不可覆盖的证据回执
```

图中下半部分表示已经确认 API 缺口后的补全闭环：

```text
已识别 API 缺口
  → A. 复用已安装 ACLNN 候选，或审计固定版本的 ops-nn 源码候选
  → B. 两层官方查询均明确无匹配后，由固定 CANNBot Skills + 用户配置模型
       生成自定义算子草稿
  → 通过 AsNumpy 公共 API 边界完成接入
  → 资格化 → 激活能力 → 重扫原始工作负载
  → 仅当所有调用点合格后才重新进入迁移
```

不同路线的当前输出刻意保持区别：已安装官方二进制发现可到 `CANDIDATE_READY`；目前只有已审计的
installed-ACLNN MatMul 定向切片可到 `ADAPTER_CANDIDATE`；固定 `ops-nn` 当前只提供
`SOURCE_EVIDENCE_ONLY`；CANNBot/model 路线产生 `UNVERIFIED_OPERATOR_DRAFT`。通用集成、资格化、
激活、重扫、混合路线编排和回滚仍是目标路径，尚未形成端到端自动闭环。

</details>

> **核心不变量：**输入工作负载不会被预先分类；缺口不会被静默绕过；新补全 API 必须接入
> AsNumpy 并通过原工作负载重扫，迁移流程才能继续。

## 当前已经实现什么

| 领域 | 当前状态 | 边界 |
|---|---|---|
| 普通 NumPy 静态分析 | 已实现 | 全文件静态分析，不执行源码、不调用模型 |
| 五态能力判定 | 已实现 | 绑定当前 AsNumpy/CANN/SoC 身份与审计证据 |
| 配置模型迁移 | 已实现 | OpenAI Chat Completions-compatible；模型身份、响应、AST、哈希任一异常即失败关闭 |
| 数值验证 | 已实现 | 原脚本与迁移脚本分别由 `python -I` worker 执行；这不是 OS 沙箱 |
| strict-NPU resident 基准 | 已实现但范围受限 | 要求精确合格画像，不代表通用能力或冷启动 E2E |
| 官方算子发现 | 已实现 | 先检查已安装 ACLNN，再检查固定本地 `ops-nn` 只读源码证据；发现本身不负责构建或接入 |
| 自定义算子草稿 | 部分实现 | 两层明确无匹配后，CANNBot Skills 约束用户配置模型；输出工程仍未验证 |
| 通用缺口补全 | 部分实现 | 官方二进制通用链到 `CANDIDATE_READY`；仅已审计 MatMul 到 `ADAPTER_CANDIDATE`；资格化与激活仍在规划 |
| CodeArts 协作 | 可选集成 | 仓库包含角色/Skill；真实控制面仍需在目标账号验证 |

## 快速开始

### CPU/shim 开发体验

无需 CANN 或 NPU，即可查看 CLI 并运行 shim 冒烟检查：

```bash
git clone --branch main --depth 1 https://gitcode.com/li-xingHIT/numpilot.git
cd numpilot
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
python -m numpilot --help
python -m numpilot doctor --backend shim
```

完整测试套件是固定发布矩阵的门禁，不是任意 Python 3.10+ 环境都能复现的快速检查；运行时必须隔离
CANN 注入的 `PYTHONPATH` 和代理环境，详见[贡献指南](CONTRIBUTING.md)。

### NPU 源码优先安装

目标机器必须预装兼容的 Ascend 驱动、CANN、OPP、Python 开发头文件、Git、CMake、Make 或 Ninja、
GCC 11.2+，以及系统 `fmt`/`spdlog` CMake 开发包。NumPilot 不会运行 `sudo`，也不会修改 shell
启动文件。

```bash
./install.sh --allow-network \
  --cann-env /usr/local/Ascend/cann-9.0.0/set_env.sh \
  --venv /absolute/user-owned/numpilot-venv \
  --persistent-root /absolute/user-owned/numpilot-runtime

. /absolute/user-owned/numpilot-runtime/activate-numpilot.sh
python -m numpilot bootstrap --verify --json
python -m numpilot doctor --backend npu
```

`--venv` 的直接父目录必须预先存在、归当前用户所有，且不能对 group/world 开放写权限。
请把示例路径替换为当前账号独占的绝对路径。`--allow-network` 是安装依赖和在缓存缺失时获取固定
AsNumpy 源码的显式授权。完整依赖矩阵、CANNBot 可选安装、安装回执、strict 画像检查、排错和安全
清理步骤见 **[完整安装指南](docs/installation.zh-CN.md)**。

当前源码树没有已核实的公开包发行版本，因此不要把 `pip install numpilot` 写成安装入口。

## 配置模型

```bash
python -m numpilot configure-llm --init
# 编辑命令打印出的私有 llm.json。
python -m numpilot configure-llm --check   # 只检查本地配置，不联网
python -m numpilot configure-llm --probe   # 一次真实请求，可能产生费用
```

配置目录权限必须为 `0700`，`llm.json` 必须为 `0600`。Token、私有 endpoint、代理地址和原始模型
响应正文不得提交。代理失败关闭与回执规则见[模型合同](docs/model-contract.md)。

> [!WARNING]
> 所有 configured-model 生成或修复请求都可能产生费用。`migrate` 会发送完整输入源码；
> gap/operator 路径会发送有界缺口合同、静态提取的语义函数、适用的草稿或诊断，以及选定的
> CANNBot Skill 上下文；AsNumpy adapter 路径会发送按目标投影的源码/头文件上下文、Skill 上下文、
> overlay 与适用的诊断。CANNBot 提供固定版本的工作流上下文，草稿仍由用户配置的远端模型生成。
> `verify-script`、`benchmark-script` 与 `run_demo.sh` 会以当前用户权限执行原脚本和候选；
> `python -I` **不是** OS 沙箱。只能发送有权披露的材料、执行可信程序；不可信输入须放入独立容器
> 或虚拟机。

## 运行 ordinary 主链

输入必须可信，并把最终结果赋值给一个明确变量（以下为 `result`）：

```bash
SOURCE=/absolute/path/to/trusted_script.py
RUN_DIR="$(mktemp -d "${TMPDIR:-/tmp}/numpilot-run.XXXXXX")"

python -m numpilot analyze "$SOURCE" --require-direct --out "$RUN_DIR/analysis.json"
python -m numpilot migrate "$SOURCE" \
  --mode script --engine llm --backend npu --fallback-policy reject \
  --out "$RUN_DIR/migration"
python -m numpilot verify-script "$SOURCE" \
  --migration-report "$RUN_DIR/migration/script-migration-report.json" \
  --output result --out "$RUN_DIR/verification.json"
```

`./run_demo.sh --quick` 是仓库的分阶段演示封装，会额外运行范围绑定基准。默认每次运行都会创建新的
仓库外状态目录，可用 `python -m numpilot paths --json` 查看配置路径。性能结论必须满足单独的
[strict-NPU 基准合同](docs/script-benchmark.md)。

分析器会先为每个调用点记录独立判定，再使用同一套五态语义生成整份工作负载的
失败关闭汇总结果：

| 状态 | 含义 |
|---|---|
| `DIRECT` | 精确运行时画像已资格化请求的调用表面 |
| `FALLBACK_CONSENT` | 存在已审计 CPU/Host 路径，需要用户明确授权 |
| `GAP_REQUIRED` | API 缺失，或用户拒绝已审计回退 |
| `CAPABILITY_AUDIT_REQUIRED` | 接口看似存在，但运行时身份或来源尚未资格化 |
| `BLOCKED` | 动态或未建模 Python 行为无法安全迁移 |

## 运行时支持与证据范围

包元数据支持 Python 3.10+，但这不等于 strict-NPU 兼容矩阵。当前 strict 画像绑定 Linux/AArch64、
Ascend 910B4、CANN 9.0.0、CPython 3.10.20，以及固定 AsNumpy 源码、安装文件和回执哈希。其他硬件、
CANN、Python ABI、wrapper、原生库或回执在独立审计前只能算基础安装或候选证据；未公开的工作记录
不作为发布证据。

仓库保留以下可审计开发态证据：

- [Poisson-Ritz 应用形态证据](evidence/competition-poisson-stage-20260828/README.md)
- [逐元素优势区间证据](evidence/competition-ordinary-20260828/README.md)
- [证据索引与离线审计命令](evidence/README.md)

这些回执不能外推到不同硬件、CANN、Python、AsNumpy、输入 shape、dtype 或计时口径。

## 分支与发布模型

仓库治理规范只允许两个长期分支：

- `main`：发布分支；Tag 和发布制品只能来自通过门禁的 `main` 提交。
- `develop`：集成分支，也是日常贡献与验证目标。

外部贡献应从 fork 向 `develop` 提交 Pull Request；发布提升通过受评审的 `develop → main` Pull
Request 完成。两个分支在 GitCode 上**必须**配置保护，禁止 force push、历史改写和未经评审的直接
修改。本地引用已经对齐；服务端默认分支、远端 heads 与保护规则仍须管理员核验。详见
[分支治理与发布约定](docs/SUBMISSION.md)。

## 文档

- [安装与验收](docs/installation.zh-CN.md)
- [当前架构与实现边界](docs/architecture.md)
- [文档中心](docs/README.md)
- [脚本数值验证合同](docs/script-verification.md)
- [strict-NPU 基准合同](docs/script-benchmark.md)
- [Gap 恢复状态机](docs/gap-recovery-state.md)
- [变更记录](CHANGELOG.md)
- [问题追踪](https://gitcode.com/li-xingHIT/numpilot/issues)
- [安全策略](SECURITY.md)
- [贡献指南](CONTRIBUTING.md)

## 许可证与商标

NumPilot 使用 [Apache License 2.0](LICENSE)。第三方组件保留各自许可证，详见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 与[品牌资产清单](docs/assets/brand/README.md)。产品名称和
Logo 仅用于技术识别，权利归各自所有者，不表示官方背书。截图衍生资产公开发布前，仓库所有者仍须
确认适用的商标与使用授权。
