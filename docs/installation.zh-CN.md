# 安装与验收 NumPilot

[English](installation.md) · [README](../README.zh-CN.md)

本文把三种不同结果分开说明，不能把前一层当作后一层的证明。

| 结果 | 能证明什么 | 是否需要 NPU |
|---|---|---:|
| CPU/shim 开发 | CLI、shim 冒烟与环境无关检查可以运行 | 否 |
| NPU 基础安装 | CANN、NPU 与固定 AsNumpy 构建可以完成冒烟执行 | 是 |
| strict-NPU 就绪 | 完整运行时身份与请求调用的已审计画像精确匹配 | 是，且画像必须精确 |

> [!WARNING]
> 当前 strict 画像绑定 Linux/AArch64、Ascend 910B4、CANN 9.0.0、CPython 3.10.20，以及精确的
> AsNumpy 源码、安装文件和回执哈希。包与安装器接受 Python 3.10+，但不同 SoC、CANN、Python
> ABI、wrapper、原生库或回执在独立资格化前必须返回 `CAPABILITY_AUDIT_REQUIRED`。

## 1. CPU/shim 开发环境

没有 Ascend 主机时，可用以下路径了解项目或参与开发：

```bash
git clone --branch main --depth 1 https://gitcode.com/li-xingHIT/numpilot.git
cd numpilot

python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'

python -m numpilot --version
python -m numpilot doctor --backend shim
env -u PYTHONPATH python -m pytest -q -p no:cacheprovider \
  tests/test_install_script.py tests/test_notebook_quickstart.py
```

包元数据要求 Python 3.10+，基础依赖为 `numpy>=1.26`、`loguru>=0.7`、`rich>=13`。
`doctor --backend shim` 只是开发检查，不证明 NPU 能力。
完整测试套件属于固定发布矩阵门禁：必须隔离 CANN 注入的 `PYTHONPATH` 与代理变量，并使用发布测试
Python 环境。任意 Python 3.10+ 主机不能自动视为完整测试资格化环境。

## 2. NPU 安装前提

运行固定组件模式的 `install.sh` 前，需要在 NumPilot 之外准备：

- 兼容的 Ascend 驱动、固件、CANN Toolkit 与 OPP；
- `npu-smi info` 可见的 NPU；
- Python 3.10+、`venv`/`pip`；
- 当前用户独占的 venv、运行回执空间和持久化状态目录。

固定 wheel 安装不需要 Git、CMake、Make/Ninja、GCC/G++、Python 开发头文件或系统
`fmt`/`spdlog` CMake 包。只有维护者显式选择源码优先构建时，才需要以下工具链；Ubuntu 主机管理员
可用示例准备：

```bash
sudo apt-get update
sudo apt-get install -y \
  git cmake make ninja-build g++ python3-dev python3-venv libfmt-dev libspdlog-dev
```

其他发行版的包名不同。CANN、驱动、固件和 OPP 必须按目标 Ascend/CANN 文档安装；NumPilot 不会
安装或修改这些组件。

固定组件模式开始前检查：

```bash
python3 --version
npu-smi info
test -f /usr/local/Ascend/cann-9.0.0/set_env.sh
```

源码优先模式再额外检查 `git --version`、`g++ --version`、`cmake --version` 和 Python
`Python.h`/`fmt`/`spdlog` 开发包。

请把 CANN 路径替换为实际准备审计的精确安装。选择 `latest` 可能通过基础冒烟，但不会自动让新版
CANN 进入当前 strict 画像。

## 3. 固定 AsNumpy 内部组件（发布/比赛模式）

正式发布或比赛交付时，直接使用仓库随附的 `components/asnumpy/`（或同一 release 的不可变附件）。
其中的 manifest 已绑定 wheel、LICENSE、SBOM、METADATA/RECORD 和安装后 native 哈希。不要使用
PyPI `latest`、版本范围或浮动 Git 分支；组件缺失或哈希不匹配就是发布阻断，安装器不得退回下载源码。

```bash
python -m numpilot bootstrap --apply \
  --asnumpy-component components/asnumpy \
  --receipt /secure/runtime/bootstrap-receipt.json
python -m numpilot bootstrap --verify --json
```

release manifest 还必须绑定 wheel 的 `METADATA`/`RECORD`、安装后的
`asnumpy/__init__.py` 与 native `_core` 哈希、Python/架构标签，以及已审计的 CANN/SoC 画像。
组件旁应保留上游 Apache-2.0 LICENSE，并在 SBOM/逐文件许可证清单中单列。参考锁定值、历史
wheel 摘要和升级/回滚流程见
[AsNumpy 内部组件锁定与发布合同](ASNUMPY-INTERNAL-COMPONENT-LOCK.zh-CN.md)。

当前附件只适用于 Linux AArch64、CPython 3.10 和 CANN 9.0.0；它不包含 CANN 驱动、固件或 OPP。
因此现场不再需要拉取/编译 AsNumpy 源码，但仍要在目标 venv 中执行一次本地 wheel 安装，并先
加载匹配的 CANN `set_env.sh`。

下方源码优先流程仅用于隔离地构建和审计下一版组件；它在显式授权时可能获取**固定**源码，
不是固定发行版的用户安装合同。

## 4. NPU 源码优先安装（开发/构建准备）

发布候选使用 `main`，集成开发才使用 `develop`。当前 Alpha 快照没有已核实的公开包发行，因此
不要用 `pip install numpilot` 替代以下流程。

选择两个由当前用户独占的绝对目录。把 venv 与运行状态放在仓库外，更方便源码审查、安全清理和
证据边界管理。

```bash
git clone --branch main --depth 1 https://gitcode.com/li-xingHIT/numpilot.git
cd numpilot

./install.sh --allow-network \
  --asnumpy-source /absolute/path/to/pinned/asnumpy \
  --cann-env /usr/local/Ascend/cann-9.0.0/set_env.sh \
  --venv /absolute/user-owned/numpilot-venv \
  --persistent-root /absolute/user-owned/numpilot-runtime

. /absolute/user-owned/numpilot-runtime/activate-numpilot.sh
```

`--allow-network` 是必需的显式授权。安装器可能访问选择的 Python 包索引，并在缓存缺失时获取固定
AsNumpy 源码。它不会运行 `sudo`、修改 shell 启动文件、保存模型凭据或安装到系统 Python。

可选方式：

```bash
# 使用 origin/commit/tree 精确匹配且干净的 AsNumpy checkout。
./install.sh --allow-network \
  --asnumpy-source /absolute/path/to/pinned/asnumpy \
  --cann-env /absolute/path/to/CANN/set_env.sh \
  --venv /absolute/user-owned/numpilot-venv \
  --persistent-root /absolute/user-owned/numpilot-runtime

# 固定组件模式下显式选择 Python 包索引；不要把凭据写进 URL。
./install.sh --allow-network \
  --pip-index-url https://mirror.example.invalid/simple \
  --asnumpy-component "$PWD/components/asnumpy" \
  --cann-env /absolute/path/to/CANN/set_env.sh \
  --venv /absolute/user-owned/numpilot-venv \
  --persistent-root /absolute/user-owned/numpilot-runtime
```

源码构建没有 OS 级网络沙箱，依赖范围也不是完整发布锁。固定 Git commit 不保证本地 wheel 逐字节
一致；正式发布必须保存资格化时实际使用的 wheel、哈希、constraints 和 wheelhouse。

## 5. 安装后分层验收

所有检查都应在生成的 activation 环境中运行。

### 5.1 身份与隔离依赖

```bash
python -m numpilot --version
python -m numpilot paths --json
env -u PYTHONPATH python -m pip --isolated check
```

清除 `PYTHONPATH` 是为了把项目 venv 与 CANN 注入的可选 Python 包隔离。CANN 完整注入环境中的
vendor 包缺少可选依赖，不能单独作为 NumPilot venv 损坏的证明。

### 5.2 Bootstrap 回执与 AsNumpy

```bash
NUMPILOT_RECEIPT="$(python -m numpilot paths --get bootstrap_receipt)"
python -m json.tool "$NUMPILOT_RECEIPT"
python -m numpilot bootstrap --verify --json
```

apply 回执应包含 `status="ready"`、`ready=true`、固定 repository/commit/tree 身份和通过的 AsNumpy
验证。`bootstrap --verify --json` 应报告官方 `(1000, 1000)` `float32` 冒烟结果和非空原生模块
身份。必须同时审查 apply 回执和现场 verify；单独 verify 会采集当前哈希，但不会自行重建 provenance。

再做一次直接本地检查：

```bash
python - <<'PY'
from importlib.metadata import version
import asnumpy as ap

device = ap.ones((1000, 1000), dtype=ap.float32)
host = device.to_numpy()
assert host.shape == (1000, 1000)
assert str(host.dtype) == "float32"
print("asnumpy", version("asnumpy"), "smoke=PASS")
PY
```

该命令只证明本地 AsNumpy/NPU 可以执行这一冒烟算子。

### 5.3 NPU doctor 与 strict 画像判定

```bash
python -m numpilot doctor --backend npu

python -m numpilot analyze examples/poisson_source_smooth_l1_map.py \
  --require-direct --json
```

`doctor` 检查 CANN/NPU/AsNumpy 基础运行；需要机器可读结果时可使用 `doctor --json`，但它不是
strict 画像资格化。`analyze --require-direct` 是下一道门。退出码 `1` 且结果为 `CAPABILITY_AUDIT_REQUIRED`，表示真实的
运行时画像漂移，不是应该用 `--skip-env-check` 绕过的普通安装错误。

## 6. 可选 CANNBot Skills

ordinary direct 路径不需要 CANNBot。只有在阅读官方
[CANNBot LICENSE](https://gitcode.com/cann/cannbot-skills/blob/master/LICENSE) 并接受 **CANN Open Software
License Agreement Version 2.0** 后，才为 gap/operator 路径安装。传入接受参数前，还应在干净 CANNBot
checkout 中运行 `git show 1ef6372a538f39b0b7ed34666a3e1f465fb4936a:LICENSE` 核对固定提交的精确条款。

同时安装 AsNumpy 和固定 CANNBot checkout：

```bash
./install.sh --allow-network \
  --with-cannbot --accept-cannbot-license \
  --cann-env /absolute/path/to/CANN/set_env.sh \
  --venv /absolute/user-owned/numpilot-venv \
  --persistent-root /absolute/user-owned/numpilot-runtime
```

也可以在基础安装后增加固定 checkout：

```bash
python -m numpilot setup-skills --accept-cannbot-license
python -m numpilot setup-skills --check
python -m numpilot bootstrap --verify \
  --with-cannbot --accept-cannbot-license
```

固定身份为 `cann/cannbot-skills@1ef6372a538f39b0b7ed34666a3e1f465fb4936a`。验证会核对精确 origin、
HEAD 和 NumPilot 实际消费的有界 Skill 文件集合；它不代表整个仓库或所有 Skill 均已获认证。

只读检查已有的精确 checkout，不触发获取：

```bash
python -m numpilot setup-skills \
  --root /absolute/path/to/cannbot-skills --check
```

CANNBot Skills 与 `cann/ops-nn` 是不同依赖。安装器不会自动获取 `ops-nn`；当前 official-source
resolver 只接受用户显式提供、与版本匹配的本地 checkout，且现有源码映射只固定 CANN 9.0.0。

## 7. 配置并探测模型

```bash
python -m numpilot configure-llm --init
# 编辑命令打印出的私有 schema-2 llm.json。
python -m numpilot configure-llm --check
python -m numpilot configure-llm --probe
```

`--check` 不联网，只检查本地配置和当前代理门禁。`--probe` 会执行真实请求、可能产生费用，并要求
预期 readiness marker。
配置需要 `base_url`、`auth_token`、`model`；`provider` 只是审计标签，当前协议为
`openai-chat-completions`。

配置目录必须为 `0700`，`llm.json` 必须为 `0600`。Token、endpoint、代理地址和原始响应正文不得
提交。

> [!WARNING]
> 所有 configured-model 生成或修复调用都可能产生费用。`migrate` 会发送完整输入源码。
> gap/operator 生成与修复会发送有界缺口合同、静态提取的语义函数、选定的固定 CANNBot Skill
> 内容，以及适用的当前文件或诊断。AsNumpy adapter 生成与修复只发送本地目标投影选中的源码/头文件
> 上下文、仓库自带的 integration Skill、overlay 与适用的诊断。CANNBot 提供工作流上下文；生成
> 未验证草稿的是用户配置的远端模型，而不是 CANNBot 本身。使用非本地 endpoint 前应审查这些
> payload 类别。
>
> `verify-script`、`benchmark-script` 与演示会以当前用户权限执行原脚本和生成候选；`python -I`
> 进程隔离不是 OS 沙箱。只能发送有权披露的材料并执行可信程序；不可信输入必须放入独立容器或
> 虚拟机。

endpoint 必须经过代理时：

```bash
export HTTP_PROXY=http://127.0.0.1:PORT
export HTTPS_PROXY=http://127.0.0.1:PORT
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTPS_PROXY"
export NUMPILOT_LLM_REQUIRE_PROXY=1
python -m numpilot configure-llm --probe
```

失败关闭代理模式会在网络传输前拒绝缺少协议匹配代理或被 `NO_PROXY` 绕过的目标。

## 8. 首次运行

针对自己的可信脚本，请使用[中文 README 的完整 `analyze → migrate → verify-script` 示例](../README.zh-CN.md#运行-ordinary-主链)。
`./run_demo.sh --quick` 运行仓库内置的分阶段示例，并额外执行范围绑定基准。若模型配置位于独立
运行时目录，可显式传入 `--llm-config /absolute/persistent/numpilot-runtime/.numpilot-config/llm.json`；
每次运行都会写入 `python -m numpilot paths --json` 所示的新仓库外目录。

## 9. 常见问题

| 现象 | 检查方向 |
|---|---|
| 缺少 `Python.h` | 安装所选解释器对应的 Python 开发头文件 |
| 无法创建 venv | 安装发行版对应的 Python venv 包 |
| 缺少 `fmt`/`spdlog` CMake package | 安装系统开发包；NumPilot 不会运行 `sudo` |
| 找不到 CANN 环境 | 传入精确 `--cann-env /.../set_env.sh`，避免混用多个 CANN 路径 |
| 找不到 `npu-smi` 或 `libascendcl.so` | 在 NumPilot 外修复驱动/CANN 安装 |
| `bootstrap --verify` 失败 | 检查 apply 回执、模块路径、原生哈希和生成的 activation 文件 |
| `doctor` 通过但分析要求审计 | 基础安装成功，但 strict 画像不匹配 |
| CANNBot 检查失败 | 检查精确 origin、commit、Skill 文件是否干净、缓存路径和许可证决定 |
| 模型 probe 失败 | 检查私有配置、TLS、精确响应模型、代理变量与 `NO_PROXY` |
| 自定义回执路径被拒绝 | 父目录须已存在、当前用户所有、不是符号链接、不可被组/其他用户写入 |

新 shell 应 source 生成的 activation 文件，不要手工拼接多个 CANN 版本。该脚本支持重复 source，并会
清理已知 toolkit/devlib 冲突。

## 10. 安全清理

NumPilot 没有 `uninstall` 命令。先记录所有精确路径：

```bash
python -m numpilot paths --json
```

保持 NumPilot 专用 venv 已激活时，可选择只卸载其中的包：

```bash
python -m pip uninstall numpilot asnumpy
deactivate
```

随后逐一核对专用 venv 与 persistent-root 的绝对路径，再决定是否删除。卸载包不会删除
`python -m numpilot paths --get llm_config` 指向的私有 Token 文件；如有需要应单独处理这个精确文件。

安装器从未修改 CANN、驱动、固件、OPP 或 shell 启动文件，这些组件不属于 NumPilot 清理范围。
禁止对 home 或任何未经核对的宽泛路径执行递归删除。
