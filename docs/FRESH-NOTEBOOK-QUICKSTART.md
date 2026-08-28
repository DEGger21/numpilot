# NumPilot 全新机器源码验收

本文只面向一台与开发机没有代码、缓存、wheel 或私有配置重叠的新机器。用户只手工克隆
NumPilot；AsNumpy 从固定官方源码现场编译。CANNBot 只在 gap/operator lane 中从固定官方
提交拉取并校验，direct migration 不依赖它。

当前验收不接入 ops-nn，也不使用 wheelhouse。NumPilot 锁定的 AsNumpy source profile 中
公开 `matmul` 会回到 host NumPy；对官方最新提交的独立审计也得到同样结论。因此第一轮
可在用户明确接受 fallback 后验证兼容性迁移；数值实跑必须是审核产物后的独立授权步骤。两者都不得
声明 strict-NPU 或发布加速比，也不得复用历史 P0 wheel 的 180× 结果。

## 1. 新机器前置条件

- Linux aarch64、Ascend 910B、驱动、CANN 和 `npu-smi` 已由平台准备；
- CPython 3.10+，包含 `venv` 和 `Python.h`；
- Git、GCC/G++ 11.2+；
- 系统 fmt、spdlog CMake 开发包。Ubuntu 通常对应 `libfmt-dev`、
  `libspdlog-dev`；NumPilot 不会自动运行 sudo；
- 能访问 GitCode 和 Python 包索引。

不要从旧机器复制 `.venv`、`~/.cache/numpilot`、bootstrap receipt、AsNumpy wheel、
CANNBot checkout、LLM Token 或 `llm.json`。

> 发布状态（2026-08-27）：框架 hardening 集成候选尚无远端不可变 tag；`v0.2.0a2` 只是
> 基线且不包含本文合同。全新机器必须等待发布说明给出新的不可变 tag，不能用旧 tag 冒充
> 当前验收。

## 2. 只克隆 NumPilot

```bash
NUMPILOT_RELEASE_TAG="${NUMPILOT_RELEASE_TAG:?请填写 Day 5 发布说明中的不可变 tag}"
test "$NUMPILOT_RELEASE_TAG" != v0.2.0a2
git clone --branch "$NUMPILOT_RELEASE_TAG" --depth 1 \
  https://gitcode.com/li-xingHIT/numpilot.git
cd numpilot
git status --short --branch
git rev-parse HEAD
```

HEAD 必须是发布说明所列的新 tag commit，工作树必须干净。

## 3. 一键源码安装

direct lane 默认不安装 CANNBot：

```bash
PERSISTENT_ROOT=/absolute/platform-persistent/root
./install.sh --allow-network \
  --persistent-root "$PERSISTENT_ROOT"
source "$PERSISTENT_ROOT/activate-numpilot.sh"
```

只有使用 gap/operator lane 时，才审阅固定 CANNBot commit
`1ef6372a538f39b0b7ed34666a3e1f465fb4936a` 的
[CANN Open Software License Agreement v2.0](https://gitcode.com/cann/cannbot-skills/blob/1ef6372a538f39b0b7ed34666a3e1f465fb4936a/LICENSE)，
接受后给安装命令增加 `--with-cannbot --accept-cannbot-license`。标准持久 HOME 可省略
`--persistent-root`；安装器会打印 activation 路径。临时 HOME/Notebook 应把根放在平台明确
持久化的目录中。

若当前机器的默认 Python 私有镜像缺少依赖，必须由用户显式选择可信索引。例如：

```bash
./install.sh --allow-network \
  --pip-index-url https://pypi.org/simple
```

该选项只影响初始 Python 构建/运行依赖安装；安装器失败后不会自动换源。不要把包含
凭据的私有 URL 写入 Notebook 或仓库。

如 CANN 没有在当前 shell 配好，显式提供安装包的环境脚本：

```bash
./install.sh --allow-network \
  --cann-env /absolute/path/to/fixed-cann/set_env.sh
```

安装器会：

1. 在仓库内创建或复用真实的 `.venv`，绝不覆盖非 venv 路径；
2. 安装 NumPilot `.[bootstrap]` 和明确的源码构建工具；
3. 解析持久路径，原子写入并立即 source 无凭据 `activate-numpilot.sh`，固定
   venv、config/state/cache/receipt 与所选 CANN；它只清理 CANN toolkit/devlib 冲突，保留
   driver、NNAL/ATB 和无关自定义路径；
4. 在同一激活环境中展示只读 bootstrap plan；
5. 拉取并核验 NumPilot 固定的官方 AsNumpy commit/tree；
6. 现场执行 CMake/PEP 517 构建、校验 wheel RECORD、安装并跑 README smoke；
7. 仅在显式选择 gap/operator lane 时拉取并逐文件校验固定 CANNBot bundle；
8. 将回执写入 `python -m numpilot paths --get bootstrap_receipt` 所示的 XDG 状态路径；
   只有用户显式给出 `--receipt PATH` 时才换到该路径；
9. 执行离线 `bootstrap --verify` 和 `doctor --backend npu`，全部通过后才打印 ready。

该 receipt 不只用于安装日志。NumPilot 运行时会以它绑定现场编译后可能不可重现的
`_core.so` hash，同时继续用内置审计 hash 核对每个 Python wrapper；仅有 commit/tree、
receipt、当前模块与包版本全部一致时，才授予 pinned-source capability。

脚本不会保存或写死代理。若新机器确实需要网络代理，应在该 shell 中设置新机器自己的
`http_proxy`/`https_proxy`；不要照搬开发机的 `127.0.0.1:7897`，除非新机器也确实
运行着那个代理。

## 4. 手工源码路径

用户也可以自己准备依赖，但必须使用 bootstrap plan 给出的精确 commit，不能直接跟随
浮动的 master。direct-only 先安装 NumPilot 并查看 AsNumpy plan：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[bootstrap]'
python -m numpilot bootstrap --plan --build-asnumpy --json
```

若选择 gap/operator lane，再给 plan 增加 `--with-cannbot --accept-cannbot-license`。分别把所选的
AsNumpy/CANNBot checkout 放到 plan 中的 commit，确保官方 origin、clean worktree，再运行相应
apply；下例是包含 CANNBot 的 gap/operator 路径：

```bash
install -d -m 700 "$HOME/.local/state/numpilot-manual"
python -m numpilot bootstrap --apply --build-asnumpy \
  --asnumpy-cache /absolute/path/to/asnumpy \
  --with-cannbot \
  --cannbot-cache /absolute/path/to/cannbot-skills \
  --accept-cannbot-license \
  --receipt "$HOME/.local/state/numpilot-manual/bootstrap.json"
```

上面使用了自定义 receipt，后续新 shell 必须保持：

```bash
export NUMPILOT_BOOTSTRAP_RECEIPT="$HOME/.local/state/numpilot-manual/bootstrap.json"
```

只接受绝对路径；receipt 父目录必须由当前用户拥有、不是符号链接且不可被 group/world 写入。
不设该变量时，NumPilot 只查找自己的默认 XDG receipt，不扫描当前目录。

若 AsNumpy 已由用户完全自行编译安装，只能做基础复验：

```bash
python -m numpilot bootstrap --verify
python -m numpilot setup-skills \
  --root /absolute/path/to/cannbot-skills --check
```

这证明当前安装能够运行，但不能证明已安装二进制来自 NumPilot 固定源码。需要完整来源回执时，
使用 `--asnumpy-cache` 让 bootstrap 负责构建。

## 5. 检查安装结果

```bash
source /absolute/path/printed/by/installer/activate-numpilot.sh
source /absolute/path/printed/by/installer/activate-numpilot.sh  # 幂等复验
python -m numpilot --version
python -m numpilot paths --json
python -m numpilot bootstrap --verify
# 仅当安装时选择了 gap/operator lane：
python -m numpilot bootstrap --verify \
  --with-cannbot --accept-cannbot-license
python -m numpilot doctor --backend npu
```

必须看到：

- AsNumpy `ap.ones((1000, 1000), dtype=float32)` smoke 通过；
- 若选择 gap/operator lane，CANNBot 能从已验证本地 bundle 回答“现在写 Ascend C 算子有哪些能参考的 Skill？”；direct-only 不要求该项；
- `network_requested_by_numpilot=false` 出现在 verify 结果中；
- doctor 检测到 Python、NumPy、CANN、OPP、NPU 和可运行 AsNumpy；
- doctor 没有报告不同 CANN 根或 `devlib`；同根 alias/重复路径至少会明确告警；
- bootstrap receipt 为 `ready=true`，且来源模式为 `built-from-pinned-source`。
- Case01 的静态 capability 分类中，`audit_id` 必须含
  `pinned-source-receipt-v1`，已审计原生 API 不得出现 `UNKNOWN`。

安装 smoke 不等于所有 API 都在 NPU 上执行。任一 runtime identity 或 capability 门失败时，
保存完整非敏感输出并停止，不得把它改写成通过。

## 6. 创建 configured-model 私有配置

克隆 NumPilot 不会自动生成 LLM 配置，也不会复制开发机配置。第一次运行：

```bash
python -m numpilot configure-llm --init
```

命令会在 activation 固定的 config 目录（标准 HOME 默认仍为
`~/.config/numpilot/llm.json`）创建文件，目录 0700、文件 0600，并打印实际路径。手工填写
`base_url`、`auth_token`、精确 `model`、审计标签 `provider` 和固定
`protocol=openai-chat-completions`。响应 model 默认必须精确一致；网关 alias 只有显式加入
`response_model_aliases` 才会接受。GLM 示例是 `model=glm-5.3`、`provider=glm`，不是硬锁。
然后运行：

```bash
python -m numpilot configure-llm --check
python -m numpilot configure-llm --probe
```

也可以先设置 `NUMPILOT_CONFIG_HOME`，让配置生成在指定的仓库外私有目录。Token、代理值、
`llm.json` 和模型响应正文都不得写入仓库或验收报告。

## 7. 迁移前停点

把待测纯 NumPy 文件上传到新机器后，只做静态检查和能力分类；在用户明确说“开始迁移”前，
不要调用模型，也不要生成迁移文件。开始后，Case01 因已知 `matmul` fallback 必须由用户
明确选择：

```bash
# 本轮允许兼容性 fallback；结果不得用于 NPU 加速声明
python -m numpilot migrate /absolute/path/to/01_numpy.py \
  --backend npu --fallback-policy accept \
  --out /tmp/numpilot-case01
```

所有迁移/算子/adapter 源码必须由已配置模型生成；本地 AST 只做审核，不能在模型失败
时替代生成。第一轮只处理 Case01，不自动继续 Case02–10。

`migrate` 只做静态扫描、configured-model 生成和产物审核，不执行原文件或迁移文件。如果用户之后
明确要求“实际运行”，先记录原文件和模型产物 SHA-256，复核产物 AST，再在独立子进程
运行 NumPy 原文件与 AsNumpy 迁移文件。该步只能形成兼容性/数值证据，不得形成 NPU 性能证据。

## 8. 本轮通过标准

- 新机器只手工克隆了 NumPilot；
- AsNumpy 的 receipt 证明其从固定官方源码现场构建，README smoke 通过；
- 若选择 gap/operator lane，CANNBot 固定 commit、许可证接受和 bundle hash 均进入 receipt；
- configured-model 配置由新机器单独创建，probe 成功且没有泄露 Token；
- 用户接受 fallback 后，configured-model 迁移产物通过 AST 审计；若用户另行授权实际运行，
  则独立进程中的数值结果正确，且不把它写成 `migrate` 已执行；
- 报告明确 `strict_npu=false`、`performance_valid=false`；
- 没有发布、暗示或复用任何 NPU 加速比。

任一项不满足都保留现场并报告，不继续处理下一案例。
