# NumPilot 全新 Ascend 环境快速开始

本文从零验证比赛 ordinary 主链：安装固定 AsNumpy → 配置模型 → 静态分析 → 迁移 → 数值
对拍 → strict-NPU resident 评测。运行产物和凭据全部位于仓库外。

> 当前源码版本为 `0.3.0a1.dev0`。比赛冻结时应将下面的 `main` 替换为已验证的不可变 tag，
> 并从干净 checkout 重跑证据；不要把开发态回执称为最终 RC。

## 1. 前置条件

- Linux aarch64、Ascend 910B、驱动、CANN 与 `npu-smi`；
- CPython 3.10+（包含 `venv` 与 Python headers）；
- Git、GCC/G++ 11.2+、CMake 3.26+，以及 Make 或 Ninja；
- `fmt`、`spdlog` 的 CMake 开发包；
- 首次准备依赖时可访问 GitCode 与 Python 包索引。

不要复制其他机器的 `.venv`、LLM Token、`llm.json`、bootstrap receipt 或 AsNumpy wheel。

## 2. 获取可发布分支

```bash
git clone --branch main --depth 1 \
  https://gitcode.com/li-xingHIT/numpilot.git
cd numpilot
git status --short --branch
git rev-parse HEAD
```

`main` 始终保持可发布；日常集成通过 Pull Request 进入 `develop`，发布提升再通过受保护的
`develop → main` Pull Request 完成，两个分支均禁止 force push。冻结比赛版本后应改用
`git clone --branch <immutable-tag>`，该 tag 必须创建自已验证的 `main` commit。工作树必须干净。

## 3. 固定来源安装

```bash
./install.sh --allow-network \
  --persistent-root /absolute/persistent/numpilot-runtime
source /absolute/persistent/numpilot-runtime/activate-numpilot.sh
```

安装器不会执行 `sudo`、修改 shell 启动文件或保存模型凭据。它会：

1. 创建隔离 venv；
2. 展示只读 bootstrap plan；
3. 从 NumPilot 固定的官方来源构建 AsNumpy；
4. 校验 wheel RECORD、wrapper/native 哈希与 smoke；
5. 写入仓库外原子 bootstrap 回执；
6. 执行 `bootstrap --verify` 和 NPU doctor。

只有 gap/operator lane 需要 CANNBot：

```bash
./install.sh --allow-network \
  --persistent-root /absolute/persistent/numpilot-runtime \
  --with-cannbot --accept-cannbot-license
```

该选项必须在阅读固定版本许可证后显式选择；ordinary direct lane 不依赖 CANNBot。

## 4. 验证运行时身份

```bash
python -m numpilot --version
python -m numpilot paths --json
python -m numpilot bootstrap --verify
python -m numpilot doctor --backend npu
```

必须确认 AsNumpy smoke、bootstrap 来源回执、CANN/SoC/Driver、模块路径与 wrapper/native 哈希
全部通过。安装成功只证明环境可用，不代表任意 NumPy API 都是 NPU 原生实现。

## 5. 配置真实模型

```bash
python -m numpilot configure-llm --init
# 编辑命令打印的私有 llm.json
python -m numpilot configure-llm --check
python -m numpilot configure-llm --probe
```

配置采用 OpenAI-compatible Chat Completions 最小合同，至少包含 `base_url`、`auth_token`、
精确 `model`、`provider` 审计标签和 `protocol=openai-chat-completions`。默认路径通常为
`~/.config/numpilot/llm.json`，目录权限应为 `0700`、文件权限为 `0600`。Token、代理、私有
endpoint 和模型响应正文不得进入仓库或证据包。

若环境要求模型流量经过代理，应同时设置大小写 `HTTP_PROXY` / `HTTPS_PROXY`，并设置
`NUMPILOT_LLM_REQUIRE_PROXY=1` 后再执行 `--probe`。强制模式会在请求前拒绝缺失代理或被
`NO_PROXY` 绕过的 endpoint；只设置 `ALL_PROXY` 不能替代与 endpoint 协议匹配的代理变量。

## 6. 运行 3–5 分钟主演示

```bash
./run_demo.sh --quick
```

默认输入是 `examples/poisson_source_smooth_l1_map.py`，数据类型固定为 `float32`。脚本顺序
执行：

```text
doctor → analyze --require-direct → migrate --engine llm
       → verify-script → benchmark-script → receipt-only summary
```

`--quick` 使用 2 次预热和 7 个样本；默认 20 个样本；`--full` 使用 30 个样本。每轮写入新的
仓库外状态目录，不覆盖既有回执。

## 7. 分步复核

```bash
RUN=/tmp/numpilot-poisson-demo
SRC=examples/poisson_source_smooth_l1_map.py

python -m numpilot analyze "$SRC" \
  --require-direct --out "$RUN/qualified-analysis.json"

python -m numpilot migrate "$SRC" \
  --mode script --engine llm --backend npu --fallback-policy reject \
  --out "$RUN/migration"

python -m numpilot verify-script "$SRC" \
  --migration-report "$RUN/migration/script-migration-report.json" \
  --output result --out "$RUN/script-verification-report.json"

python -m numpilot benchmark-script "$SRC" \
  --migration-report "$RUN/migration/script-migration-report.json" \
  --output result --warmup 3 --samples 20 \
  --out "$RUN/script-benchmark-report.json"
```

只有 benchmark 回执中的 `performance_valid=true` 才允许发布 speedup。主结论是设备驻留的
重复计算区间；冷启动、一次输入准备和最终 D2H 必须单独报告。

## 8. 提交前门禁

```bash
python tools/check_release_tree.py
sha256sum -c evidence/MANIFEST.sha256
python -m pytest -q -p no:cacheprovider
git diff --check
git status --short
```

最终证据必须绑定干净 commit/tag、目标 AsNumpy/CANN/SoC 身份和完整原始样本。任一身份、
数值、provenance 或性能门失败时停止发布，不得降级为看似成功的 NPU 结论。
