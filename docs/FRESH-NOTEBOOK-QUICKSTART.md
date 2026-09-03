# NumPilot 全新 Ascend 环境快速开始

本文从零验证比赛 ordinary 主链：安装固定 AsNumpy → 配置模型 → 静态分析 → 迁移 → 数值
对拍 → strict-NPU resident 评测。运行产物和凭据全部位于仓库外。

> 当前源码版本为 `0.3.0a1.dev0`。比赛冻结时应将下面的 `main` 替换为已验证的不可变 tag，
> 并从干净 checkout 重跑证据；不要把开发态回执称为最终 RC。

## 1. 前置条件

- Linux aarch64、Ascend 910B、驱动、CANN、OPP 与 `npu-smi`；
- CPython 3.10+（包含 `venv`/`pip`）；
- 当前用户独占的运行目录，以及首次准备 NumPilot 基础依赖时可用的 Python 包索引；
- 仅在维护者显式执行 `--asnumpy-source`/`--build-asnumpy` 构建下一版组件时，才需要 Git、
  GCC/G++ 11.2+、Python headers、CMake 3.26+、Make/Ninja 和 `fmt`/`spdlog` CMake 开发包；
- 若选择源码构建或 CANNBot，首次准备相应固定 checkout 时还需访问 GitCode；固定组件模式只需
  从已批准的 Python wheelhouse/包索引准备 NumPilot 基础依赖。

不要复制其他机器的 `.venv`、LLM Token、`llm.json` 或 bootstrap receipt；AsNumpy wheel 只能使用
随 release manifest 绑定并通过 SHA-256 校验的附件。

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

## 3. 固定 AsNumpy 组件安装

```bash
./install.sh --allow-network \
  --asnumpy-component "$PWD/components/asnumpy" \
  --persistent-root /absolute/persistent/numpilot-runtime
source /absolute/persistent/numpilot-runtime/activate-numpilot.sh
```

安装器不会执行 `sudo`、修改 shell 启动文件或保存模型凭据。默认会从当前 release 的
`components/asnumpy/` 读取 manifest、LICENSE、SBOM 和精确 wheel，不访问 AsNumpy 上游；它会：

1. 创建隔离 venv；
2. 展示只读 bootstrap plan；
3. 校验本地 AsNumpy wheel 的文件名、SHA-256、RECORD、wrapper/native 哈希和 CANN/Python/架构画像；
4. 用 `pip --isolated --no-index --no-deps` 安装已校验的本地 wheel，并执行 smoke；
5. 写入仓库外原子 bootstrap 回执；
6. 执行 `bootstrap --verify` 和 NPU doctor。

如果组件附件被放在 release 目录之外，可显式指定它：

```bash
python -m numpilot bootstrap --plan \
  --asnumpy-component /absolute/release/components/asnumpy --json
```

组件缺失或哈希不匹配会失败关闭；不会因为缓存缺失而 `git clone`/`git fetch` 或改走源码构建。
`--build-asnumpy` 只供维护者在隔离构建机准备下一版，不能用于比赛现场安装。

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

若比赛运行时已将配置放在仓库外，可直接指定该文件（不会复制或打印 Token）：

```bash
RUNTIME_ROOT=/absolute/persistent/numpilot-runtime
python -m numpilot --llm-config "$RUNTIME_ROOT/.numpilot-config/llm.json" configure-llm --check
```

若环境要求模型流量经过代理，应同时设置大小写 `HTTP_PROXY` / `HTTPS_PROXY`，并设置
`NUMPILOT_LLM_REQUIRE_PROXY=1` 后再执行 `--probe`。强制模式会在请求前拒绝缺失代理或被
`NO_PROXY` 绕过的 endpoint；只设置 `ALL_PROXY` 不能替代与 endpoint 协议匹配的代理变量。

## 6. 运行 3–5 分钟主演示

```bash
RUNTIME_ROOT=/absolute/persistent/numpilot-runtime
./run_demo.sh --llm-config "$RUNTIME_ROOT/.numpilot-config/llm.json" --quick
```

如果当前 shell 已通过 `activate-numpilot.sh` 导出 `NUMPILOT_LLM_CONFIG_FILE`，也可以省略
`--llm-config`；显式参数只传递私有 JSON 的路径，不会把 Token 放入命令或回执。

默认输入是 `examples/poisson_source_smooth_l1_map.py`，数据类型固定为 `float32`。脚本顺序
执行：

```text
doctor → analyze --require-direct → migrate --engine llm
       → verify-script → benchmark-script → receipt-only summary
```

`--quick` 使用 2 次预热和 7 个样本；默认 20 个样本；`--full` 使用 30 个样本。每轮写入新的
仓库外状态目录，不覆盖既有回执。

提交前先运行只读就绪度审计；它不会执行案例源码或联网，并会明确列出尚未具备的 CodeArts、
华为云、视频和真实数据证据：

```bash
python -m numpilot --llm-config "$RUNTIME_ROOT/.numpilot-config/llm.json" \
  competition-readiness --json
```

## 7. 分步复核

```bash
# Receipts are write-once; use a fresh private directory for every attempt.
RUN=$(mktemp -d /tmp/numpilot-poisson-demo.XXXXXX)
chmod 700 "$RUN"
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
python tools/check_release_tree.py --require-asnumpy-component
sha256sum -c evidence/MANIFEST.sha256
python -m pytest -q -p no:cacheprovider
git diff --check
git status --short
```

最终证据必须绑定干净 commit/tag、目标 AsNumpy/CANN/SoC 身份和完整原始样本。任一身份、
数值、provenance 或性能门失败时停止发布，不得降级为看似成功的 NPU 结论。
