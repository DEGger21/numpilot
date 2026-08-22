# NumPilot 当前进展（2026-08-02 UTC）

> 状态：**进行中，尚未提交，最终回归被用户中断后待重跑。**
>
> 工作目录：`/home/ma-user/work/numpilot`
> 分支：`feat/numpilot-demo`
> 开始本轮时 HEAD：`694e3f9`

## 1. 本轮目标与最新设计决定

本轮先恢复了仓库交接状态，并确认修改前基线为 `106 passed`、工作树干净。

随后重新设计 LLM 凭据配置。最终决定不是把真实 `.env.llm` 放入仓库或要求用户手工创建、`source` 文件，而是：

1. 开源仓库自带完整的配置能力；
2. 用户运行 `python -m numpilot configure-llm`，直接在 NumPilot 内填写 Base URL、Model 和 Auth Token；
3. Token 使用隐藏输入，不进入命令行参数或日志；
4. 框架自动保存到用户配置目录，POSIX 下目录 `0700`、文件 `0600`，并使用临时文件、`fsync`、`os.replace` 原子写入；
5. CI/流水线继续支持 `NUMPILOT_LLM_*` 环境变量，且环境变量逐字段优先；
6. `run_demo.sh`、直接 CLI 和 wizard 统一使用 Python 配置解析器，不再执行 `.env.llm` Shell 文件；
7. 真实 Token 永远不属于开源仓库内容。

本轮早先曾在 `/home/ma-user/work/.env.llm` 创建空模板；该方向被否决后，空文件已删除，里面从未写入真实凭据。

## 2. 当前已落盘的代码改动

### 新增

- `numpilot/llm_config.py`
  - 默认路径：`$XDG_CONFIG_HOME/numpilot/llm.json`，否则 `~/.config/numpilot/llm.json`；
  - 支持 `NUMPILOT_CONFIG_HOME`、`NUMPILOT_LLM_CONFIG_FILE` 覆盖；
  - 环境变量完整时不读取本地文件，避免 CI 被损坏的用户配置阻断；
  - 环境变量不完整时按字段用用户配置补齐；
  - 校验 URL、Token、Model；
  - 配置权限检查、原子保存；
  - 状态输出不包含 Token；
  - 更新配置时 Token 留空可保留旧值。

- `tests/test_llm_config.py`
  - 覆盖 `0700/0600`、逐字段优先级、完整环境绕过损坏文件、损坏/宽权限 fail-closed、Token 脱敏、原子写失败保留旧文件、非 TTY 快速失败、CLI `--token-stdin`、无需 Shell source 自动加载等。

### 修改

- `numpilot/__main__.py`
  - 新增 `configure-llm` 子命令；
  - 新增 `--check`、`--base-url`、`--model`、`--provider`、`--token-stdin`；
  - `migrate/bench-npu --engine llm` 在调用模型前自动加载框架配置；
  - 非交互终端不会等待输入，缺配置时 fail fast。

- `numpilot/wizard.py`
  - 删除旧 `load_env_llm()`；
  - 选择 LLM 时复用统一配置模块；
  - 只有真正交互式选择 LLM 时才允许现场配置；
  - 显式非交互 LLM 缺配置时返回配置错误；
  - UI 中的模型名称改为读取实际配置，不再固定写死 `glm-5.1`。

- `numpilot/engines/llm.py`
  - strict 未配置提示改为 `python -m numpilot configure-llm`；
  - 文档说明真实凭据不入库，用户配置由框架管理。

- `run_demo.sh`
  - 删除 `.env.llm` 查找与 Shell `source`；
  - `--llm` 先调用统一的 `configure-llm --check`，交互 TTY 下可进入配置；
  - 默认解释器由机器专属 `/home/lixing/np310/bin/python` 改为可移植的 `python3`，仍可用 `NUMPILOT_PY` 覆盖；
  - 模型标签不再写死。

- `pyproject.toml`
  - 新增 console script：`numpilot = "numpilot.__main__:main"`。

- `README.md`
  - 改为框架内配置流程；
  - 说明私有配置位置、权限、CI 环境变量和 strict 行为；
  - 移除主要使用说明中的机器专属 Python 路径。

- `tests/test_wizard.py`
  - 移除旧 `.env.llm` 解析测试；
  - 新增 AST 不读取 LLM 配置、非交互 LLM 自动使用用户配置、缺配置快速失败等测试。

## 3. 验证状态（不要误报）

修改前：

```text
python -m pytest -q tests
106 passed
```

第一次相关测试：

```text
31 passed, 1 failed
```

唯一失败是 `test_run_demo_does_not_source_configuration` 把注释中的英文单词 `source` 误判为 Shell 命令；测试随后已改为只检查实际以 `source ` 或 `. ` 开头的 Shell 行。

修正后的相关测试与全量测试尚未获得最终结果：后续验证命令在运行时被用户主动中断。因此，**当前不能宣称新实现已全部通过**。

## 4. 下一位继续时先执行

```bash
cd /home/ma-user/work/numpilot

bash -n run_demo.sh
python -m pytest -q tests/test_llm_config.py tests/test_wizard.py tests/test_migration_ast.py
python -m pytest -q tests
git diff --check
git status --short
```

然后重点复核：

1. `configure-llm` 在真实 TTY 中是否能隐藏 Token 输入并成功更新；
2. 配置 JSON schema/version 与未知字段是否需要更严格的 fail-closed；
3. 所有测试是否通过临时 `NUMPILOT_CONFIG_HOME` 隔离，绝不读取开发机真实凭据；
4. `run_demo.sh --llm` 的非 TTY 行为和退出码是否满足 CI；
5. CLI 配置错误、wizard 面板和异常中是否完全不泄漏 Token；
6. 打包安装后 `numpilot configure-llm` console script 是否可用。

如发现实现问题，优先修复并重跑全量测试，再更新 `docs/AI-HANDOFF.md`。未获得用户明确授权前不要提交 Git。

## 5. 原始主线与参考项目

LLM 配置收尾后，项目原本准备继续“真实缺口算子闭环”。用户提供了两个本地参考：

- `/home/ma-user/work/asnumpyoplogin`：AsNumpy 自定义缺口算子接入 demo；
- `/home/ma-user/work/cannbot-skills`：Ascend C 算子编写与验收技能集合。

旧交接文档仍把 OS 级隔离执行器列为 P0，但用户最新补充明显指向真实缺口算子闭环。恢复该主线前，应先与用户确认纵向目标，或按最新明确指令继续。
