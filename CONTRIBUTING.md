# 为 NumPilot 做贡献

感谢你帮助改进 NumPilot。项目把“能运行”和“能可信地声明”分开：代码、测试、来源身份和
原始证据必须互相对应。

## 开发环境

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
env -u PYTHONPATH python -m pytest -q -p no:cacheprovider \
  tests/test_install_script.py tests/test_release_tree.py \
  tests/test_release_artifact_manifest.py tests/test_notebook_quickstart.py
```

完整测试与 `python -m build` 是固定、干净发布环境中的门禁。先清理 CANN 注入的 `PYTHONPATH` 及
大小写代理变量，并使用发布矩阵指定的 Python；不能把任意 Python 3.10+ 环境中的结果等同于发布
资格化。

离线开发使用 `--backend shim`；不要在没有真实 NPU、capability provenance 和数值门禁的
情况下提交性能结论。运行产物默认写到仓库外，执行 `python -m numpilot paths` 可查看位置。

完整的 CPU/shim 与 NPU 安装步骤见
[中文安装指南](docs/installation.zh-CN.md) / [English installation guide](docs/installation.md)。

## 分支与 Pull Request

仓库规范只允许 `main` 和 `develop` 两个长期分支。`main` 是发布分支，`develop` 是日常集成分支；
两者必须在 GitCode 服务端配置保护，实际设置以管理员验收为准。

- 外部贡献请在 fork 中创建短期分支，并向中央仓库 `develop` 提交 Pull Request；
- 不向中央仓库推送 feature、个人或 release 分支；
- 发布提升通过 `develop → main` Pull Request 完成，禁止绕过评审直接推送 `main`；
- 两个长期分支均禁止 force push、历史改写和删除；
- Tag 与发布制品只能来自已经通过发布门禁的 `main` 提交。

完整保护规则、服务端验收项与发布流程见 [分支治理与比赛提交约定](docs/SUBMISSION.md)。

## 提交要求

- 新功能或修复应包含回归测试；
- 不提交 Token、私有 endpoint、`.env`、虚拟环境、编译目录或未经筛选的运行树；
- 第三方代码、模板、数据和生成算子必须记录来源、固定版本和许可证；
- 性能数据必须带原始样本、硬件/软件身份、计时口径、数值门禁和 SHA-256；
- 修改 CLI 或证据 schema 时同步更新 README、相关文档与兼容性说明；
- 提交前在对应测试矩阵运行完整测试、`python -m build` 和 `git diff --check`，并如实报告失败项。

## 变更说明

请说明问题、设计边界、验证命令和结果。涉及 NPU 的变更还应区分 shim、静态审计、真实
NPU 运行、profiler 与可发布性能结论，避免把其中一种证据扩大为另一种。

提交贡献即表示你有权提交相关内容，并同意该贡献按仓库顶层 [Apache-2.0](LICENSE) 发布；
已有单独许可证标记的第三方材料除外。
