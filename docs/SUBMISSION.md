# 分支治理与比赛提交约定

NumPilot 只维护两个长期分支：`main` 和 `develop`。`main` 始终保持可发布、可演示和可用于
比赛交付；`develop` 用于日常开发与集成。远端不保留长期 feature、release 或个人分支，
运行中间产物则始终留在仓库外状态目录。

## 分支职责

### `main`：可发布与比赛交付

`main` 只接收已经在 `develop` 完成集成验证的内容，包括：

- NumPilot 产品源码、公开 CLI、安装器与 CodeArts 角色/Skill；
- ordinary 主链、当前示例和可复现测试；
- 面向评委与用户的当前文档、架构源文件和演示 PPT；
- 经过筛选、带 SHA-256 清单且宣传边界明确的当前证据；
- 发布、审计、摘要渲染等通用工具。

任何时候检出 `main`，都应能通过发布树检查和完整测试。比赛 ZIP、wheel、sdist、RC 与正式
tag 必须从 `main` 的干净提交生成，不得从开发工作树或其他分支临时打包。

### `develop`：日常开发与集成

源码、测试、文档和演示材料的日常修改统一进入 `develop`。合入前必须完成与风险相匹配的
测试；准备发布时，再把同一批已验证提交通过正常 merge 或 fast-forward 提升到 `main`。
不得通过 force push、历史改写或复制文件来同步两个分支。

两个分支都不得包含：

- 私有模型配置、Token、endpoint、代理或原始响应；
- venv、cache、build、dist、egg-info、日志、profiler 数据库、数值数组；
- WIP、AI 交接稿、阶段计划、失败轮次和重复证据；
- 来源或再分发授权不清晰的大体积历史材料。

## 日常开发流程

```bash
git switch develop
git pull --ff-only origin develop

# 修改并验证
python tools/check_release_tree.py
python -m pytest -q -p no:cacheprovider
git diff --check

git add <files>
git commit -m "<type>: <summary>"
git push origin develop
```

运行产物必须写到 `python -m numpilot paths` 所示的仓库外目录。公开证据只能从一次完整、
可审计的运行中筛选；加入后需要重新生成并校验 `evidence/MANIFEST.sha256`。

## 提升到 `main`

只有 `develop` 的目标提交通过发布门禁后才能提升。历史保持线性时优先 fast-forward；需要
保留评审边界时可使用正常 merge。两种方式都不得 force push。

```bash
git switch develop
git pull --ff-only origin develop
python tools/check_release_tree.py
python -m pytest -q -p no:cacheprovider
git diff --check
git status --short

git switch main
git pull --ff-only origin main
git merge --ff-only develop

# 在 main 上复核同一门禁，然后正常推送
python tools/check_release_tree.py
python -m pytest -q -p no:cacheprovider
git diff --check
git status --short
git push origin main
```

若无法 fast-forward，应先确认分歧来源，再通过代码评审执行正常 merge；不得用 rebase、reset
或强制推送覆盖已经共享的 `main`/`develop` 历史。

## 冻结、Tag 与打包

日常开发可以继续使用 `0.3.0a1.dev0`。准备最终 RC 时：

1. 在 `develop` 的干净提交上重跑目标 NPU configured-model 主链；
2. 替换仍带 `git_dirty=true` 的开发态证据；
3. 冻结 `pyproject.toml`、`numpilot.__version__`、CHANGELOG 与 SECURITY；
4. 完成完整门禁后，以正常 merge/fast-forward 将目标提交提升到 `main`；
5. 在 `main` 运行 `python tools/check_release_tree.py --require-clean`；
6. 从 `main` 的全新 checkout 构建 wheel 与 sdist，并做双环境安装冒烟；
7. 生成 release manifest、SBOM 和许可证清单；
8. 仅在同一个已验证的 `main` commit 上创建 annotated tag，并正常推送分支与 tag。

完整门禁见 [比赛 RC 发布清单](RELEASE-CHECKLIST-DEMO-V1.md)。直接压缩日常工作目录不是有效
提交方式；比赛 ZIP 应来自已验证 tag 的 `git archive`、wheel/sdist 或干净 checkout。
