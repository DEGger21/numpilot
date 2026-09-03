# 分支治理与比赛提交约定

NumPilot 的治理规范只允许两个长期分支：`main` 和 `develop`。目标是让 `main` 保持可发布、可演示
并可用于比赛交付；`develop` 用于日常开发与集成。远端不得保留长期 feature、release 或个人分支，
但这些服务端状态在有权限的管理员验收前不能视为已经生效。运行中间产物始终留在仓库外状态目录。

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

源码、测试、文档和演示材料的日常修改统一通过 Pull Request 进入 `develop`。合入前必须完成与
风险相匹配的测试；准备发布时，再通过单独的 `develop → main` 提升 Pull Request 进入 `main`。
不得通过直接 push、force push、历史改写或复制文件来绕过两个分支的评审关系。

两个分支都不得包含：

- 私有模型配置、Token、endpoint、代理或原始响应；
- venv、cache、build、dist、egg-info、日志、profiler 数据库、数值数组；
- WIP、AI 交接稿、阶段计划、失败轮次和重复证据；
- 来源或再分发授权不清晰的大体积历史材料。

## 日常开发流程

中央仓库不接受 feature、个人或 release 分支。贡献者应从 fork 的短期分支向中央 `develop`
提交 Pull Request：

```bash
git fetch upstream develop
git switch -c change/<short-topic> upstream/develop

# 修改并验证
python tools/check_release_tree.py
env -u PYTHONPATH -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u NO_PROXY -u http_proxy -u https_proxy -u all_proxy -u no_proxy \
  -u NUMPILOT_LLM_REQUIRE_PROXY \
  python -m pytest -q -p no:cacheprovider
git diff --check

git add <files>
git commit -m "<type>: <summary>"
git push origin change/<short-topic>
```

其中 `upstream` 指中央仓库，`origin` 指贡献者 fork。随后创建
`origin:change/<short-topic> → upstream:develop` Pull Request。合入或放弃后删除 fork 中的短期
分支；中央仓库继续只保留 `main` 与 `develop`。

运行产物必须写到 `python -m numpilot paths` 所示的仓库外目录。公开证据只能从一次完整、
可审计的运行中筛选；加入后需要重新生成并校验 `evidence/MANIFEST.sha256`。

## 提升到 `main`

只有 `develop` 的目标提交通过发布门禁后才能提升。发布维护者创建
`develop → main` Pull Request；`main` 禁止直接 push，本地 merge 结果也不能绕过服务端评审和
流水线门禁。

```bash
git switch develop
git pull --ff-only origin develop
python tools/check_release_tree.py
env -u PYTHONPATH -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u NO_PROXY -u http_proxy -u https_proxy -u all_proxy -u no_proxy \
  -u NUMPILOT_LLM_REQUIRE_PROXY \
  python -m pytest -q -p no:cacheprovider
git diff --check
git status --short
```

Pull Request 合并后，在 `main` 的全新 checkout 对同一提交重跑发布门禁。不得用 rebase、reset、
本地直推或强制推送覆盖已经共享的 `main`/`develop` 历史。

## GitCode 服务端保护基线

仓库内文档不能代替托管平台保护规则。管理员应在 GitCode 项目设置中逐项确认：

- 默认分支为 `main`；
- `main` 与 `develop` 都是保护分支，禁止删除和 force push；
- `main` 禁止直接 push，只允许满足评审与流水线条件的 Pull Request；
- `develop` 至少限制直接 push 权限，并要求日常变更通过 Pull Request；
- `v*` 发布 Tag 受保护、禁止移动或覆盖；
- 启用 Pull Request 审查、合入条件、审计日志与必要通知；
- 在确认默认分支和保护规则生效后，删除中央仓库遗留的 feature/release/个人分支。

GitCode 官方设置入口见[项目设置](https://docs.gitcode.com/v1-docs/docs/repo/setting/)与
[保护分支](https://docs.gitcode.com/v1-docs/docs/repo/code/protected-branch/)。当前仓库尚无可证明由
GitCode/CodeArts 执行的流水线配置，因此在 required checks 真正接线前，发布清单中的测试仍须
保留人工可审计结果，不能把文档约定写成已经机器强制。

服务端验收时执行：

```bash
git ls-remote --symref origin HEAD
git ls-remote --heads origin
```

期望默认 HEAD 指向 `refs/heads/main`，heads 只列出 `main` 与 `develop`。该检查需要有权访问远端
的凭据；本地 remote-tracking refs 已清理并不能证明服务器端分支已删除。

## 冻结、Tag 与打包

日常开发可以继续使用 `0.3.0a1.dev0`。准备最终 RC 时：

1. 在 `develop` 的干净提交上重跑目标 NPU configured-model 主链；
2. 替换仍带 `git_dirty=true` 的开发态证据；
3. 冻结 `pyproject.toml`、`numpilot.__version__`、CHANGELOG 与 SECURITY；
4. 冻结 AsNumpy 内部组件：选择一份实际可取得的 CPython/架构 wheel，记录字节、metadata、
   安装后 native 哈希，并禁止发布安装路径访问浮动上游源码；具体合同见
   [AsNumpy 内部组件锁定说明](ASNUMPY-INTERNAL-COMPONENT-LOCK.zh-CN.md)；
5. 完成完整门禁后，通过受保护的 `develop → main` Pull Request 提升目标提交；
6. 在 `main` 运行 `python tools/check_release_tree.py --require-clean --require-asnumpy-component`；
7. 从 `main` 的全新 checkout 构建 wheel 与 sdist，并做双环境安装冒烟；
8. 生成 release manifest、SBOM 和许可证清单，并把 AsNumpy 组件作为同一 release 的独立附件
   绑定进去；
9. 仅在同一个已验证的 `main` commit 上创建 annotated tag，并正常推送分支与 tag。

完整门禁见 [比赛 RC 发布清单](RELEASE-CHECKLIST-DEMO-V1.md)。直接压缩日常工作目录不是有效
提交方式；比赛 ZIP 应来自已验证 tag 的 `git archive`、wheel/sdist 或干净 checkout。
