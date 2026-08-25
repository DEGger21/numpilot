# Demo v1 发布清单

本清单适用于 NumPilot `0.2.0a1` / Git tag `v0.2.0a1`。每项都是手工发布门禁；任一必需项
失败都应停止发布、保留日志并修复后从干净源码重新开始。

## 1. 冻结发布身份

- [ ] `pyproject.toml`、`numpilot/__init__.py` 和 CHANGELOG 的版本一致为 `0.2.0a1`；
- [ ] Python 支持范围与测试矩阵一致，Demo v1 最低版本为 Python 3.10；
- [ ] CHANGELOG 的 `0.2.0a1 - 2026-08-25` 内容与待发布 commit 一致；
- [ ] README、架构图、文档链接和安全支持策略不再指向临时开发分支；
- [ ] 确认待发布 commit、固定 AsNumpy wheel、CANN/算子材料和精选 evidence 的身份与
  SHA-256，记录到 release manifest；
- [ ] 完成 secret scan，确认源码、文档、测试、LLM 回执和精选 evidence 不包含凭据或
  不应公开的环境信息。

## 2. 干净源码门禁

- [ ] 所有预期源码、测试和文档均已纳入版本控制；没有意外的未跟踪文件；
- [ ] `git status --short` 为空，HEAD 是计划发布的唯一 commit；
- [ ] `git diff --check` 通过；
- [ ] 从该 commit 创建全新 checkout/worktree，后续构建和安装验证只在该干净副本进行；
- [ ] 完整运行树仍位于仓库外；只提交经过筛选、带 manifest 的公开 evidence。

## 3. 测试与静态验证

- [ ] 在 Python 3.10 的干净环境运行完整测试：
  `python -m pytest -q -p no:cacheprovider`；
- [ ] 运行离线、合成输入的普通脚本扫描/迁移 smoke test，不能使用比赛案例代替自动测试；
- [ ] 验证 `python -m numpilot --help`、`python -m numpilot --version` 及所有公开子命令入口；
- [ ] 核对精选证据：`sha256sum -c evidence/MANIFEST.sha256`；
- [ ] 保存 Python、NumPilot、NumPy、AsNumpy、CANN、设备和操作系统版本到发布记录。

## 4. 从零构建

仓库中已有的 `dist/`、`build/`、`*.egg-info` 或任何旧 wheel/sdist **不得复用、覆盖后冒充
新构建，或纳入发布附件**。构建输出必须进入一个新的空目录，并绑定干净 checkout 的 HEAD。

- [ ] 在干净环境安装锁定/记录版本的构建工具；
- [ ] 使用 `python -m build --outdir <新的空目录>` 同时生成 wheel 和 sdist；
- [ ] 记录构建命令、构建工具版本、HEAD、开始/结束时间和输出文件 SHA-256；
- [ ] 检查 wheel 与 sdist，至少包含：完整 `numpilot` 包、`numpilot/script_migration.py`、
  `LICENSE`、`THIRD_PARTY_NOTICES.md`、console entry point，以及 5 个 AsNumpy integration
  skill data files；sdist 还必须包含全新 Notebook 指南/模板和精选 Case 01 evidence；
- [ ] 确认分发包未包含私有 LLM 配置、完整运行树、缓存、日志、旧 dist、上游源码 checkout、
  未筛选 profiler 产物或本机绝对路径材料；
- [ ] 对 wheel/sdist 执行标准 metadata/包完整性检查，并保存检查输出。

## 5. 全新环境安装验证

- [ ] 创建两个彼此独立的全新 Python 3.10 venv；
- [ ] 第一个 venv 只从本次 wheel 安装，第二个只从本次 sdist 安装；
- [ ] 两个 venv 都验证 `import numpilot`、版本号、`python -m numpilot --help`、console
  `numpilot --help` / `--version` 和合成普通脚本的离线 smoke test；
- [ ] 在目标 Ascend 环境单独验证固定 AsNumpy/CANN 组合；NPU 证据必须绑定源码、迁移产物、
  wheel、profile 和运行环境哈希，且不得把外部 smoke 误写为迁移阶段已执行验证；
- [ ] 在干净 CPython 3.10/aarch64 Demo 环境运行 `numpilot bootstrap --verify`，确认
  `ap.ones((1000, 1000), dtype=ap.float32)`、实际版本、模块路径和 native hash 均通过；
- [ ] 若发布 CANNBot 准备能力，运行
  `numpilot bootstrap --verify --with-cannbot --accept-cannbot-license`，确认本地
  固定 bundle 能回答“现在写 Ascend C 算子有哪些能参考的 Skill？”并列出已校验清单；
- [ ] 若承诺“仅有 CANN + Python”或离线安装，发布并验证完整、哈希绑定的 wheelhouse；否则
  明确说明首次安装 NumPilot 基础依赖需要可用包索引或由用户预先提供本地 wheel；
- [ ] 保存安装日志和验证产物，失败的环境不得用于签署发布结论。

## 6. 许可证、SBOM 与发布附件

- [ ] 为 wheel、sdist、二进制/OPP 和比赛交付包分别生成完整 SBOM 与逐文件许可证清单；
- [ ] 核对 `LICENSE`、`THIRD_PARTY_NOTICES.md`、AsNumpy、CANN/ops-nn/CANNBot 材料及精选
  evidence 内的许可证和再分发限制；
- [ ] 核对根目录 PDF、图、外部材料和模型生成产物的来源与再分发授权；不明确的材料不发布；
- [ ] 为每个发布文件生成 SHA-256，并生成包含文件名、字节数、SHA-256、来源 commit 和
  构建环境的 release manifest；
- [ ] release manifest、SBOM、许可证清单和必要复核日志与发布附件一起保存。

## 7. Push、Tag 与 Release

- [ ] 复核最终 clean HEAD 与所有测试、包、manifest 中记录的 commit 完全一致；
- [ ] 先正常 push 发布分支/目标分支；禁止 force push；
- [ ] 在同一已验证 commit 创建 annotated tag：`git tag -a v0.2.0a1`；
- [ ] push `v0.2.0a1`，再次确认远端 tag 指向已验证 commit；
- [ ] 创建 `NumPilot Demo v1 (0.2.0a1)` release，上传本次新构建产物、SHA-256、SBOM、
  许可证清单、release manifest 和选定 evidence；
- [ ] 从 release 下载产物并复核 SHA-256，最后执行一次全新 venv 安装 smoke test；
- [ ] 记录发布人、复核人、UTC 时间、远端 commit/tag URL 和最终验证结果。

发布过程不应修改、补写或重新封装已经签署哈希的产物。任何变化都必须提升构建序号或版本，
从干净源码重新执行本清单。
