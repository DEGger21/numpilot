# 比赛 RC 发布清单

本清单适用于下一次 NumPilot 比赛 RC；版本和 tag 必须在冻结时从 `pyproject.toml` 唯一确定，
不得继续复用历史 `v0.2.0a1`/`v0.2.0a2` 标签。每项都是手工发布门禁；任一必需项失败都应
停止发布、保留日志并修复后从干净源码重新开始。

日常发布准备统一在 `develop` 完成；只有通过本清单的提交才能通过受保护的
`develop → main` Pull Request 进入始终可发布的 `main`。比赛交付、RC 和 annotated tag 只能
从已复核的 `main` commit 生成。两个长期分支均禁止直接绕过评审、force push 或历史改写。

## 1. 冻结发布身份

- [ ] `pyproject.toml`、`numpilot/__init__.py` 和 CHANGELOG 的版本完全一致，且版本尚未被发布；
- [ ] Python 支持范围与测试矩阵一致，Demo v1 最低版本为 Python 3.10；
- [ ] CHANGELOG 的 Unreleased 内容已冻结到目标版本/date，且与待发布 commit 一致；
- [ ] 英文 `README.md`、中文 `README.zh-CN.md`、双语架构图、文档链接和安全支持策略与
  `main`/`develop` 双分支约定一致；
- [ ] 确认待发布 commit、固定 AsNumpy wheel、CANN/算子材料和精选 evidence 的身份与
  SHA-256，记录到 release manifest；
- [ ] 阅读并冻结 [AsNumpy 内部组件锁定与发布合同](ASNUMPY-INTERNAL-COMPONENT-LOCK.zh-CN.md)：
  release 只消费一份明确的 CPython/架构 wheel，不解析 `latest`、版本范围或浮动 Git 引用；
- [ ] 实际 AsNumpy wheel 本体已放入受控发行附件（或已验证的内部制品库），文件名、字节数、
  SHA-256、METADATA/RECORD、安装后 Python/native 模块哈希均已写入 release manifest；开发态
  receipt 中的历史摘要不能替代可取得的二进制；
- [ ] 已确认发布模式不会因组件缺失而 `git clone`/`git fetch`/下载 AsNumpy 源码或静默升级，
  并保留一份失败关闭测试/日志作为证据；若代码门禁尚未落地，不得勾选本项；
- [ ] 完成 secret scan，确认源码、文档、测试、LLM 回执和精选 evidence 不包含凭据或
  不应公开的环境信息。

## 2. 干净源码门禁

- [ ] 所有预期源码、测试和文档均已纳入版本控制；没有意外的未跟踪文件；
- [ ] `python tools/check_release_tree.py --require-clean --require-asnumpy-component` 通过：组件
  wheel、LICENSE、SBOM 与锁定身份均已校验；只含两份当前公开证据，无 WIP、
  profiler 数据库、数组、日志、缓存、构建目录或大体积历史材料；
- [ ] `git status --short` 为空，HEAD 是计划发布的唯一 commit；
- [ ] `git diff --check` 通过；
- [ ] 从该 commit 创建全新 checkout/worktree，后续构建和安装验证只在该干净副本进行；
- [ ] 完整运行树仍位于仓库外；只提交经过筛选、带 manifest 的公开 evidence。

## 3. 测试与静态验证

> 当前 gap-recovery 测试夹具已在运行时物化到独立临时目录，不再绑定开发机的绝对路径；本地工作树
> 的全量回归可以通过，但这不替代目标 Python 3.10/AArch64/CANN 矩阵的干净 checkout 复核。
> Poisson qualification harness 仍要求发布矩阵中的 CPython 3.10；在该矩阵和第二环境完成复核前，
> 不得把当前开发树标记为 RC。

- [ ] 在 Python 3.10 的干净环境运行完整测试：
  `env -u PYTHONPATH -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u NO_PROXY -u http_proxy -u https_proxy -u all_proxy -u no_proxy -u NUMPILOT_LLM_REQUIRE_PROXY python -m pytest -q -p no:cacheprovider`；
- [ ] 运行离线、合成输入的普通脚本扫描/迁移 smoke test，不能使用比赛案例代替自动测试；
- [ ] 运行 `benchmark-script` 的 parent/worker/CLI 全部 fail-closed 测试，并在目标 910B4 上对
  同一 configured-GLM artifact 完成至少 20 样本的 opt-in 真机验收；
- [ ] 验证 `python -m numpilot --help`、`python -m numpilot --version` 及所有公开子命令入口；
- [ ] 核对精选证据：`sha256sum -c evidence/MANIFEST.sha256`；
- [ ] 用 PowerPoint/WPS/LibreOffice 分别打开 `docs/numpilot-zh.pptx` 与 `docs/numpilot-en.pptx`，
  确认无字体替换导致的溢出，实线/虚线与当前/目标状态语义一致，并与 README 双语 SVG 的能力
  边界一致；
- [ ] 在 GitCode 桌面端和移动端实际预览 `docs/numpilot-zh-v6.svg` 与 `docs/numpilot-en-v6.svg`，
  确认内嵌 Logo、轮廓文字、README 缩放展示和高清图链接均能正常渲染；
- [ ] 保存 Python、NumPilot、NumPy、AsNumpy、CANN、设备和操作系统版本到发布记录。

## 4. 从零构建

仓库中已有的 `dist/`、`build/`、`*.egg-info` 或任何旧 wheel/sdist **不得复用、覆盖后冒充
新构建，或纳入发布附件**。构建输出必须进入一个新的空目录，并绑定干净 checkout 的 HEAD。

- [ ] 在干净环境安装锁定/记录版本的构建工具；
- [ ] 使用 `python -m build --outdir <新的空目录>` 同时生成 wheel 和 sdist；
- [ ] 记录构建命令、构建工具版本、HEAD、开始/结束时间和输出文件 SHA-256；
- [ ] 将本次实际交付的 AsNumpy 内部 wheel 放入同一 release 的 `components/asnumpy/`（或登记
  不可变内部制品地址），并在主 release manifest 中绑定组件 artifact hash；不能只记录
  `1929a3ba…` commit/tree；
- [ ] 检查 wheel 与 sdist，至少包含：完整 `numpilot` 包、ordinary migration/verification/
  benchmark 父层与两个私有 worker、`LICENSE`、`THIRD_PARTY_NOTICES.md`、console entry point、
  CodeArts 4 个 agent + migration gate Skill，以及全部 AsNumpy integration Skill data files；
  sdist 还必须包含双语 README/安装指南/架构 PPT/SVG、`run_demo.sh`、competition example、
  Notebook 指南/模板和精选 evidence；
- [ ] 确认分发包未包含私有 LLM 配置、完整运行树、缓存、日志、旧 dist、上游源码 checkout、
  未筛选 profiler 产物或本机绝对路径材料；
- [ ] 对 wheel/sdist 执行标准 metadata/包完整性检查，并保存检查输出。

在干净源码树、且构建输出位于仓库外时，用标准库审计工具生成并立即复核发布清单：

```bash
python tools/release_artifact_manifest.py \
  /path/to/numpilot-<version>-py3-none-any.whl \
  /path/to/numpilot-<version>.tar.gz \
  --repo-root . \
  --require-clean \
  --require-asnumpy-component \
  --output /path/to/release-manifest.json
python tools/release_artifact_manifest.py \
  --verify /path/to/release-manifest.json \
  --artifact-dir /path/to \
  --require-asnumpy-component
```

该清单逐文件绑定归档内容、包元数据、发布文件哈希和 Git 溯源，但不替代完整的标准 SBOM
和逐文件许可证复核。

## 5. 全新环境安装验证

- [ ] 创建两个彼此独立的全新 Python 3.10 venv；
- [ ] 第一个 venv 只从本次 wheel 安装，第二个只从本次 sdist 安装；
- [ ] 两个 venv 都验证 `import numpilot`、版本号、`python -m numpilot --help`、console
  `numpilot --help` / `--version` 和合成普通脚本的离线 smoke test；
- [ ] 在目标 Ascend 环境单独验证固定 AsNumpy/CANN 组合；NPU 证据必须绑定源码、迁移产物、
  wheel、profile 和运行环境哈希，且不得把外部 smoke 误写为迁移阶段已执行验证；
- [ ] 在第二个 Ascend/华为云环境从发布 wheel 运行唯一主链
  `analyze → configured GLM migrate → verify-script → benchmark-script`，保存原始回执；
- [ ] 在干净 CPython 3.10/aarch64 Demo 环境运行 `numpilot bootstrap --verify`，确认
  `ap.ones((1000, 1000), dtype=ap.float32)`、实际版本、模块路径和 native hash 均通过；
- [ ] 在无上游网络、空 AsNumpy 源码缓存的隔离环境中，仅用发行附件的锁定 wheel 完成安装和
  README smoke；确认组件缺失/哈希篡改时命令失败关闭，且不会回退到源码获取；
- [ ] 在第二个干净环境再次核对 wheel 字节 hash、`asnumpy/__init__.py`、native `_core`、
  Python ABI/SoC/CANN 画像和 receipt 绑定；
- [ ] 若发布 CANNBot 准备能力，运行
  `numpilot bootstrap --verify --with-cannbot --accept-cannbot-license`，确认本地
  固定 bundle 能回答“现在写 Ascend C 算子有哪些能参考的 Skill？”并列出已校验清单；
- [ ] 若承诺“仅有 CANN + Python”或离线安装，发布并验证完整、哈希绑定的 wheelhouse；否则
  明确说明首次安装 NumPilot 基础依赖需要可用包索引或由用户预先提供本地 wheel；
- [ ] 保存安装日志和验证产物，失败的环境不得用于签署发布结论。

## 6. 许可证、SBOM 与发布附件

- [ ] 为 wheel、sdist、二进制/OPP 和比赛交付包分别生成完整 SBOM 与逐文件许可证清单；
- [ ] AsNumpy 作为独立内部组件单列其 Apache-2.0 LICENSE、来源 commit/tree、artifact/module/
  native hashes 和再分发范围；确认 CANN/OPP/ACLNN/CANNBot/ops-nn 的许可证没有被合并或误写成
  NumPilot 自有代码；
- [ ] 核对 `LICENSE`、`THIRD_PARTY_NOTICES.md`、AsNumpy、CANN/ops-nn/CANNBot 材料及精选
  evidence 内的许可证和再分发限制；
- [ ] 核对根目录 PDF、图、外部材料和模型生成产物的来源与再分发授权；不明确的材料不发布；
- [ ] 为每个发布文件生成 SHA-256，并生成包含文件名、字节数、SHA-256、来源 commit 和
  构建环境的 release manifest；
- [ ] release manifest、SBOM、许可证清单和必要复核日志与发布附件一起保存。

## 7. Push、Tag 与 Release

- [ ] 确认已完成集成验证的提交通过 Pull Request 进入受保护的 `develop`；
- [ ] 创建并完成满足审批与流水线条件的 `develop → main` Pull Request，不得直接 push、复制提交、
  rebase 已共享历史或跳过评审门禁；
- [ ] 在 `main` 的干净 checkout 重跑发布门禁，复核 HEAD 与所有测试、包、manifest 中记录的
  commit 完全一致；
- [ ] 确认远端 `main` commit 与 Pull Request 中复核的提交一致；禁止补推或 force push；
- [ ] 仅在该已验证的 `main` commit 创建与冻结版本一致的 annotated tag；
- [ ] push 该 tag，再次确认远端 tag 指向已验证 commit；
- [ ] 创建与冻结版本一致的比赛 RC release，上传本次新构建产物、SHA-256、SBOM、
  许可证清单、release manifest 和选定 evidence；
- [ ] 从 release 下载产物并复核 SHA-256，最后执行一次全新 venv 安装 smoke test；
- [ ] 记录发布人、复核人、UTC 时间、远端 commit/tag URL 和最终验证结果。

发布过程不应修改、补写或重新封装已经签署哈希的产物。任何变化都必须提升构建序号或版本，
从干净源码重新执行本清单。
