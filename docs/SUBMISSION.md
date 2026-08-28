# 比赛提交分支约定

`release/competition-submission` 是 NumPilot 比赛打包与后续完善的唯一发布分支。它从已验证的
framework-hardening 提交建立，保留完整产品源码与测试，同时把开发历史和运行中间产物留在
其他分支或仓库外状态目录。

## 分支职责

发布分支只接受以下内容：

- NumPilot 产品源码、公开 CLI、安装器与 CodeArts 角色/Skill；
- ordinary 主链、当前示例和可复现测试；
- 面向评委与用户的当前文档、架构源文件和演示 PPT；
- 经过筛选、带 SHA-256 清单且宣传边界明确的当前证据；
- 发布、审计、摘要渲染等通用工具。

以下内容不得进入：

- 私有模型配置、Token、endpoint、代理或原始响应；
- venv、cache、build、dist、egg-info、日志、profiler 数据库、数值数组；
- WIP、AI 交接稿、阶段计划、失败轮次和重复证据；
- 来源或再分发授权不清晰的大体积历史材料。

## 日常工作流

```bash
git switch release/competition-submission
git pull --ff-only

# 修改并验证
python tools/check_release_tree.py
python -m pytest -q -p no:cacheprovider
git diff --check

# 常规提交与推送；禁止 force push
git push
```

运行产物必须写到 `python -m numpilot paths` 所示的仓库外目录。公开证据只能从一次完整、
可审计的运行中筛选，加入后重新生成 `evidence/MANIFEST.sha256`。

## 冻结与打包

开发分支可以持续使用 `0.3.0a1.dev0`。准备最终 RC 时才执行：

1. 在干净提交上重跑目标 NPU 的 configured-model 主链；
2. 替换仍带 `git_dirty=true` 的开发态证据；
3. 冻结 `pyproject.toml`、`numpilot.__version__`、CHANGELOG 与 SECURITY；
4. 运行 `python tools/check_release_tree.py --require-clean`；
5. 从全新 checkout 构建 wheel 与 sdist，并做双环境安装冒烟；
6. 生成 release manifest、SBOM 和许可证清单；
7. 在已验证 commit 上创建 annotated tag，正常 push，禁止改写历史。

完整门禁见 [比赛 RC 发布清单](RELEASE-CHECKLIST-DEMO-V1.md)。直接压缩日常工作目录不是有效
提交方式；比赛 ZIP 应来自 `git archive`、wheel/sdist 或干净 checkout。
