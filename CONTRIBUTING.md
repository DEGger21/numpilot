# 为 NumPilot 做贡献

感谢你帮助改进 NumPilot。项目把“能运行”和“能可信地声明”分开：代码、测试、来源身份和
原始证据必须互相对应。

## 开发环境

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
python -m build
```

离线开发使用 `--backend shim`；不要在没有真实 NPU、capability provenance 和数值门禁的
情况下提交性能结论。运行产物默认写到仓库外，执行 `python -m numpilot paths` 可查看位置。

## 提交要求

- 新功能或修复应包含回归测试；
- 不提交 Token、私有 endpoint、`.env`、虚拟环境、编译目录或未经筛选的运行树；
- 第三方代码、模板、数据和生成算子必须记录来源、固定版本和许可证；
- 性能数据必须带原始样本、硬件/软件身份、计时口径、数值门禁和 SHA-256；
- 修改 CLI 或证据 schema 时同步更新 README、相关文档与兼容性说明；
- 提交前运行 `python -m pytest -q`、`python -m build` 和 `git diff --check`。

## 变更说明

请说明问题、设计边界、验证命令和结果。涉及 NPU 的变更还应区分 shim、静态审计、真实
NPU 运行、profiler 与可发布性能结论，避免把其中一种证据扩大为另一种。

提交贡献即表示你有权提交相关内容，并同意该贡献按仓库顶层 [Apache-2.0](LICENSE) 发布；
已有单独许可证标记的第三方材料除外。
