# Ordinary script 数值验证

`verify-script` 是 ordinary `migrate` 之后的独立、显式执行步骤。它不修改原迁移报告，
而是将原源码、`script-migration-report.json`、同目录 `migrated.py`、两个执行环境和明确命名的
输出变量绑定到一份 write-once 数值验证回执。

迁移本身仍然不执行用户代码，迁移报告继续如实保持
`execution_performed=false`、`numerical_verification=not_run`。只有用户另外运行
`verify-script`，才会执行两侧脚本。

## 最短命令

```bash
python -m numpilot verify-script /absolute/path/to/source.py \
  --migration-report /absolute/path/to/run/script-migration-report.json \
  --output result \
  --out /absolute/path/to/run/script-verification-report.json
```

每个 `--output NAME` 都必须是原脚本和迁移脚本执行后存在的全局变量；可重复传入多个变量。
当前合同只接受 numeric/bool scalar 或 ndarray leaf，不接受 object、字符串、任意容器或 pickle。
stdout/stderr 会被丢弃，打印文本不能伪装成验证输出。

如果两侧依赖位于不同 Python 环境，可显式指定解释器：

```bash
python -m numpilot verify-script source.py \
  --migration-report run/script-migration-report.json \
  --output result --out run/script-verification-report.json \
  --source-python /absolute/cpu-venv/bin/python \
  --migrated-python /absolute/npu-venv/bin/python
```

worker 使用 `python -I`；因此不会从 `PYTHONPATH` 或用户 site 猜测运行时。目标 NumPy/AsNumPy
必须已经安装在所选解释器环境中。NumPilot 会保留 `.venv/bin/python` 的调用路径，不能把它
解析成 base Python 后悄悄丢失虚拟环境。

## 准入与执行合同

执行前必须同时满足：

- report 是 schema 2 ordinary script 迁移报告；
- status 是 `migrated_strict_npu`、`migrated_scoped_npu` 或
  `migrated_with_fallback`；
- `ast_audit=passed`，且迁移阶段没有声称已经执行、验证或产生性能结论；
- report 顶层和 `analysis.scan` 中的 source path/SHA-256 与当前原文件一致；
- report 的 `migrated_sha256` 与其同目录 `migrated.py` 一致；
- 输出名唯一、是 Python identifier，timeout 和输出大小在有界范围内；
- 回执路径不存在，且不与任一输入别名冲突。

随后 NumPilot：

1. 把已经校验的 source、migrated 和 worker 字节复制到私有临时目录；
2. 在两个独立 PID 和进程组中运行 source/migrated worker；
3. 为 worker 使用私有 HOME/XDG 目录，移除模型 Token、API key、密码、云 AK/SK 和 SSH agent
   等明显敏感环境变量；
4. 丢弃 stdout/stderr，通过 JSON manifest 与 `allow_pickle=False` 的 `.npy` 传递有界输出；
5. 在父进程从已校验字节加载数组，比较变量名、shape、dtype、NaN/Inf 模式和数值；
6. 每个浮点 dtype 使用 `atol + rtol * abs(reference)`，整数和布尔要求完全相等；
7. 每侧执行后重新哈希 source/report/migrated，发现漂移立即失败；
8. 原子、独占发布权限 `0600` 的 JSON 回执，绝不覆盖已有文件。

回执记录两个 PID、worker SHA-256、Python invocation/realpath/prefix、解释器文件大小与
SHA-256、NumPy/AsNumPy/
`asnumpy._core` 的版本、路径和有界文件哈希，以及逐输出 comparison。它不保存输出值、stdout、
stderr、异常消息、Token 或 traceback。

## 退出状态

- `0`：所有明确命名的输出数值一致，`status=passed`；
- `1`：执行、输出合同或数值比较失败，且 `status=failed` 回执已经发布；
- `2`：参数、解释器、source/report/artifact 输入绑定或 write-once 输出边界尚未准入，因此不
  发布缺少完整哈希绑定的回执。

所有失败使用稳定 code。`source_hash_mismatch`、`migrated_hash_mismatch`、
`source_path_mismatch` 和 `invalid_report` 属于执行前准入失败，不发布回执；准入完成后的
`missing_output`、`unsupported_dtype`、`script_exception`、`execution_timeout` 和
`numerical_mismatch` 会发布 hash-bound `status=failed` 回执。

## 不能外推的结论

子进程隔离只提供状态分离、独立 PID 和可终止 timeout，**不是 OS sandbox**。待验证代码仍能
访问当前用户本来可访问的文件、设备和网络；只能对受信源码和迁移产物显式运行。当前 worker
在私有临时工作目录执行，因此只适合自包含脚本或使用明确绝对路径的数据；外部数据还没有进入
本回执的 hash contract。

本命令只证明明确命名输出在回执记录的两个解释器/runtime 中数值一致。所有回执固定：

```text
strict_npu_verified=false
performance_valid=false
os_sandbox=false
```

即使 migration report 来自 strict/scoped 路径，`verify-script` 也不会仅凭数值一致宣称 native
NPU、封死 host fallback、kernel profiler 或加速比。此类结论仍需 runtime identity、调用 profile、
deny-fallback/NPU 执行门和独立 benchmark；现有 ops-nn MatMul 真机纵切即使用了更强的定向
资格化 harness。
