# ops-nn MatMul 缺口迁移案例

本案例从纯 NumPy 脚本出发，选择 API 审计报告中仍走宿主 NumPy 的
`numpy.matmul` 作为缺口，经 NumPilot 解析官方候选、接入 AsNumPy、限定能力范围并由已配置
模型完成迁移，最后在 Ascend 910B4 上封死宿主回退进行数值对拍。

> **2026-08-28 provenance 更正：**下述历史执行证据仍能证明当时的 `_core.matmul`
> 原生调用、host-fallback guard 与数值结果，但旧 `official-resolution.json` 选择并绑定的是
> `libopapi_nn.so`，而独立 ELF/loader 审查确认 `_core` 实际链接和解析的是
> `libopapi.so`。因此旧 qualification 不能证明“所选 DSO 被复用”，不得继续作为新 Adapter
> 的资格化依据。替代的 discovery-only 收据是
> [`official-resolution-linkage-v2.json`](official-resolution-linkage-v2.json)；它尚未取得新的
> Adapter/build/NPU qualification。当前状态见
> [`docs/PROGRESS-20260828.md`](../../docs/PROGRESS-20260828.md)。

## 案例与迁移结果

- 原始脚本：[`examples/ops_nn_matmul_inference.py`](../../examples/ops_nn_matmul_inference.py)，
  使用两个 `float32` 二维矩阵并调用 `np.matmul`；
- 缺口证据：[`pre-integration-gap-v2/gap-request.json`](pre-integration-gap-v2/gap-request.json)，
  在接入前将 `matmul` 判为 `CPU_FALLBACK`，拒绝 fallback 后生成一个 GapSpec；
- 迁移产物：[`final-migration/migrated.py`](final-migration/migrated.py)，由已配置的
  `glm-5.3` 实际远程调用生成，使用 `asnumpy.full` 和 `asnumpy.matmul`；
- 迁移报告：[`final-migration/script-migration-report.json`](final-migration/script-migration-report.json)，
  状态为 `migrated_scoped_npu`，AST 审计通过；该报告本身不宣称已执行或有性能结论；
- 独立执行证据：[`execution-evidence.json`](execution-evidence.json)，原始 NumPy 与迁移脚本
  均被真实执行，原生 MatMul 调用计数为 1，宿主回退调用计数为 0，最大绝对误差为
  `6.984919309616089e-10`。

原始工作盘的 API 审计报告不属于本仓库；可公开、source-bound 的缺口判定保存在
[`pre-integration-gap-v2/script-migration-report.json`](pre-integration-gap-v2/script-migration-report.json)，
对应 GapSpec 保存在
[`pre-integration-gap-v2/gap-request.json`](pre-integration-gap-v2/gap-request.json)。

## ops-nn 复用方式

解析器先检查已安装 CANN，再检查固定版本 ops-nn，最后才允许配置模型生成新 kernel。
本机 CANN 9.0.0 已包含 `aclnnMatmulGetWorkspaceSize`/`aclnnMatmul`，因此本案例采用
`installed_cann_adapter_candidate`：

- ops-nn 来源固定为 `https://gitcode.com/cann/ops-nn.git` 的 `v9.0.0`，commit
  `0413e303e6cd1986b592c3e039f2579ac0f10b62`，算子目录 `matmul/mat_mul_v3`；
- ops-nn 中 API 头文件与已安装 `aclnn_matmul.h` 的 SHA-256 同为
  `170e9a38439f9ee7f1f6b36b4be63858dc8162d949a4840c78956beaed13eba1`；
- AsNumPy 公开 `matmul` 对本案例范围直接调用已有 `_core.matmul`/ACLNN 路径，没有复制或
  重建整套 ops-nn，也没有修改系统 OPP；
- `cubeMathType` 使用 `0`（KEEP_DTYPE），避免 `float32` 计算被降为 `float16`。

历史候选与已安装库/符号哈希见 [`official-resolution.json`](official-resolution.json)，历史资格化
来源、构建和真机门禁见 [`qualification-receipt.json`](qualification-receipt.json)；两者受上方
2026-08-28 provenance 更正约束。当前重新解析的 `libopapi.so` 候选见
[`official-resolution-linkage-v2.json`](official-resolution-linkage-v2.json)。

## 已验证范围

NumPilot 只对精确运行时身份开放 `matmul` 的以下范围：两个位置参数、二维、C 连续、非空、
`float32`，且内维相等。1-D、批处理/广播、混合 dtype、空张量和关键字变体仍保持未审计，
不会因本案例被泛化为 NPU 原生能力。

资格化包括 AsNumPy MatMul 定向测试 `9 passed`，以及随机 `512 x 512 float32` 真机对拍，
最大绝对误差 `8.940696716308594e-08`。本案例只证明执行路径和数值正确性，
`performance_valid=false`，不发布性能结论。

## 复核

已有迁移产物可用独立 harness 复核。先激活一个已经安装依赖的 Python 环境，将
`ASNUMPY_RUNTIME` 指向按本仓库集成 patch 构建的、与 qualification receipt 哈希一致的隔离
runtime。输出路径必须不存在：

```bash
REPO_ROOT="${NUMPILOT_REPO_ROOT:-$PWD}"
ASNUMPY_RUNTIME="${ASNUMPY_RUNTIME:?请指向资格化的 AsNumPy 隔离 runtime}"
export PYTHONPATH="$ASNUMPY_RUNTIME:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export NUMPILOT_OPS_NN_MATMUL_AUDIT_RECEIPT="$REPO_ROOT/evidence/ops-nn-matmul-20260827/qualification-receipt.json"
python tools/ops_nn_matmul_case_evidence.py \
  --source examples/ops_nn_matmul_inference.py \
  --migrated evidence/ops-nn-matmul-20260827/final-migration/migrated.py \
  --migration-report evidence/ops-nn-matmul-20260827/final-migration/script-migration-report.json \
  --out /tmp/ops-nn-matmul-execution-evidence.json
```

原始审计环境的绝对路径会保留在 JSON 中作为运行身份，不是可移植命令。隔离 wheel 没有收入
Git 精选包；其精确源码 patch 位于
[`integrations/asnumpy/0002-ops-nn-matmul-native-fp32.patch`](../../integrations/asnumpy/0002-ops-nn-matmul-native-fp32.patch)，
公开证据用 SHA-256 绑定 patch、wheel 和 runtime。所有结论均限定于本 README 记录的
AsNumPy/CANN/SoC 和输入合同。
