# 比赛交付与作者接管（Phase 10）

> **历史交付记录（2026-11-05）**：本页保留 Phase 10 的旧版本验收与接管证据，数值绑定文中记录的旧运行时，不代表当前静态 AsNumpy 基线或当前 demo 结果。当前复现入口请使用 `examples/demo/` 与仓库根目录的安装文档。

本页是 Phase 10（比赛交付与作者接管）的验收记录和操作入口。基线为
`f7888eb35c42d50d6604b8f66c6c5400a2cd9c9c`（`update 20261105`，2026-11-05），
环境为 Ascend 910B3、CANN `9.2.0-beta.2`、Python `3.12.14`、NumPilot
`0.3.0a1.dev0`、AsNumpy `0.4.1.post0+numpilot.bee27ba2`。

> 本页只记录可复现的命令和回执字段。运行目录、wheel、benchmark receipt 和模型回执
> 默认保存在仓库外；不要把私有绝对路径、token、endpoint、原始模型响应或 prompt
> 明文写进公开文档或提交。

## 1. 交付物

| 交付物 | 仓库内位置 | 状态 |
| --- | --- | --- |
| NumPilot 源码、CLI、测试 | `numpilot/`、`tests/` | 1856 passed（Phase 10 最终工作树复跑） |
| 文档与架构图 | `docs/`、`docs/assets/architecture/` | 已收口，含本页 |
| AsNumpy 源码基线与 wheel | `components/asnumpy/` | manifest、wheel、SBOM、锁定说明齐全 |
| CANNBot 静态 bundle | `components/cannbot-skills/` | 离线可读；运行时不会下载 |
| 150 例 benchmark 与预期路由 | `benchmark/cases/`、`benchmark/expectations.json` | 150/150 honored，unexpected 0 |
| 性能矩阵工具与口径 | `numpilot/perf_matrix.py`、`tools/performance_matrix.py`、本页链接的 `performance-matrix.zh-CN.md` | 170/170 数值检查通过；`performance_valid=false` |
| 干净 checkout 与发布证据 | 仓库外 `runs/`（见 §7） | release tree、clean install、replay、wheel/sdist 均已验收 |
| 演示路线 | 本页 §5 | direct、gap、错误候选、replay、性能五条 |

## 2. 验收结果

以下结果均在基线 `f7888eb` 的干净 checkout 上产生；`release-tree.json` 的
`ok=true`、461 个文件、12,125,621 bytes，`warnings=[]`、`errors=[]`。加入本页后，
工作树为 462 个文件、12,138,967 bytes；提交前 `--require-clean` 会按设计失败。

| 验收项 | 关键回执 | 结果 |
| --- | --- | --- |
| release tree | `release-tree.json`、`release-tree-postdocs.json` | 基线 `ok=true`（461 文件）；加入交付页后 462 文件，dirty-allowed `ok=true`，`--require-clean` 在作者提交前按设计失败 |
| clean checkout | `checkout/` | `f7888eb`，461 个文件，`git status` clean |
| clean install | `clean-install.log`、`doctor-shim.txt` | `numpilot 0.3.0a1.dev0` 安装成功；`doctor --backend shim` exit 0 |
| 全量测试 | `pytest-phase10.log` | `1856 passed in 71.26s`（最终工作树，含新增文档） |
| wheel / sdist | `dist/`、`release-artifact-manifest.json` | wheel sha `f65fc6de…c341ea`，sdist sha `6be6efa1…f7731c`；manifest `release_candidate_ready`，`--verify` 2/2 passed |
| wheel 独立安装 | `wheel-venv/`、`wheel-replay.json` | 新 venv 安装 wheel 后 `--version` 正常，replay exit 0 |
| 只读 replay | `replay.json` | `replay_mode=read_only`、`execution_performed=false`、`actual_remote_call=false`、`receipt_integrity.status=passed` |
| direct 示例路由 | `analyze-direct.json` | `CAPABILITY_AUDIT_REQUIRED`；未调用模型，未执行源码 |
| gap 示例路由 | `analyze-gap.json` | `BLOCKED`；需要显式契约改写，未调用模型，未执行源码 |
| benchmark 子集（shim） | `benchmark-subset.json`、`wheel-benchmark-subset.json` | 30 例 `status=pass`，`unexpected=0`；16 migrated、14 unsupported |
| 全量 benchmark（shim） | Phase 8 `full-suite-03/summary.json` | 150 例 `status=pass`；68 migrated、15 refused、66 gap stop、1 audit failure；direct 51/51、gap 83（66 honored + 17 improved）、audit 1/1、reject 15/15 |
| gap 数值复验 | `gap-verify.json` | `status=passed`、`numerically_consistent=true`、`max_abs_diff=0.0`、`strict_npu_verified=false` |
| 错误候选拒绝 | Phase 6 `verification.json` | `status=failed`、`failure.code=numerical_mismatch` |
| 性能矩阵 | Phase 9 `matrix-02/perf-matrix.json` | 170/170 检查通过；`status=profile_required`、`performance_valid=false` |
| 发布树公开扫描 | Phase 7 `public-scan-final.json`、Phase 10 `public-scan-phase10.json` | Phase 7：435 个文本文件、0 命中；Phase 10：451 个文本文件、0 命中（1 条测试夹具已人工复核并列入 allowlist） |

## 3. 干净机器安装

```bash
git clone <repo-url> numpilot
cd numpilot
git checkout f7888eb35c42d50d6604b8f66c6c5400a2cd9c9c

python3.12 -m venv .venv
. .venv/bin/activate

# PEP 621 的 license = "Apache-2.0" 需要 setuptools>=77
python -m pip install "setuptools>=77" "loguru>=0.7" "rich>=13"
python -m pip install --no-build-isolation .

python -m numpilot --version          # numpilot 0.3.0a1.dev0
python -m numpilot doctor --backend shim
```

`doctor --backend shim` 允许两个已知警告：CANN 路径同根别名，以及 shim 后端不需要
AsNumpy。需要真实 NPU 时另行安装锁定的 AsNumpy wheel，并接受“运行时身份未审计”的
fail-closed 边界。

构建发布工件：

```bash
python -m pip install "build>=1"
python -m build
python tools/release_artifact_manifest.py \
  --output <runs>/release-artifact-manifest.json \
  --require-clean --require-asnumpy-component \
  dist/*.whl dist/*.tar.gz
python tools/release_artifact_manifest.py --verify <runs>/release-artifact-manifest.json \
  --artifact-dir dist
```

## 4. 发布树与工件校验

```bash
python tools/check_release_tree.py \
  --require-clean --require-asnumpy-component --json
```

基线 `f7888eb` 上应得到 `ok=true`，文件数 461，`errors=[]`；提交本页后应得到
`ok=true`，文件数 462。新增文档尚未提交时，`--require-clean` 会按设计失败；
先由作者提交，再重跑该命令。

工件 manifest 的验收：

- `status=release_candidate_ready`
- `source.commit=f7888eb…`、`source.dirty=false`
- wheel/sdist 的每个文件都有相对路径、sha256 和 size_bytes
- `--verify` 输出 `artifacts_verified=2`、`status=passed`

## 5. 五条演示路线

### 5.1 direct 成功（确定性、不调用模型）

`benchmark_suite` 的 `A_direct_simple` 组使用显式 `ap.*` 改写和 shim 后端，
逐用例产生 `receipt.json` 与 `report.json`：

```bash
python -m numpilot.benchmark_suite \
  --root benchmark/cases \
  --out <runs>/direct-demo \
  --backend shim \
  --groups A_direct_simple D1_partial_low_kernel \
  --json
```

预期：`status=pass`、`total=30`、`expected=29`、`improved=1`、`unexpected=0`；
direct 路由 15/15 honored，gap 路由 14 honored + 1 improved；状态计数为
16 migrated、14 unsupported。抽查：

```bash
python - <<'PY'
import json
p = "<runs>/direct-demo/cases/A_direct_simple/A001_add_sqrt/receipt.json"
d = json.load(open(p))
print(d["observed"]["status"], d["observed"]["numerical_verified"],
      d["observed"]["max_rel_diff"])
PY
```

预期 `migrated True 0.0`。边界：`backend=shim`，`strict_npu=false`，证明的是
NumPy 语义下的 direct 数值一致，不是 NPU 执行或加速结论。

### 5.2 gap 成功（scoped-NPU 数值一致）

`examples/concatenate_feature_join.py` 的唯一缺口是 `np.concatenate`，已通过官方
ACLNN overlay 完成解析、候选、API gate、集成和全程序验证。

已有迁移工件的复验（不调用模型）：

```bash
python -m numpilot verify examples/concatenate_feature_join.py \
  --migration-report <runs>/concatenate-phase2-20261103/final-migration/script-migration-report.json \
  --output joined \
  --out <runs>/gap-verify.json --json
```

预期：`status=passed`、`numerically_consistent=true`、`joined` 为 (3, 6)、
18 个元素、`max_abs_diff=0.0`、`max_rel_diff=0.0`；`strict_npu_verified=false`、
`performance_valid=false`。此路径会执行 AsNumpy/NPU，但不会调用模型。

在已有官方 overlay 证据的机器上从头生成 gap 工件（会调用已配置的模型，必须先确认
`llm.json`；overlay 必须来自已经通过六门 gate 的证据目录）：

```bash
python -m numpilot migrate examples/concatenate_feature_join.py \
  --backend npu --fallback-policy reject \
  --capability-overlay <runs>/concatenate-phase2-20261103/qualification/capability-overlay.json \
  --qualification-evidence-root <runs>/concatenate-phase2-20261103 \
  --out <runs>/concatenate-gap
python -m numpilot verify examples/concatenate_feature_join.py \
  --migration-report <runs>/concatenate-gap/migration/script-migration-report.json \
  --output joined \
  --out <runs>/concatenate-gap/verification.json --json
```

预期迁移报告 `status=migrated_scoped_npu`，验证 `status=passed`。发布时必须保留
`strict_npu_verified=false` 和 `performance_valid=false`，不能把 scoped-NPU
数值一致写成 strict-NPU 或性能结论。

### 5.3 错误候选拒绝

Phase 6 的负向运行保留了一个真实数值不匹配的候选：

```bash
python - <<'PY'
import json
r = json.load(open("<runs>/phase6-negative-20261104/verification.json"))
print(r["status"], r["failure"]["code"])
print(r["comparisons"])
PY
```

预期 `failed numerical_mismatch`，比较结果为不一致。该回执是 write-once 的，
修复必须走有界 repair 链（最多 3 轮），不能覆盖原始失败证据。

### 5.4 只读 replay

```bash
python -m numpilot replay <runs>/phase7-replay-20261104 --json
```

预期：`replay_mode=read_only`、`execution_performed=false`、
`actual_remote_call=false`、`receipt_integrity.status=passed`。replay 只读取
已有 workflow report 和阶段回执，不读取 `llm.json`、不访问网络、不执行源码或 NPU。
该示例的 workflow 状态是 `blocked`，因为它是负向运行；replay 证明的是回执完整性，
不是迁移成功。

### 5.5 性能口径

性能矩阵与完整口径见 [性能矩阵（Phase 9）](performance-matrix.zh-CN.md)。要点：

- `matmul` 和 `sin_matmul` 两个 profile，覆盖 64–1024；
- 单核 CPU 对单卡 NPU，含 resident、warm E2E、cold E2E；
- 170/170 数值检查通过，但 `sin`/`matmul` 的 provenance 为 `UNKNOWN`，
  `status=profile_required`、`performance_valid=false`；
- 因此所有比率只作诊断，不能作为公开加速比。

## 6. 已知边界

1. 当前 AsNumpy 运行时 `audit_id=unaudited-asnumpy-version`。在完成运行时审计前，
   `analyze` 对真实脚本返回 `CAPABILITY_AUDIT_REQUIRED` 或 `BLOCKED`，并在调用模型前
   fail-closed；这是产品边界，不是遗漏。
2. official ACLNN overlay 只覆盖 `concatenate`、`ones`、`zeros`。`sin`、`matmul`
   虽然 `available=true`，但 provenance 为 `UNKNOWN`，不能发布 strict-NPU 加速比。
3. benchmark 全量结果使用 `backend=shim`，`strict_npu=false`；它证明路由、改写和
   NumPy 语义数值一致，不证明 NPU 执行或性能。
4. 干净安装需要 `setuptools>=77`；系统自带 67.4.0 会拒绝 PEP 621 的
   `license = "Apache-2.0"`。
5. 本次交付没有执行 commit、push、merge 或 reset；所有提交由作者手动完成。

## 7. 证据索引

仓库外运行目录（相对 `<runs>`）：

- `phase10-delivery-20261105/`：`release-tree.json`、`release-tree-postdocs.json`、
  `checkout/`、`venv/`、`wheel-venv/`、`clean-install.log`、`doctor-shim.txt`、
  `replay.json`、`analyze-direct.json`、`analyze-gap.json`、`benchmark-subset.json`、
  `wheel-benchmark-subset.json`、`public-scan-phase10.json`、`pytest-phase10.log`、
  `dist/`、`release-artifact-manifest.json`、`gap-verify.json`、`wheel-replay.json`；
- `phase8-benchmark-20261104/full-suite-03/`：150 例 summary、逐用例 receipt/report；
- `phase9-performance-20261105/matrix-02/`：`perf-matrix.json`、`PERF-MATRIX.md`、
  170 个 worker receipt、`SHA256SUMS`；
- `phase7-replay-20261104/`：workflow report、阶段回执、replay 输出；
- `phase6-negative-20261104/`：失败验证回执和二次复验；
- `concatenate-phase2-20261103/final-migration/`：迁移报告、`migrated.py`、
  全程序验证回执。

## 8. 作者接管清单

1. 复核工作树：`git status --short`、`git log -1 --oneline`；确认 HEAD 是
   `f7888eb`，新增的交付文档尚未提交。
2. 提交前重跑公开扫描：检查 `docs/`、README、计划书和日志中没有私有绝对路径、
   token、endpoint、原始响应或 prompt 明文。
3. 提交本阶段文件（建议一起提交）：
   - `docs/delivery.zh-CN.md`（新增）
   - `docs/README.md`（修改）
   - `README.md`、`README.zh-CN.md`（修改）
4. 提交后重跑 `tools/check_release_tree.py --require-clean --require-asnumpy-component`，
   应恢复 `ok=true`。
5. 备份：复制工作树、`documents/` 和 Phase 10 证据到
   `npu-copy/history/20261105-stage27-phase10-delivery/`，刷新 `current`，
   生成并校验 `MANIFEST.sha256`；排除 `.git`、`llm.json`、私钥、虚拟环境、
   `__pycache__`、`build/dist` 和 `.pytest_cache`。
6. 若后续要解锁 strict-NPU 或性能结论，先审计当前 AsNumpy 运行时，再新建
   `matrix-03` 重新测量；不要复用或改写 `matrix-02` 的证据。
