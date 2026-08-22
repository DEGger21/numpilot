# Profile-aware runtime 与真实 NPU composite probe 交接（2026-08-13）

这是 2026-08-12 qualification/overlay 交接之后的当前主线。旧文档里“runtime 接线尚未
完成”和“本机 NPU 不可用”的结论已经过时；本文件不宣称 custom operator 已资格化。

## 实测证据索引

- [`docs/evidence/heat-diffusion-npu-20260813.md`](evidence/heat-diffusion-npu-20260813.md)：
  纯 NumPy 二维热扩散单步网络经 AST 全量迁移后的 strict-NPU、三层零误差与真实性能留档；
  主证据为 CPU `15.552 ms`、NPU `14.424 ms`、`performance_valid=true`、无 fallback。

## 已完成：profile-aware runtime

- capability schema 升为 `1.1`，qualification gate report schema 升为 `2`；四个运行相关
  gate 必须绑定全部 typed `signature_variants` 的规范 SHA-256 与完整覆盖数，旧 v1 证据
  或只覆盖部分签名的证据失败关闭；
- `TensorProfile` / `OperatorCallProfile` / `AnalysisProfileSet` 使用严格 schema；当前唯一
  接受的 scope grammar 是 `dtype=float32`、`rank>=N`、`contiguous=true|false`，三项
  必须齐全且 AND 到每个 operand；未知字段、缺元数据、非 tensor、非 native endian、
  ndarray subclass 和半 scoped capability全部拒绝；
- `NpuBackend` 只接受显式 `OverlayRuntimeConfig`，不从环境自动发现。overlay 全量复验、
  all-or-nothing，base capability、runtime callable、环境和 overlay admission 使用同一快照；
- `AnalysisAgent` 只使用调用方提供且与 source SHA、qualified name、行列和签名精确绑定的
  profile，不执行 `build_inputs`、不从 AST 猜 dtype/shape/layout；
- `RuntimeProfileGuard` 在每个真实 `ap.*` 调用的目标 dispatch 前复验 profile；
  Verification 和 bench 的 common/artifact 两条路径都接入 guard。pending、matched、
  rejected 回执分离，外部不能通过修改回执或原 manifest 字典提升状态；
- strict receipt 会完整反序列化 capability。第三方 validator 自报 scoped matched、缺字段、
  未知 constraint 或缺少真实 guard 均拒绝；
- strict NPU execution session 绑定资格审计时的 exact callable。直接替换 target、修改已建模
  Python callable `__code__`、函数依赖或 mutable default 会在 dispatch 前拒绝；生产
  `NpuBackend` 的 manifest 与 target 来自同一次 lazy lookup。此机制只做意外漂移检测，
  不声称能抵御同进程恶意 introspection/monkeypatch；自定义 backend/subclass 属可信扩展边界。

CLI 显式参数（`migrate` 和 `bench-npu`）为：

```text
--capability-overlay <overlay.json>
--qualification-evidence-root <trusted-root>
--analysis-profiles <source-bound-profiles.json>
```

overlay 与 evidence root 必须成对出现，只允许 NPU backend；任何读取、schema、CAS、环境、
module hash、callable 或 profile 漂移都不回退 base capability。

## 真实 NPU composite `cbrt` 探针

隔离工作区（未改系统 Python、系统 OPP 或原 AsNumpy 仓库）：

```text
/home/ma-user/work/numpilot-gap-runs/asnumpy-runtime-20260813/
```

私有 wheel：

```text
cbrt-dist/asnumpy-0.3.1.dev4+g1929a3ba4.cbrt20260813-cp310-cp310-linux_aarch64.whl
size=668639
sha256=ca581bfbedd2168386ea5f9694afe7a3e9b6e68ae2a4748b31c177f304d26c5f
```

实现是 AsNumpy/C++ 里的 ACLNN 组合：

```text
Abs -> PowTensorScalar(1/3) -> Sign -> Mul -> EqScalar(0) -> SWhere(original x)
```

最后的 zero select 保留 `-0.0`。public API 在任何 ACLNN 调用前拒绝非 float32。2026-08-13
22:45 UTC，从没有继承污染 `LD_LIBRARY_PATH` 的 shell source
`/usr/local/Ascend/cann-9.0.0/set_env.sh` 后，`npu-smi` 看到 910B4 Health OK；私有 venv
实测结果：

```text
pytest test_cbrt.py: 2 passed in 2.55s
oracle cases: 200015 (finite 200012)
nan_mask_equal=true, inf_mask_equal=true, signbit_equal=true
output_dtype=float32
allclose(rtol=2e-5, atol=2e-6)=true
max_relative_error=2.7102692001790274e-06
max_ulp_error=44
boundary sizes 0,1,63,64,65,1023,1024,1025: all passed
```

`max_abs_error=12582912.0` 来自接近 float32 极值范围的 wide-magnitude case；相对误差和
上述 allclose 门禁通过。需要按实际容差用途判断 44 ULP 是否可接受，不能把它改写成
bitwise/ULP 精确。

### 证据边界

这次 probe 证明“纯 ACLNN、无 NumPy CPU fallback 的复合 public API 可以在本机真实 NPU
执行并满足当前 float32 oracle 容差”。它不是单 custom kernel，不是 private OPP，也没有
由受信 harness 生成 qualification schema v2 的 artifact/build/UT/numerical/public/NPU 六门
报告。因此：

- 不把它写入 `SUPPORTED_OPS`；
- 不发布/激活 qualification overlay；
- 不把 composite wheel 当成 `CUSTOM_NPU`；
- `numpy.cbrt` 继续保持 `generated_unverified`。

## Custom OPP 的真实 blocker

生成工程位于：

```text
/home/ma-user/work/numpilot-gap-runs/cbrt-qualification-20260813/operator-project
```

当前只有 8 个 scaffold 文件：kernel 仍是 TODO，CMake/op_host 错配为 `ascend910`，而本机
目标是 `ascend910b`/dav-2201；tiling 无空 tensor、溢出、UB/Power scratch、多核/tail
建模。arch2201 `Compares` 要求每次 float32 count 以 64 元素对齐。该工程没有被构建或
安装，不能进入资格验证。

CANN 9.0 的 `libopapi`/`libopapi_math` 没有 `aclnnCbrt`；910B 也不能使用仅面向其它架构
启用的 SIMT Cbrt。可信单 custom-op 路径仍需按 arch22 registry template 实现
Abs/Power/Compares/Select kernel、正式 tiling、L0/L2/public AsNumpy 接入，再由私有、版本化
OPP root 生成六门 schema v2 evidence。

## 下一步

1. 在独立 custom-op 源目录实现 ascend910b arch22 kernel/tiling/L0/L2，禁止运行现有危险
   `build.sh` 或 `numpyexample/build_custom_ops.sh`；
2. 用外部 build dir 和全新 private OPP root 构建安装，绝不落系统 OPP；
3. 由可信 harness 对每个 signature variant 生成 schema v2 六门报告并绑定所有哈希；
4. 发布 overlay 后，用本文件已完成的 profile-aware CLI 跑 qualified + activated 的迁移、
   数值和真实性能回归；
5. 同进程 fingerprint 是当前 MVP 的意外漂移加固，不是恶意代码安全边界。Python closure
   与 module/mutable dependency graph 对同进程代码仍可达；运行不受信源码的最终安全边界
   必须是独立、受限执行进程。
