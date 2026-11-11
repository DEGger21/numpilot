# np.cbrt 官方 ACLNN 候选审计（2026-11-11）

本文记录 `numpy.cbrt` 缺口在目标机器上的 official-first 解析与候选物化结果。解析和物化只读检查
已安装 CANN 的头文件、动态库和回执，不读取 `llm.json`、不调用模型、不联网、不编译源码、不导入
AsNumpy、不执行目标程序、不激活任何产物。

## 固定目标

- Qualified API：`numpy.cbrt`
- 目标 SoC：Ascend910B3
- CANN 版本：9.2.0-beta.2
- 已安装 CANN 根：由 `--cann-root` 显式指定；本文不记录私有绝对路径
- 固定 AsNumpy 组件：`0.3.1.dev70+g60d139c54`，wheel SHA-256
  `d3ea03b2fd8c6eace56f532e2b30a917d63a138abe654c0020c14499898ed682`
- 恢复运行时审计标识：
  `asnumpy-0.3.1-dev70-g60d139c54-internal-component-manifestff21a984-wheel-d3ea03b2-core44c5b349-v3`
- 恢复运行时 identity SHA-256：
  `e01f3baf024ecf51a3903315b07b12b7f08d08ad311f3073a34589eb37dc3fd7`

## 只读证据

| 工件 | 证据 |
|---|---|
| `include/aclnnop/aclnn_pow.h` | SHA-256 `08bc50d532ffb621fa5c8d397204717c59626a7b88fc0ed21791a0badb2ef593`，5459 bytes |
| `lib64/libopapi.so` | SHA-256 `0b93d5b6cdfda98f1176d1703fafdbc513d87d5b805ac7984d617a209615e58b`，53791584 bytes，2609 个 `aclnn*` 动态符号 |
| `lib64/libopapi_math.so` | SHA-256 `19bfa9a2b86bfe48e7113c6a98e4220ad12bbf42da8786b3956321515c036db7`，13181880 bytes，同时定义两个所需符号 |
| `lib64/libopapi_nn.so` | SHA-256 `814b9d8caf647d7bd6bfe8c2b29525641fe5c80b32ee7648378d6e1c9a095ff8`，18547536 bytes；不定义 `aclnnPowTensorScalar` / `aclnnPowTensorScalarGetWorkspaceSize` |

`aclnn_pow.h` 声明两段式接口：

```c
ACLNN_API aclnnStatus aclnnPowTensorScalarGetWorkspaceSize(
    const aclTensor* self, const aclScalar* exponent, const aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor);
ACLNN_API aclnnStatus aclnnPowTensorScalar(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
    const aclrtStream stream);
```

`nm -D --defined-only` 确认：

- `libopapi.so`：`aclnnPowTensorScalar`（`0x801508`）和
  `aclnnPowTensorScalarGetWorkspaceSize`（`0x806380`）均为已定义动态符号。
- `libopapi_math.so`：同名两个符号也均为已定义动态符号（`0x505628` / `0x50a4a0`）。
- `libopapi_nn.so`：两个符号都不存在，因此不在该映射的库优先级内。

CANN 9.2.0-beta.2 的 `aclnn_pow.h` 和上述三个 DSO 中都没有 `aclnnCbrt` /
`aclnnCbrtGetWorkspaceSize`。`cbrt` 只能通过已审计的 `aclnnPowTensorScalar` 组合实现，不能
伪称存在原生 cbrt 内核。

## 语义审计边界

`numpy.cbrt` 不是普通的分数次幂，适配器必须逐项处理以下差异：

1. **负数**：`pow(-8, 1/3)` 在实数域无效；`cbrt(-8) = -2`。实现必须使用
   `sign(x) * pow(|x|, 1/3)` 或等价的符号位组合，不能直接调用 `pow(x, 1/3)`。
2. **有符号零**：`np.cbrt(-0.0)` 返回 `-0.0`，且 `signbit` 为真。AsNumpy 现有
   `Copysign` 由 `Absolute(x1) * Sign(x2)` 组合，而 `Sign(-0.0) = +0.0`，会丢失负零。
   适配器不得直接复用该组合；必须使用 ACLNN 层符号位分支或等价实现，并在 qualification 中
   用 `signbit` / `nextafter` 检查 `-0.0`、`+0.0` 的符号。
3. **NaN 与无穷**：`cbrt(NaN) = NaN`，`cbrt(+Inf) = +Inf`，`cbrt(-Inf) = -Inf`。
   符号组合必须保持 `-Inf` 的符号，且不能把 NaN 静默改写为有限值。
4. **dtype 提升**：NumPy 2.5.4 实测为 float16→float16、float32→float32、
   float64→float64、int8/uint8/bool→float16、int16→float32、int32/int64→float64。
   AsNumpy 现有 `Power` 标量重载将输出硬编码为 `ACL_DOUBLE`、标量硬编码为 `ACL_FLOAT`，
   会破坏 float32 输入的输出 dtype；适配器必须直接使用 `aclnnPowTensorScalar`，
   显式构造输出张量和计算 dtype。
5. **指数精度**：`1/3` 不能精确表示，`pow(|x|, 1/3)` 与 `np.cbrt` 在末位可能不同。
   qualification 必须用 NumPy oracle 按 dtype 设定容差，同时对特殊值、符号、整数转换和
   dtype 做精确断言；不得用整体 `allclose` 掩盖特殊值错误。

## 审计映射

Catalog revision `installed-cann-aclnn-adapter-candidates-v5` 为同一 ACLNN 映射保留两个
独立审计画像：

| candidate_id | CANN | SoC | 库优先级 |
|---|---|---|---|
| `numpy.cbrt-aclnn-pow-tensor-scalar-910b3` | 9.2.0-beta.2 | Ascend910B3 | `libopapi.so`, `libopapi_math.so` |
| `numpy.cbrt-aclnn-pow-tensor-scalar` | 9.0.0 | Ascend910B4 | `libopapi_math.so`, `libopapi.so` |

两个画像的版本集合和 SoC 集合是独立成员集，解析器按精确 pair 过滤，不能把
`9.0.0/910B3` 或 `9.2.0-beta.2/910B4` 的交叉组合当作已审计。

910B3 画像优先 `libopapi.so` 的原因：reviewed AsNumpy 接入计划中 `bindings/python/CMakeLists.txt`
的 `_core` 链接图声明 `opapi`，库绑定证明要求选中的库 basename 与 `lib<link_item>.so` 一致；
`libopapi_math.so` 是第二审计提供者，`libopapi_nn.so` 因缺少符号被排除。

## 解析与物化回执

`resolve-gaps` 生成 write-once 回执，绑定源程序、GapSpec、目标 SoC/CANN 和上述工件摘要：

- `resolution_id`：`78697021063ccc1bc399351f0746e955ae2e0784edd1bdb019e48340ccc423fa`
- 解析回执原始文件 SHA-256：
  `39f8d198e277371ebca4faf6e821ab00afdf720d484c3276c99965e6f36a4716`
- 解析回执 canonical SHA-256（candidate provenance 绑定值）：
  `bb1aeb0663e73488e4fe759828d6f54b38015b4d5842d85bd13c29f58e1da662`
- `status`：`official_candidate_found`
- `route`：`installed_cann_adapter_candidate`
- `installed_cann.status`：`exact_candidate`
- `reason_code`：`header_and_dynamic_symbols_present`
- `semantic_equivalence`：`unverified`
- `required_next_gate`：`generate_and_qualify_asnumpy_adapter`

官方候选链随后生成并绑定了以下 write-once 回执：

| 工件 | SHA-256 |
|---|---|
| `handoff.json` | `c7060e15bee4b9799dd754d20c247e05ef59d3e85cc404a62cb7a21135faf35f` |
| `preflight.json` | `641b15aa5888a11dc2557f9c58b8c21fe958a31a814fa7eb02216baabd284dd0` |
| `adapter-input.json`（inert） | `32c1c961637dc1cf2f8ea9e25a7cad896bd71a746af9591bb20743c9a5526f8b` |
| `context-bound.json` | `03ee0909fb9186157581134aa468941d9d7d086398845e4a78f538ce7c5b8c38` |
| `context-gate.json` | `493deddfc698dd2ed45e94edcc9e51931f6be90cc234fd151db48481d9849da7` |
| recovery candidate `cbrt-3dd23c65b5ccd7fe.json` | `7a1b6aebca333df99f82bbb1203f235ceb53fc2839a0767e0c1fda9136b6809d` |

Recovery 首次物化为 `CANDIDATE_READY` 时：

- `phase`：`candidate_ready`
- `outcome`：`ready`
- `revision`：5
- `state_sha256`：`b4a603871853f83d2ea06bd579aa6fb13a40a3c20af1b5d2646b8251d555358f`
- `next_action`：`prepare_adapter_candidate`
- `candidate_id`：`candidate-e3b5fe9f8434dd1c8934`
- 该次物化回执的模型调用、构建、导入、执行、激活副作用均为 `false`

## 集成与 I001 资格

后续修复轮次与集成均保留 write-once 证据，不覆盖失败回执：

| 工件 | SHA-256 |
|---|---|
| `adapter-v7` 报告（repair_2） | `4e85d4b31f11fc9006523e30cc1ed5d95e0c2c4b595cdf6df59c03716d23712b` |
| `adapter-v9` 报告（repair_3） | `de549580c8a39c57a6f4544e515514456f0aa37d2cd65f1883395d1da88ae419` |
| `adapter-v9` overlay `src/asnumpy/math.py` | `5c0a14eed2ea03bddeb898d4c67aad321d5558ddf0917616ed75f5ebf734f18e` |
| `adapter-v9` overlay `src/asnumpy/__init__.py` | `25ae7943b416e9cc955df20bf9d2bc9c3e63d33424e66b4984ebd3ab7f62d834` |
| 集成预检 `preflight-integrate-v3.json` | `5fee09e83caf2fccdb4fff6c8079940308462957850dd2af904331ae942e9935` |
| 集成事务 `integration-v4/integration-transaction.json` | `4d0d1b964b56f5789b5f2907f94fd8ca6e988cfd8e68863a7d9332c7e990ff30` |
| 集成 gate 结果 `integration-v4/gate/result.json` | `d482360f6fb211052db6f59ad622cf28c4adf7695d44199fe4afbc6f7f9b4b54` |
| I001 资格回执 `i001-qualification-v4.json` | `9f8b37731ec39d67578e67e637a3d6dee5295b3b82374f5c7911d06b6ea55c12` |

集成预检结果为 `status=ready`、`code=ready_for_integration`，所有 side effect 均为
`false`。`integration-v4` 在隔离 AsNumpy 副本中完成构建、加载和确定性 NumPy oracle
gate：

- `status=passed`，transaction `int-bd26f33797b9a9a69fc2f19d`
- gate API 为 `numpy.cbrt`，12 个 case 全部通过，`oracle_comparisons=110`
- `dispatch_observation={"native_calls": 110, "fallback_calls": 0}`
- recovery 推进到 `revision=13`、`phase=integrated_candidate`、`state_sha256=9766b4f8d1cd9ec54c76bc08ad45a6be869887f7680c8a3bc765aa7798754dc4`

I001 资格回执在 `integration-v4/runtime/site` 上运行原始
`benchmark/cases/I_gap_model/I001_cube_root/program.py`，结果 `status=passed`：

- dtype `float32`、shape `[257]`，最大绝对误差 `4.76837158203125e-07`，最大相对误差
  `1.6618028467015123e-07`
- `signbit_matches=true`，`negative_zero_signbit=true`，NaN/Inf 模式、负值符号、
  dtype/shape 和重复调用稳定性全部通过
- 单独的 `1e30` 特殊值探针最大相对误差 `1.2288e-06`，在 `rtol=2e-6` 下通过；
  I001 的 `[-27, 27]` 仍使用 `rtol=1e-6` 严格比较
- 资格脚本对首个公开调用做了 `_cbrt` 绑定观测（`native_calls=1`，`fallback_calls=0`）；
  110 次原生调用的完整路由证据来自上面的 integration gate，不由该单次观测代替
- 计时仅作边界记录，不作性能结论：I001 小输入 CPU `0.008130 ms` / NPU
  `0.438893 ms`；1,000,000 元素 CPU `26.906746 ms` / NPU `2.295518 ms`

## 边界

该文档证明已安装 CANN 中存在与审计映射一致的 ABI、头文件和动态符号，并证明
`adapter-v9` 在隔离构建、strict-NPU 路由和 I001 float32 NumPy oracle 上通过。它不证明
存在原生 `aclnnCbrt` 内核，也不证明 float16/64、整数、bool 等全部 dtype 已覆盖；当前
native binding 只接受 `ACL_FLOAT`，I001 的 float32 路径已覆盖。尚未执行正式 official
qualification producer 重跑、activation、全量 150-case benchmark 或性能结论。手工实现或
模型输出都不能跳过构建、路由、数值对拍和性能门禁。

## 复现方式

```bash
export CANN_ROOT=/path/to/installed/cann
python -m numpilot resolve-gaps \
  benchmark/cases/I_gap_model/I001_cube_root/program.py \
  --target-soc Ascend910B3 \
  --cann-version 9.2.0-beta.2 \
  --cann-root "$CANN_ROOT" \
  --out /absolute/private/cbrt-resolution.json
```

复现时必须比较完整 `resolution_id`，不能只比较 `status`。任何头文件、库、CANN 版本或 SoC
漂移都会使该回执失效，需要重新审计。
