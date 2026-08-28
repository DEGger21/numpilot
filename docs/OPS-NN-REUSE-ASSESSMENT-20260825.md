# 官方算子复用评估与当前实现状态

> 首次结论：2026-08-25 UTC<br>
> 状态更新：2026-08-27 UTC<br>
> 当前结论：官方算子候选的**只读发现、统一路由与 hash-bound 收据门禁已经落地**；
> 通用的 official Artifact、定向构建、私有 OPP 与自动编排仍未形成闭环；但
> `numpy.matmul` 已完成一个 installed-ACLNN 优先、ops-nn 来源固定、AsNumPy 接入、六门
> 资格化、精确 runtime 激活、rescan 和最终迁移的受限真机案例。

## 1. 最终采用的解析优先级

NumPilot 只有在普通脚本已经得到 source-bound `GapSpec` 后，才进入缺口恢复。新 kernel 不是
第一选择，固定优先级如下：

```text
GapSpec
  → 1. 已安装 CANN ACLNN 候选（只读检查 header、ELF symbol、版本与文件 hash）
      ├─ exact candidate：逐字段语义审计 → configured model 生成 AsNumpy adapter
      ├─ clean no_match：继续查询固定版本的本地官方源码
      └─ incompatible / ambiguous / error：失败关闭，不伪装成“没有官方实现”
  → 2. 本地、版本固定的 cann/ops-nn Git 对象（只读，不 checkout/fetch/build）
      ├─ exact candidate：逐字段语义审计 → official Artifact → 私有 OPP
      │                    → configured model 生成 AsNumpy adapter
      ├─ clean no_match：才允许进入新 kernel 路径
      └─ incompatible / ambiguous / error：失败关闭
  → 3. CANNBot Skill + configured model 生成新的 Ascend C kernel/adapter
  → integrate → 六门 qualification → activate → rescan → final migration
```

当前项目实际使用的 configured model 是 GLM；产品合同仍按“用户配置的精确模型身份 + 回执”
表达。官方 kernel 的来源必须记录为 `official_source`，不能伪装成 GLM 生成代码；为项目新增
的迁移、adapter 和新 kernel 仍遵守 configured-model/GLM 生成边界。

这一顺序避免两个错误：已有 ACLNN 时重复编译官方源码，以及官方候选存在或解析异常时直接让
模型手写同名算子。只有两个官方层级都得到可信的 `no_match`，新 kernel generation 才有资格
开始。

## 2. 已实现的只读发现原语

### 2.1 已安装 CANN ACLNN

`numpilot/gaps/installed_cann.py` 已实现显式 CANN 根目录下的只读候选发现：

- 规范化并约束 CANN 根目录，核对版本证据；
- 读取候选 ACLNN header，并用绝对路径的 `nm` 或 `readelf` 检查共享库动态符号；
- 记录 header、library、symbol set、catalog mapping 等 hash-bound 证据；
- 返回 `exact_candidate`、`no_match`、`incompatible`、`ambiguous` 或 `error`；
- 不 import CANN、不加载 `.so`、不执行用户代码、不调用模型，也不修改 toolkit。

当前小型审计 catalog（CANN 9.0.0、精确 `Ascend910B4`）包含 `numpy.cbrt`、
`numpy.arange`、`numpy.cumsum`、`numpy.cumprod`、
`numpy.random.normal`、`numpy.matmul`、`operator.erf` 和 `operator.smooth_l1_loss` 的 ACLNN
adapter 候选映射。
该映射只证明“版本匹配的 header 与两阶段 ACLNN 符号存在”，不证明 NumPy 语义等价。例如
`cbrt` 必须实现 `sign(x) * pow(abs(x), 1/3)` 而不能直接对负数做分数幂；signed zero、NaN、
Inf、`axis=None`、dtype promotion、随机数状态、原地输出、标量 shape 和异常边界仍需独立审计。

### 2.2 固定版本的 ops-nn 本地源码

`numpilot/gaps/official_source.py` 已实现本地 Git object resolver。当前生产 pin 只有 CANN
9.0.0：

| 字段 | 固定值 |
|---|---|
| official origin | `https://gitcode.com/cann/ops-nn.git` |
| ref | `refs/tags/v9.0.0` |
| commit | `0413e303e6cd1986b592c3e039f2579ac0f10b62` |
| tree | `827d73df88f73353821b63ed4e0d1711b6d2cb50` |
| license | CANN Open Software License Agreement Version 2.0 |

本机 CANN 为 9.0.0，因此不能直接跟随本地 `ops-nn` 的 9.1.0 `master` 工作树。resolver 会从
本地仓库读取上述 tag 对应的 immutable Git object，核对 official origin、commit、tree、项目
版本、许可证标识、候选目录树和证据文件 hash；它不会切换工作树、checkout、clone、fetch、
下载依赖或构建源码。

名称归一化命中默认仍是 `ambiguous`/`requires_audit`。即使外部语义审计回执与 candidate、
API、CANN、SoC 和证据 hash 全部绑定，resolver 给出的也只是下游可审计候选，
`activation_allowed` 始终为 `false`。

### 2.3 统一只读路由与收据门禁

`numpilot/gaps/official_resolution.py` 已把两个 resolver 串成失败关闭的只读路由。
`resolve-gaps` 在不读取模型配置、不读取 CANNBot Skill 的情况下生成 write-once、权限 `0600`
的 `official-resolution.json`。收据绑定源码绝对路径与 hash、完整 GapSpec hash、CANN/SoC、
显式 CANN/ops-nn 根目录、许可证接受状态以及两个 provider 的证据，并以
`resolution_id` 绑定完整内容。

`complete-gaps --official-resolution-report` 会在模型 readiness 和 Skill 读取之前重新静态准备
GapSpec、校验收据并重新盘点当前 CANN/ops-nn 证据。候选命中、证据漂移、收据篡改、版本
不兼容、歧义或解析错误都会阻断新 kernel；只有 fresh `resolution_id` 证明所有 Gap 的
installed CANN 和 pinned ops-nn 均为 clean `no_match`，才会允许
configured model 进入新 kernel generation。

最小显式流程如下；路径必须替换为当前机器的实际绝对路径：

```bash
OFFICIAL_RECEIPT=/absolute/private/run/official-resolution.json

python -m numpilot resolve-gaps examples/gap_cbrt.py \
  --out "$OFFICIAL_RECEIPT" \
  --target-soc Ascend910B4 \
  --cann-version 9.0.0 \
  --cann-root /absolute/path/to/cann-9.0.0 \
  --ops-nn-root /absolute/path/to/ops-nn \
  --accept-cann-osl

python -m numpilot complete-gaps examples/gap_cbrt.py \
  --out /absolute/private/run/gap-output \
  --target-soc Ascend910B4 \
  --cann-version 9.0.0 \
  --provider llm \
  --official-resolution-report "$OFFICIAL_RECEIPT"
```

有 Gap 的 `complete-gaps --provider llm` 现在强制要求该收据；未提供时会在模型 readiness 和
Skill 读取之前返回非零。无 Gap 时则直接生成本地 `no_gaps` 报告，不进行无意义的模型探测。
当前 installed ACLNN catalog 只审计精确的 `Ascend910B4`；`Ascend910B` family 名称会在任何
toolkit/ops-nn 读取前失败关闭，其他精确 SoC 也必须新增独立审计，不能从 B4 推断。

## 3. 为什么官方候选仍不能直接使用

名字、header、符号或源码目录存在，都不能回答以下问题：

- NumPy/public API 的签名、默认值、broadcast、dtype promotion、shape/layout 是否一致；
- mutation、alias、返回值和错误行为是否一致；
- 当前 SoC 是否真实可构建、运行，动态 shape/空 tensor/NaN/Inf 等边界是否覆盖；
- adapter 是否通过公开 AsNumpy API 返回正确的 `ndarray`，且没有 host fallback；
- 构建出的二进制是否与源码、工具链、CANN、SoC 和安装清单 hash-bound。

例如 `smooth_l1_loss_v2` 的 reduction 与标量输出 shape、`apply_gradient_descent` 的原地
mutation/alias 都必须对照 GapSpec，不能由同名关系推断。resolver 不负责解决这些问题，也不
会把候选写入 capability overlay。

## 4. 必须保留的安全与许可证边界

1. **路由门**：仅真实 missing API，或用户拒绝的 audited fallback，能产生 GapSpec；
   callable + `UNKNOWN` 仍先做 capability/source audit。
2. **版本与来源门**：已安装 ACLNN 绑定显式 CANN 根目录、版本和文件 hash；ops-nn 绑定
   official origin、ref、commit、tree、候选目录与证据文件 hash。版本不兼容不是 `no_match`。
3. **许可证门**：ops-nn 使用 CANN OSL v2。只读 resolver 会验证许可证身份和 hash；任何源码
   使用、构建或安装前仍必须由用户显式接受，并把接受状态写入后续 receipt。
4. **无隐式联网**：`pip install` hook、`import`、`doctor`、`migrate`、gap resolver 和
   `complete-gaps` 都不得隐式 clone/fetch/download。需要获取源码或依赖时必须使用单独、显式、
   可审计的联网动作。
5. **无系统写入**：候选发现绝不写 CANN/OPP。未来构建只能先落到仓库外隔离目录和私有 OPP，
   带安装清单、hash、激活环境和回滚路径；不得覆盖 `${ASCEND_HOME_PATH}/opp` 的系统内容、
   修改 shell 启动文件或静默提权。
6. **来源门**：官方源码必须成为 hash-bound `official_source` Artifact/manifest，不能伪装成
   `remote_llm`。adapter 和 qualification validator 必须显式接受并审计该来源。
7. **资格化门**：无论 ACLNN 已安装、官方源码复用还是模型新 kernel，都必须通过
   `artifact_integrity`、`build`、`unit_test`、`numerical_validation`、
   `public_asnumpy_api`、`npu_execution` 六门。`activation` 是六门之后的独立状态。

## 5. 实现状态：已完成与待完成

| 层级 | 当前状态 | 不能据此声称 |
|---|---|---|
| installed CANN ACLNN resolver | 已实现只读候选发现与证据绑定原语 | 语义合格、adapter 已生成、API 已接入 |
| pinned ops-nn resolver | 已实现 9.0.0 本地 Git object 解析与固定 commit/tree | 已接受许可证、已构建、已安装 |
| 统一 official resolution 编排/CLI | 已实现只读路由、write-once 收据、fresh recheck 与模型/Skill 前强制门禁 | 候选已自动接入、旧收据可跳过当前证据重验 |
| `official_source` Artifact/manifest | 待实现 | qualification 已接受官方来源 |
| 定向构建与私有 OPP 事务 | 待实现 | 系统 CANN 可写、已有卸载/回滚 |
| configured-model AsNumpy adapter 接入 | 已有独立生成/repair 原语；官方候选上下文接线待完成 | candidate 已成为 public AsNumpy API |
| 六门 qualification 与 activation | schema/overlay 原语已存在；官方链 runner 待完成 | 已通过真实 NPU 或已激活 |
| rescan 与最终迁移 | 待统一编排 | 缺口已经从原脚本消失 |

上述表格描述通用能力。MatMul 定向例外已经完成：固定 ops-nn v9.0.0
`matmul/mat_mul_v3` 来源，验证本机 `aclnn_matmul.h` 与官方头文件同哈希，AsNumPy 对限定
二维 `float32` 合同调用 `_core.matmul`/`aclnnMatmul`，六门资格化和真实 NPU 执行通过，
随后 ordinary migration 得到 `migrated_scoped_npu`。完整证据见
[`evidence/ops-nn-matmul-20260827/README.md`](../evidence/ops-nn-matmul-20260827/README.md)。
这不等于通用 runner 已实现，也不允许推广到 1-D、批量广播、混合 dtype、空张量或性能结论。

除这个明确的 hash-bound 例外外，当前最远仍只能报告“发现了候选”或“未发现候选”，不能报告
“算子已修复”。
`incompatible`、`ambiguous`、`error` 也不能降级成 `no_match` 后偷偷调用模型。

## 6. 架构图与后续验收

三页可编辑 draw.io 图见
[官方算子解析与缺口恢复架构图](numpilot-official-operator-resolution.drawio)，包含总体主流程、
官方候选门禁，以及运行时/资格化/证据边界。

下一阶段验收至少需要：

- 用真实 GapSpec 在目标环境继续验收 installed ACLNN 命中、ops-nn 命中、两层可信
  `no_match`、版本不兼容和解析错误；
- 完成 hash-bound `official_source` Artifact 与 validator；
- 在私有 OPP 验证定向构建、发现、激活、卸载和回滚，不触碰系统 OPP；
- 完成 configured-model adapter、六门证据、activation、rescan 和最终迁移；
- 在干净环境证明任一异常都失败关闭，且 resolver 全程无模型调用、无网络、无系统写入。

## 7. 官方依据

- [ops-nn 仓库](https://gitcode.com/cann/ops-nn)
- [v9.0.0 项目说明](https://gitcode.com/cann/ops-nn/blob/v9.0.0/README.md)
- [v9.0.0 算子列表](https://gitcode.com/cann/ops-nn/blob/v9.0.0/docs/zh/op_list.md)
- [算子编译、安装及离线依赖说明](https://gitcode.com/cann/ops-nn/blob/v9.0.0/docs/zh/invocation/quick_op_invocation.md)
- [CANN OSL v2.0](https://gitcode.com/cann/ops-nn/blob/v9.0.0/LICENSE)
