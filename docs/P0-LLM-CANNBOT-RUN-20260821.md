# P0：本地 LLM + 华为 CANNBot Skill 直接补算子

日期：2026-08-21（UTC）

## 当前结论

NumPilot 的 P0 主链路已经跑到“真实模型生成 → 编译/运行诊断回传 →
模型定向修复 → 修复链封存 → Ascend910B kernel 编译 → 私有 OPP 安装 →
六门资格化 → capability overlay 显式激活 → 完整动态 Poisson 多轮 strict-NPU”这一步：

```text
扫描用户程序
→ 识别缺失的显式 OperatorSpec 接口
→ 只为缺口生成本地算子工程
→ 在隔离目录执行 CPU oracle/CANN 构建门
→ 把精确诊断回传同一 LLM
→ 只替换失败文件并保存请求回执、前后哈希和父修复哈希
→ 复验整条父链并发布新的不可变 Artifact
→ 从封存 Artifact 重新构建直到 host/API/kernel/package 通过
→ 安装到运行目录下的私有 vendor root，不写系统 CANN
→ 用固定 AsNumpy wheel 调用公开 API 并在 910B4 上对拍 NumPy oracle
→ 生成 schema v2 六门证据并原子发布 capability overlay
→ 修复并哈希绑定限定 profile 的 outer/matmul ACLNN 路径
→ 显式加载 overlay，运行完整动态训练轮并核验 117 个调用点
→ 循环前一次上传，250 轮内接力设备参数，结束后一次下载并对拍 CPU
```

首轮不拉取、不构建、不依赖 `ops-nn`。它以后可以作为人工对照或可选实现源，但不是
启动前置条件，也不是当前生成 Artifact 的语义真源。CodeArts 接入按用户决定继续后置，
本轮只使用 `/home/ma-user/work/.numpilot-config/llm.json` 中的大模型。

必须严格区分当前结果：

- 真实远端调用、回执和原始签名草稿：已完成；
- 两个最终修复链已封存为新 Artifact；其 CPU oracle、host/API、Ascend910B kernel、
  package、私有 OPP 安装、public `asnumpy` 调用和真机 NPU 数值：已通过；
- NumPilot 正式六门 evidence 与 capability overlay：已完成；
- overlay 显式激活、调用 profile/目标身份/runtime trace 守卫：已通过；
- 完整动态训练轮 strict-NPU：已通过，117 个调用点全部命中 NPU；
- 250 轮确定性缩小案例：已通过，29,250 次实际调用零 profile 拒绝，最终 CPU/NPU
  最大绝对误差 `4.76837158203125e-07`；
- 缩小案例 profiler、已知方向 memcpy 探针、包装层复制定位和消除后重采：已完成；
- 原始大配置、预热、重复性能样本与性能结论：未完成。

因此可以写“两个自定义算子已通过正式资格化，并在一个包含 RBF、Laplacian、损失、解析
梯度和更新的动态训练轮中显式激活；确定性缩小案例又完成了 250 轮设备驻留对拍”。不能把
这个结论扩大成“原始大配置已有真实性能收益”。原始 Artifact 仍为
`generated_unverified`，中间修复候选为
`repaired_unverified`，最终封存 Artifact 本身也仍保留 `generated_unverified`；能力是否可用
由独立的 qualification evidence 与显式 overlay 决定，而不是回写生成物状态。

## 1. 扫描与定向补接口设计

`examples/poisson_ritz_step.py` 显式声明两个领域算子：

| 缺口 | NumPy oracle | 公共目标 API | 首选路线 |
|---|---|---|---|
| `operator.apply_gradient_descent` | `var - alpha * delta` | `asnumpy.apply_gradient_descent` | `direct_generation` |
| `operator.smooth_l1_loss` | 固定 `beta=1, reduction=mean` | `asnumpy.smooth_l1_loss` | `direct_generation` |

分析阶段静态提取 oracle 源码，不导入或执行被分析文件。GapSpec 和生成请求固定调用参数、
dtype/shape、目标 `ascend910b` 和 CANN `9.0.0`，并明确禁止首轮获取外部算子实现仓库。

这和当前设计框架图一致：模型先扫覆盖集和 `OperatorSpec`，已有接口直接复用，只对缺失
接口发起生成。生成物留在本地隔离目录，只有资格化通过后才允许进入能力 overlay。

## 2. 用户 LLM 的真实调用边界

配置文件：

```text
/home/ma-user/work/.numpilot-config/llm.json
```

CLI 反馈规则已经实现并验证：

- `configure-llm --check` 只检查本地配置，不冒充远端成功；
- `configure-llm --probe` 只有拿到可验证 HTTP 响应才打印 `actual_remote_call=yes`；
- fresh generation 成功打印 `verified_new_remote_artifact=yes`；
- repair 成功打印 `verified_new_remote_repair=yes` 和回执数；
- HTTP 已成功但正文无效时，repair 也会保留成功 HTTP 回执计数并打印
  `actual_remote_call=yes`、`verified_new_remote_repair=no`；
- 调用前校验失败或无 HTTP 回执时打印 `actual_remote_call=no`；
- 复用旧 Artifact 明确打印 `actual_remote_call=no action=reused_existing`；
- 不回退到 AST 伪装 LLM 成功。

报告不保存 Token、Authorization、Base URL 或原始响应正文。保存的是请求/响应/content
SHA-256、HTTP 状态、TLS 状态、模型字段、时间和有界 usage。

## 3. 官方 CANNBot Skill 的使用

固定来源：

```text
repository: https://gitcode.com/cann/cannbot-skills.git
commit:     abcb35fc0ff77f4bc5eda74de6bbcf28cbc1572b
skill:      ops/ascendc-registry-invoke-template/SKILL.md
license:    CANN Open Software License Agreement Version 2.0
files:      19
bytes:      81,750
bundle_sha256: 879f183ad762286251027734ca173398eb01e368499627c9274aaaf75e7e4e4d
```

远端模型无法读取本机 `/home/...` 路径，所以仅告诉它 Skill 的本地路径并不可靠。当前实现
会验证仓库 origin、固定 commit 和所选文件工作树，然后按阶段把必要的官方内容直接嵌入
请求。用户可传已有 checkout；其他用户首次执行 `setup-skills` 或 `complete-gaps` 时可获取
固定版本。`git clone numpilot` 和 `import numpilot` 不会隐式联网，`--offline-skills` 缺缓存
时失败关闭。

## 4. 权威 fresh generation：direct-local-v5

权威目录：

```text
/home/ma-user/work/numpilot-gap-runs/poisson-ritz-p0-20260821/direct-local-v5
```

| 算子 | Artifact ID | 文件数 | 成功 HTTP 回执 | 状态 |
|---|---|---:|---:|---|
| `apply_gradient_descent` | `draft-0a0316cdb59682d23e46` | 17 | 17 | `generated_unverified` |
| `smooth_l1_loss` | `draft-fefdeddc2fcfc65786a4` | 17 | 17 | `generated_unverified` |

最后响应内容哈希：

```text
apply : 49ef2e53cf050f7c4265dd91a4449c46478345f0267a6b3075062ab9f1d53786
smooth: 4edbc9cc753febe819abe6a17ace194983ce1066b10c2290edafaaaafea7e536
```

两个算子均为 fresh remote，34 个文件级响应全部 HTTP 200、内容哈希绑定。随后用不存在的
LLM 配置做离线复用，两份 Artifact 均复验成功并明确报告 `actual_remote_call=no`。

静态检查确认 17 文件清单完整，无 TODO/placeholder、`ops-nn`、网络获取或系统安装引用，
只声明 float32；测试不在模块顶层导入 `asnumpy`。

## 5. 模型驱动的隔离修复闭环

原始签名 draft 从未被改写。每次 repair 都：

1. 复验原始 Artifact 的文件全集和哈希；
2. 读取有界编译/测试诊断；
3. 只允许固定 17 文件计划中的指定路径；
4. 为每个文件做一次独立模型调用；
5. 生成新目录，不在原地修改；
6. 写入 `numpilot-repair-report.json`，绑定诊断、Skill、生成请求、前后文件哈希和远端回执。

本次运行目录中共持久化 39 份成功 repair 回执；加上 fresh generation，共有
73 份可审计、内容哈希绑定的成功远程回执。这个数量包含已被后续轮次替代的
候选；两条最终谱系则共绑定 57 份成功回执。另有三次 attempt 没有形成有效回执/
候选，其中早期两个 HTTP 200 响应因不符合 FILE 信封被拒绝。CLI 会分别报告
`actual_remote_call` 和 `verified_new_remote_repair`，不把无效响应冒充成功。

修复过程暴露并闭环了这些实际问题：

- tiling data 字段与 host/kernel 不一致；
- `ExtendCfgInfo`、`ShapeEqual`、`Numels()` 等 CANN 9.0 不存在的 API；
- tiling key 没有按官方 `ASCENDC_TPL_SEL_PARAM` 选择；
- kernel 引用不存在的 `alphaNum` 和临时缓冲；
- `Axpy`、`ReduceSum` 参数形式错误；
- float tensor 使用 double 字面量；
- SmoothL1 测试期望把线性段误写成 `abs-1`；
- 一版 kernel 虽能编译，但覆盖 `bounded` 后导致公式退化为 `abs(error)`；
- SmoothL1 的内部 OpType 与 CANN 已有算子同名，需改成 NumPilot 专用名；
- kernel entry 必须与内部 OpType 生成的名称严格一致；
- CANN 9 对带数字的 CamelCase OpType 存在运行时文件名推导差异。日志实际查找
  `num_pilot_smooth_l_1_loss.json`/`num_pilot_smooth_l1loss.json`，而构建产出
  `num_pilot_smooth_l1_loss.json`；最终改用无数字的内部 OpType
  `NumPilotSmoothMeanLoss`。

最后一项说明“能编译”不是“数值正确”。NumPilot 在编译门之外保留了独立 oracle/语义审查。

为避免每轮重生成已经修好的文件，`repair-draft` 新增 `--base-candidate`：上一轮候选必须
先通过原始 Artifact、repair report、完整 17 文件哈希、无额外文件和无 symlink 校验，
新报告再记录父 repair report SHA-256 和完整候选文件映射。因此后续轮次可以只修改
新失败的文件，而不重复生成已修好的部分。

最终候选：

```text
apply:
  .../direct-local-v5/repair-round4/apply_gradient_descent
smooth:
  .../direct-local-v5/repair-round9/smooth_l1_loss
```

SmoothL1 在 P0 采用单 AIV 核完成归约并直接写标量，优先保证语义正确；多核归约和性能优化
属于后续工作，不能拿当前实现宣称性能优势。

最终候选已用 `seal-repair` 做完整父链复验并发布为新的不可变 Artifact。封存动作不调用
模型，CLI 明确报告 `actual_remote_call=no action=sealed_repair_chain`；repair provenance
作为第 18 个签名文件进入 manifest。

| 算子 | 封存 Artifact ID | 文件数 | 生成回执 | repair 回执/attempt | 最终谱系回执 | 状态 |
|---|---|---:|---:|---:|---:|---|
| `apply_gradient_descent` | `draft-d124f69b25efd0696161` | 18 | 17 | 3/3 | 20 | `generated_unverified` |
| `smooth_l1_loss` | `draft-d3a03b173efd34240c66` | 18 | 17 | 20/21 | 37 | `generated_unverified` |

封存目录：

```text
.../direct-local-v5/sealed-final/apply_gradient_descent
.../direct-local-v5/sealed-runtime-final-v2/smooth_l1_loss
```

这里的 57 份回执只计算两个最终父链；前述 73 份还包含编译/运行修复期间已被后续轮替代、但仍
留存在隔离目录中的成功响应，二者口径不能混用。

## 6. 当前验证结果

| 门 | ApplyGradientDescent | SmoothL1Loss |
|---|---|---|
| 纯 NumPy oracle tests | 10 passed | 9 passed |
| public/runtime tests | 3 passed | 4 passed |
| CMake/opbuild | passed | passed |
| host/tiling/ACLNN/proto | passed | passed |
| Ascend910B kernel binary | passed | passed |
| CPack 本地 `.run` | passed | passed |
| 私有 OPP 安装 | passed | passed |
| public `asnumpy` API | passed | passed |
| NPU numerical execution | passed | passed |
| 正式 six-gate evidence | passed | passed |
| capability overlay 发布 | passed | passed |
| 显式激活与 runtime profile guard | passed | passed |
| Poisson 单步 strict-NPU | passed | passed |
| 完整动态一轮 strict-NPU | passed | passed |
| 250 轮设备状态接力 | passed | passed |

以下 package 都从两个最终封存 Artifact 的源码重新配置和构建：

```text
apply:
  .../qualification-sealed/apply-build/custom_opp_ubuntu_aarch64.run
  bytes 657255
  sha256 dfac7a5820bc8bc72940b2aa0079d8c829e350c6abcde5356f2eed08b2463232
smooth:
  .../qualification-runtime-final-v2/smooth-build/custom_opp_ubuntu_aarch64.run
  bytes 658788
  sha256 79109fcd817864e58217387a77b1bcbfe2906bcf5f5eeff808d80f72814d970d
```

两个 package 已显式指定 `--install-path` 安装到运行目录；当前 v5 资格化使用证据树内的
新私有 vendor root，旧 `private-opp-v4` 保留作历史复验记录：

```text
.../direct-local-v5/private-opp-v4/vendors/apply_gradient_descent_custom
.../direct-local-v5/private-opp-v4/vendors/smooth_l1_loss_custom
```

v9 阶段固定 AsNumpy wheel：

```text
.../direct-local-v5/asnumpy-p0-wheel-v9/
  asnumpy-0.1.dev106+g21368abf9.d20260821-cp310-cp310-linux_aarch64.whl
sha256 260062a29fb76bb738a96eea47ae0ace2dd10c2113becb06f4c81e40e1cf8b84
```

该 wheel 通过 manifest 注册两个算子，用绝对路径 `dlopen` 每个 vendor 的
`libcust_opapi.so`，支持同一进程中多个同 SONAME 的私有算子包。公开包装会把
ArrayLike 输入转成 NPU array，`asnumpy.ndarray` 也已补上 NumPy `__array__` 协议。

旧 v6 联合测试使用最终两个封存 package 和 `private-opp-v4`，保留下列结果：

```text
28 passed in 2.15s

Apply actual/expected: [0.0, 0.0, 3.0, -2.5]
Apply max abs error:   0.0
Smooth actual:         1.0390625
Smooth expected:       1.0390625
Smooth abs error:      0.0
Smooth output shape:   []
```

v9 新增 `outer`/`matmul` 公共路径的 NPU 修复，选定 linalg 测试结果为
`14 passed, 26 deselected`，并重新执行 v5 六门资格化与完整动态样例。后续 move wheel
消除 private native result 到 public wrapper 的深拷贝，选定真机测试 `19 passed`，再以
新基础身份执行 v6 六门资格化、10 轮 profiler 和 250 轮回归。

NumPilot 仓库回归：

```text
369 passed
python -m compileall -q numpilot tests examples  # passed
git diff --check                                # passed
```

## 7. 正式资格化、激活与 Poisson 动态迭代

测试使用隔离 venv，不依赖默认 Python 环境中预装 AsNumpy。机器上的普通 shell
`LD_LIBRARY_PATH` 会先加载 CANN devlib，导致 `npu-smi` 报 `drvSetDeviceInfo` 符号错误；
本次复验使用 driver-first 库顺序，读到：

```text
NPU 7, 910B4, Health OK, HBM 2898/32768 MB
```

运行时同时把两个私有 vendor root 放入冒号分隔的 `ASCEND_CUSTOM_OPP_PATH`。
两个 public API 均实际调用了自定义 ACLNN/kernel，本轮测试无 runtime skip，也未启用
CPU fallback。

qualification 只接受带签名 `operator-artifact.json` 的封存 Artifact，不直接信任可变 repair
candidate。v9 阶段权威资格化目录为：

```text
.../direct-local-v5/qualification-evidence-v5
```

外部可信 harness 从两个封存 Artifact 重新构建 package，只安装到新的私有 OPP，执行
artifact 测试、public AsNumpy API 和真实 NPU numerical worker，再生成并验证 schema v2
六门报告。结果为：

| 算子 | Qualification ID | 测试 | 数值 case | kernel launch | 最大绝对误差 |
|---|---|---:|---:|---:|---:|
| `apply_gradient_descent` | `qual-1479c0f3e8ad5fadb6f1` | 13 passed | 3 | 3 | `0.0` |
| `smooth_l1_loss` | `qual-b0233d6345dab583d8a0` | 13 passed | 3 | 3 | `5.960464477539063e-08` |

原子发布的 overlay 为
`qualification-evidence-v5/overlay/capability-overlay.json`，SHA-256 为
`08625ef3704db848281cab041e1a7da7d69777b1d425a329c6c43cea45ff1b6e`。v1/v2 失败关闭，
v3/v4 被后续基础 manifest 取代；旧 v4 overlay 与 v9 wheel 组合会被 base drift 正确拒绝。
它们均保留作审计记录；在该阶段只能将 v9 wheel 与 v5 evidence 成对使用。

旧预计算梯度单步在 v9/v5 下仍通过，目录为 `poisson-strict-npu-v2`。随后新增
`examples/poisson_ritz_iteration.py`，把 RBF、Laplacian、损失、解析梯度和更新全部纳入
唯一 `forward`，显式加载 v5 overlay 运行：

```text
.../direct-local-v5/poisson-full-iteration-strict-npu-v2
```

结果 `status=pass`、`strategy=whole_forward_npu`，111 个 NumPy 调用和 6 个领域算子调用共
117 次改写全部通过。runtime trace、profile、目标身份和三路数值比较均通过，最大绝对误差
`9.5367431640625e-07`。第一次 v1 因学习率张量不符合单元素契约而失败关闭，修正后才生成
v2 通过结果。

同一迁移产物再由 `tools/poisson_iteration_evidence.py` 运行 250 轮：循环前 30 次 H2D、
循环后 5 次 D2H，轮间只接力设备参数；29,250 次实际公开 API 调用与 `117×250` 精确一致，
profile 零拒绝。CPU loss 从 `9.948076248168945` 降至 `6.602624893188477`，NPU 最终 loss
为 `6.602625370025635`，最终状态最大绝对误差 `4.76837158203125e-07`。权威目录为
`poisson-multiround-strict-npu-v3`；该版 receipt 还绑定了 evidence runner 自身的 SHA-256。
本次仍没有采集或发布性能数字。

Profiler 首轮记录 1,235 次 `aclrtMemcpy`。已知方向探针确认 compact 方向字段无法区分
H2D/D2D/D2H；源码和调用账目定位到每个 private native 返回值的 public 包装深拷贝。
显式 one-shot move 修复后重新资格化，同一 10 轮 workload 保持 1,170 条 kernel，复制降到
恰好 30 H2D + 5 D2H 的 35 次；机器审计 10 项检查全部通过。新 wheel SHA-256 为
`afe845c45d17e7f5cd619831f555fba3c2344296ffe02f1d251017b04cf0282d`，v6 overlay SHA-256 为
`1c5aa2ae5c4ebf2db8d542f24defd9c7ca42ccc4d87b6e10fafc60c4eb708789`。同一 move wheel 的
250 轮回归目录为 `poisson-multiround-strict-npu-v4-move`，29,250 次调用再次通过。

动态样例扫描没有发现第三个缺口；两个领域算子已经由本地 v5 overlay 准入，因此本阶段
没有再次调用 LLM，也没有拉取 `ops-nn`。只有遇到未封存/未资格化的新缺口时才应重新进入
用户 LLM + CANNBot Skill 的生成链，并按 CLI 的 `actual_remote_call` 字段报告真实调用结果。

## 8. 两次隔离事件

### 8.1 系统安装器误触

曾误把官方 `.run --check` 当作只读校验，安装器实际尝试写系统 OPP。发现后立即停止，并把
本轮新增目标可恢复地移动到：

```text
/home/ma-user/work/numpilot-gap-runs/poisson-ritz-p0-20260821/unintended-system-install-quarantine-20260821
```

已确认以下目标不存在：

```text
/usr/local/Ascend/cann-9.0.0/opp/vendors/custom_nn_nn
/usr/local/Ascend/cann-9.0.0/opp/vendors/apply_gradient_descent_custom
/usr/local/Ascend/cann-9.0.0/opp/vendors/smooth_l1_loss_custom
/usr/local/Ascend/cann-9.0.0/opp/vendors/config.ini
```

隔离内容没有删除。后续仅对已核对的最终 package 执行安装器，并显式使用
`--install-path` 安装到 `direct-local-v5/private-opp-v4`。安装前后都复查了上述系统
路径，未发现新的系统 CANN 写入。

### 8.2 原始 draft 的 pytest 缓存

早期曾直接在签名 draft 中运行 pytest，生成 `.pytest_cache` 和 `tests/__pycache__`。源文件未
被修改；缓存被可恢复地移动到：

```text
.../direct-local-v5/diagnostic-quarantine-20260821
```

之后对两份签名 Artifact 做了完整离线复验。后续测试统一使用
`PYTHONDONTWRITEBYTECODE=1` 和 `-p no:cacheprovider`，构建目录也放在候选之外。

## 9. 下一步优先级

1. **已完成**：对动态迁移产物运行 profiler，以已知方向探针验证采集字段，定位并消除
   private native result 到 public `ndarray` 的逐返回值深拷贝；同一 10 轮 workload 的
   `aclrtMemcpy` 从 1,235 降至 35，1,170 条 kernel 不变；
2. 选择接近原始案例的更大规模，运行预热、重复样本与收敛/数值门禁；
3. 只在数值、fallback、传输和 profiler 门通过后生成性能结论；
4. 将当前 Poisson 专用可信 harness 收敛成可复用的多算子 qualification runner；
5. CodeArts 保持后置；用户决定切换时，再保留需求、生成、调试、测试和 Diff 证据。

## 10. 复现入口

```bash
NUMPILOT_LLM_CONFIG_FILE=/home/ma-user/work/.numpilot-config/llm.json \
python -m numpilot configure-llm --check

python -m numpilot setup-skills \
  --root /home/ma-user/work/cannbot-skills --check

NUMPILOT_LLM_CONFIG_FILE=/home/ma-user/work/.numpilot-config/llm.json \
NUMPILOT_LLM_TIMEOUT=300 \
NUMPILOT_LLM_MAX_TOKENS=4096 \
python -m numpilot complete-gaps examples/poisson_ritz_step.py \
  --out /path/to/new-empty-output \
  --target-soc ascend910b \
  --cann-version 9.0.0 \
  --provider llm \
  --max-attempts 3 \
  --skill-root /home/ma-user/work/cannbot-skills \
  --offline-skills \
  --asnumpy-reference /home/ma-user/work/numpyexample

# 后续轮只修新的失败文件；--draft 始终指向原始签名 draft。
python -m numpilot repair-draft \
  --spec /path/to/gap-spec.json \
  --draft /path/to/original/draft \
  --base-candidate /path/to/previous-repair-candidate \
  --out /path/to/new-repair-candidate \
  --generation-request /path/to/GENERATION_REQUEST.md \
  --diagnostic-file /path/to/compiler-diagnostic.txt \
  --path op_kernel/example_arch22.cpp \
  --skill-root /home/ma-user/work/cannbot-skills \
  --offline-skills

# 最终 repair 链通过构建/审查后封存；此命令不调用 LLM。
python -m numpilot seal-repair \
  --spec /path/to/gap-spec.json \
  --draft /path/to/original/signed/draft \
  --candidate /path/to/final-repair-candidate \
  --out /path/to/new-sealed-artifact

# 在新的不存在目录中，从两份封存 Artifact 重新构建、真机资格化并发布 overlay。
python tools/poisson_qualification_harness.py \
  --source examples/poisson_ritz_step.py \
  --asnumpy-python /path/to/asnumpy-venv/bin/python \
  --asnumpy-wheel /path/to/asnumpy-wheel.whl \
  --apply-spec /path/to/apply/gap-spec.json \
  --apply-artifact /path/to/sealed/apply_gradient_descent \
  --smooth-spec /path/to/smooth/gap-spec.json \
  --smooth-artifact /path/to/sealed/smooth_l1_loss \
  --out /path/to/new-qualification-evidence

# 激活必须显式给出 overlay 和 evidence root；无效、漂移或不匹配时失败关闭。
python -m numpilot migrate examples/poisson_ritz_step.py \
  --backend npu --engine ast --out /path/to/poisson-strict-npu \
  --skip-env-check \
  --capability-overlay /path/to/qualification-evidence/overlay/capability-overlay.json \
  --qualification-evidence-root /path/to/qualification-evidence
```
