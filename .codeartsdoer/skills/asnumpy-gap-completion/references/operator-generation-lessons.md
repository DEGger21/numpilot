# AsNumpy 算子接入生成经验

这份记录来自 NumPilot Phase 18.20B-8 的真实生成实测，作为
`asnumpy-gap-completion` 的复用经验。它描述的是已观察到的工程事实和
门禁边界，不把模型输出当成正确算子。

## 已验证事实

- `llm.json` 的 OpenAI-compatible DeepSeek endpoint 可以连通；readiness 和
  generation 请求均取得 HTTP 200，模型身份可由远端回执校验。
- HTTP 200、模型身份正确、token 使用量存在，都不能证明生成了可用的
  AsNumpy overlay。只有本地协议解析通过，才允许进入
  `generated_unverified`。
- 旧的 `NUMPILOT_EXACT_EDIT_V1` 要求模型同时复制 `OLD` 原文、组织多层
  `BEGIN_EDIT/END_*` 标记并保持空行精确。DeepSeek 三轮实测共 17 次请求，
  在 `EDIT_COUNT` 或 `NEW block` 阶段失败，没有发布 overlay。
- 失败时必须封存 request/response hash、远端 receipt、失败原因、上下文
  计划 hash 和所有 false gates；不能把 readiness、失败响应或候选文件当成
  成功。

## 当前协议

官方 existing-module 路由使用 `NUMPILOT_WINDOW_REPLACE_V1`。模型只输出
当前目标文件的 reviewed window：

```text
NUMPILOT_WINDOW_REPLACE_V1
PATH: exact/target/path
BASE_SHA256: full-target-sha256
WINDOW_COUNT: N
BEGIN_WINDOW 1
LABEL: reviewed-window-label
CONTENT:
complete replacement for this reviewed window
END_WINDOW 1
END_WINDOW_REPLACE
```

NumPilot 本地重新计算窗口范围和 hash，要求 label、顺序、数量、整文件
`BASE_SHA256` 全部匹配，并且重建后的差异只能落在审计过的窗口中。旧
exact-edit parser 保留用于历史回执重放和修复兼容；新 generation lane 默认
使用窗口协议。

## 交互和失败处理

1. 终端先显示 official resolution、context gate、AsNumpy context plan 和
   `plan_sha256`，再显示模型身份和本次请求边界。
2. 每个目标文件显示 `file i/n`、尝试次数、解析结果和 receipt 数量；模型
   正在生成时用心跳保持终端可见。
3. 解析失败只进入下一次有界重试；成功前不写正式组件、不 build、不 import、
   不执行 NPU、不推进 `ADAPTER_CANDIDATE`。
4. 所有尝试完成后输出隔离的
   `asnumpy-adapter-generation-failure.json`，并将阶段标记为 `blocked`。
5. 需要换用 Codex 等更强代理时，代理应读取同一份 context plan 和窗口
   协议，生成本地候选后再交给 NumPilot 做同样的机械校验；代理不能绕过
   RecoveryStore、GapSpec、manifest、ACLNN 两段式和后续数值/性能门禁。

## 继续开发的验收条件

- 协议解析正负例通过，窗口外修改、label 漂移、hash 漂移、占位符和 no-op
  均 fail-closed。
- 真实模型只有在 commentary 预告后才可再次请求；请求结束必须有可复核的
  failure/success receipt。
- 结构化候选进入 `generated_unverified` 后，仍必须依次通过 build、import、
  NumPy oracle、真实 NPU execution、performance 和 qualification，才能计入
  benchmark 迁移结果。

## 本地 Codex 实测：官方 concatenate 接入

- 查实现再判断缺口：固定新版 AsNumpy 已含 `Concatenate` 的设备拷贝实现，
  Python export 也已存在。静态覆盖表中的缺口不等于源码完全没有实现。
  新 ACLNN adapter 使用 `ConcatenateAclnn` / `concatenate_aclnn` 独立入口，
  保留旧 native 接口给 `hstack/vstack`，避免重复符号和连带收窄其他 API。
- 不为满足“每文件必须修改”制造注释。`__init__.py` 的既有 lazy export 可以
  按窗口协议原样返回；hash、label、顺序和边界仍须验证，整份候选不得无改动。
- `aclDestroyTensorList` 会释放列表内的 tensor 描述符。不要把输入 NPUArray
  自有的 `tensorPtr` 交给一个会被销毁的列表；为列表新建指向同一设备缓冲的
  元数据描述符，并明确成功与异常路径的所有权，避免双重释放。绑定层传引用，
  不用 `vector<NPUArray>` 隐式深拷贝充当零拷贝接入。
- 数值正确不证明路由正确。首次集成因 wrapper 错用 `_concatenate_legacy`，
  NumPy 对拍全部通过却未进入新 ACLNN。修复后 NumPilot 自有 gate 会验证
  `_concatenate` 对应 `concatenate_aclnn`，统计每次调用，并禁止旧路径。
  生成者自己的测试之外，应有框架独立的路由断言。
- 新 `official_candidate_gap` run 可明确选择 `--profile full-local-recovery`；
  已存在 receipt-only run 的策略保持冻结，不能直接改状态 JSON 来升权。
  项目作者已经授权本地构建时，记录对应 profile 后执行，不反复要求口头确认。
- 固定 AsNumpy bundle 可包含未提交的作者 matmul 补丁。隔离构建可逐文件验证
  component/source manifest 后复制精确文件集，不依赖新的 Git 提交，不带入
  untracked 文件；transaction 同时记录组件、archive、source manifest 哈希。
- 精简 official-candidate fixture 与原始 I006 是两个不同 source。对原始 I006
  的直接 API smoke 需保存 source hash、原输入和变换方式，不能冒充 NumPilot
  自动迁移完成。构建/对拍也不能替代 rank、provenance 和 qualification 适配。


## 正式资格与原始样例迁移实测

- `official_candidate_gap`必须按其原始schema验证，不改名成script_api_gap。
  搬移证据时携带原source/resolution字节和hash，从内容重建GapSpec/decision；路径只作身份标签。
- 隔离runtime可在sys.prefix外，但必须逐项匹配transaction、wheel、解释器、core、wrapper和source。
  明示overlay只准入已审计的调用范围，不把整个AsNumpy的audit_required改成matched。
- concatenate的输入是tensor sequence加axis，不能用通用参数个数匹配。非法输入不得退回
  一个看似完整的普通profile；matcher也必须要求sequence_axis，防单ndarray或tensor axis误准入。
- Python native调用次数不是kernel次数。使用msprof原CSV，检查ConcatD、device/task、正的
  有限耗时和runtime绑定；独立profiler4条kernel与另一轮gate2次native分别保存。
  msprof启动带-S的Python可用独立launch.sh，避免把-S识别为application路径。
- 正式qualification和原始程序迁移分开保存。API资格可以来源于精简fixture；原始I006
  则必须通过Orchestrator真实生成代码、strict NPU和原始NumPy oracle，保存source/hash/guard。
  本次I006成功不等于150例全部成功，也不证明性能加速。
- 复现入口：仓库docs/codex-adapter-walkthrough.zh-CN.md；profiler、qualification和migration
  输出均使用新目录，失败尝试保留。固定组件与共享runtime不被试验产物覆盖。

## Vitamin C 真实科学工作流经验（2026-11-11）

- 真实样例应从可追溯的分子几何和上游科学代码开始。本次用 GPU4PySCF 的 Vitamin C 20 原子几何执行 RKS/PBE/cc-pVDZ SCF，再从收敛结果生成真实 DFT 网格输入；不能用随机矩阵结果代替科学真实性。
- `dm.T` 常常不是 C-contiguous。进入已准入的 AsNumpy 二维 `float32` matmul 前，显式 `astype(np.float32)` 和 `np.ascontiguousarray`，并在 taskbook/receipt 中记录 dtype、rank、布局和形状。
- 把 SCF/网格构造、热点 contraction、rho 归约分开计时。驻留态和暖态端到端必须分别报告，不能把 H2D/D2H 隐藏在“算子耗时”里。
- 数值验收同时检查投影矩阵误差、电子数差和输出哈希；当前最大绝对误差为 `9.54e-06`，电子数差为 0。
- 演示可以先用人工/Codex 完成已知 matmul 路由，但必须明确这不是 AI 自动生成缺口算子的证据；若后续接入生成模型，仍需沿用相同 taskbook、strict-NPU gate 和独立对拍。
- 目前仅把 `dm.T @ ao` 放到 NPU，`rho` 归约仍在 CPU；不要把 20.8x 解释成完整 SCF 加速。
