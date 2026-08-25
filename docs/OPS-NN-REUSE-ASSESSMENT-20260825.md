# ops-nn 官方算子复用评估（P1 待定）

> 结论日期：2026-08-25 UTC<br>
> 当前结论：需要纳入架构，但不作为 Demo v1 默认安装依赖，也不宣称已经实现。

## 1. 结论

当 NumPilot 已确认 `GAP_REQUIRED` 时，应先查询华为官方
[`cann/ops-nn`](https://gitcode.com/cann/ops-nn)，再决定是否让 GLM 生成新的 Ascend C
kernel。命中官方实现时，目标流程变为：

```text
GapSpec
  → 匹配 CANN 版本的 ops-nn 目录快照
  → 精确语义 / dtype / shape / SoC / license 审计
  → 仅构建被选算子到私有 OPP
  → GLM 生成 AsNumpy adapter
  → 原有六门 qualification
  → capability 激活、重扫、GLM 最终迁移
```

只有不存在精确匹配，或官方实现不能满足当前 GapSpec 时，才回到现有
“CANNBot Skill + GLM 生成新 kernel”的路径。这样可以减少重复开发，但不会把“同名算子存在”
误写成“AsNumpy 已经支持”。

## 2. 为什么值得做

官方仓库已经提供算子源码、host/tiling/kernel、部分 ACLNN API、样例和测试工程，并支持用
`build.sh --pkg --soc=... --ops=...` 只构建指定算子。官方文档同时明确：源码必须选择与 CANN
版本配套的 tag/分支，使用 `master` 可能不匹配；自定义算子包默认安装到
`${ASCEND_HOME_PATH}/opp/vendors`。

对 NumPilot 已有 Poisson 纵切，CANN 9.0.0 分支的官方算子列表中已经能找到两个值得进一步
审计的候选：

- `optim/apply_gradient_descent`：官方说明的核心公式是
  `var = var - alpha * delta`，但当前 9.0.0 算子表未标记 op_api，README 只列图模式调用；
  因此它不能被描述成可直接接 AsNumpy 的 ACLNN 算子；
- `loss/smooth_l1_loss_v2`：官方实现带阈值参数和 `none/mean/sum` reduction，并提供 ACLNN
  调用说明，更适合作为第一个复用 spike。

它们只是候选。仍需逐项核对 mutation/返回值、公开签名、默认值、广播、dtype、shape、布局、
错误行为和当前 OperatorSpec。例如官方 `smooth_l1_loss_v2` 的 ACLNN 文档对 `mean/sum` 给出
`[1]` 输出，而现有 NumPy oracle 的 `np.mean(...)` 是 0-D scalar；adapter/contract 必须明确
规范并数值对拍。`apply_gradient_descent` 又同时把 `var` 描述为输入和输出，而现有 oracle 是
非原地返回，必须审计 mutation/alias。不能据此追溯性地改变已经封存的 Poisson 证据。

## 3. 必须保留的门禁

1. **路由门**：只有 missing API 或用户拒绝的 audited fallback 能进入；
   callable + `UNKNOWN` 仍先做 capability/source audit。
2. **版本门**：由当前 CANN 版本解析匹配分支/tag，再固定 official origin、commit、目录清单和
   文件 hash。2026-08-25 核对的 CANN 9.0.0 分支头为
   `fcebf031d193d641d2d1472a539bcc387b1e5f09`；它只是本次评估快照，正式实现应写入 lock
   manifest，而不是跟随 `master`。
3. **许可证门**：仓库使用 CANN Open Software License Agreement Version 2.0。下载、使用或
   构建前必须由用户显式接受，并把接受状态写入 receipt；不能由 pip hook、`import`、
   `doctor` 或 `migrate` 偷偷拉取。
4. **语义门**：名称命中不够，必须证明 GapSpec 与官方算子的输入/输出、原地语义、属性、
   dtype、shape、broadcast、layout 和异常边界一致。
5. **构建门**：只构建选中算子，先落到仓库外隔离目录和私有 OPP；不能直接覆盖系统 CANN，
   必须有安装清单、hash 和回滚路径。
6. **生成约束**：官方 kernel 可以复用，但所有为 NumPilot 新增的案例、迁移和 AsNumpy adapter
   代码仍必须由 GLM 生成，不能人工补写。
7. **资格化门**：仍需通过 `artifact_integrity`、`build`、`unit_test`、
   `numerical_validation`、`public_asnumpy_api`、`npu_execution` 六门，并绑定
   SoC/CANN/runtime/artifact 身份；`activation` 是六门通过后的独立状态，不能混入资格门计数。
8. **来源门**：现有 qualification 只接受已封存 GLM Artifact，不能把官方源码冒充成
   `remote_llm`。实现 resolver 前必须增加 hash-bound `official_source` Artifact/manifest，
   并让 adapter 与 qualification validator 显式接受、审计这种来源。

## 4. 为什么不放入默认 bootstrap

`ops-nn` 体积和构建链都明显重于普通 NumPilot 安装，并与 CANN/SoC 强绑定。官方构建说明还
区分联网与离线依赖准备；联网构建可能下载第三方软件，CANN 9.0.0 及以后还需要匹配的
`opbase`。因此不应在 `pip install numpilot`、`import` 或普通 direct migration 中整库 clone、
全量构建或写系统 OPP。

更合适的是以后增加一个显式、按需、可审计的 reuse resolver。CLI 名称尚未冻结；设计示意为
`resolve-gap --provider ops-nn`，但当前版本没有该命令，文档和架构图必须继续标记为 P1 待定。
resolver 应优先使用版本化本地 catalog/cache，只在用户显式接受许可证与网络动作后获取固定
快照，并只准备命中的算子。

## 5. Demo v1 决策与验收条件

Demo v1 只完成以下事项：

- 在架构和 10 天计划中把 ops-nn 放在“新 kernel 生成之前”；
- 记录官方来源、版本配套、许可证和定向构建约束；
- 保留现有 CANNBot/GLM 生成路径，不改动已封存 Case 01 或 Poisson 产物；
- 不增加默认下载、不宣称 resolver 已实现。

进入实现前需要满足：

- 选定一个真实 GapSpec，完成官方候选的逐字段语义对照；
- 确定私有 OPP 的安装、发现、卸载与回滚方案；
- 确定 AsNumpy adapter 的 GLM 生成契约和六门 evidence schema；
- 增加不伪装成 GLM 的 `official_source` Artifact/manifest，并接入 adapter/qualification
  validator；
- 在干净环境验证“命中官方候选”和“无匹配后转 GLM 生成”两条失败关闭路径。

## 6. 官方依据

- [ops-nn 仓库](https://gitcode.com/cann/ops-nn)
- [版本配套与项目说明](https://gitcode.com/cann/ops-nn/blob/9.0.0/README.md)
- [算子列表](https://gitcode.com/cann/ops-nn/blob/9.0.0/docs/zh/op_list.md)
- [算子编译、安装及离线依赖说明](https://gitcode.com/cann/ops-nn/blob/9.0.0/docs/zh/invocation/quick_op_invocation.md)
- [CANN OSL v2.0](https://gitcode.com/cann/ops-nn/blob/9.0.0/LICENSE)
