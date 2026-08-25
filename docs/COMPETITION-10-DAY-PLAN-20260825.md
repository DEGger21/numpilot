# NumPilot 初赛 10 天冲刺计划

> 起点：2026-08-25
> 代码冻结：2026-09-03（D10）
> 产品目标：先交付“普通 NumPy 文件能安全分类并迁移”的 Demo v1，不扩张成通用编译器

## 1. 冻结后的默认用户流程

```text
全新机器：install.sh（显式 --allow-network）
手工路径：numpilot bootstrap（默认只读 plan）/ --apply
                    │
                    ▼
用户给一个纯 NumPy .py 文件（import numpy as np）
                    │
                    ▼
       NumPilot 静态扫描，不执行源码
                    │
                    ▼
         五类确定的 capability 结果
      ┌─────────────┼───────────────────────┐
      ▼             ▼                       ▼
   DIRECT   FALLBACK_CONSENT   CAPABILITY_AUDIT_REQUIRED
      │         │      │           callable + UNKNOWN / 身份漂移
      │       接受    拒绝                    │
      │         │      └──────┐              ▼
      │         │             ▼       资格或源码审计；不生成算子
      │         │       GAP_REQUIRED
      │         │       missing / rejected audited fallback
      │         │             │
      └─────────┴──► GLM      ▼
          精确迁移       GapSpec → P1 官方 ops-nn 复用候选
               │                   │ 无精确匹配
               ▼                   ▼
          AST/回执门禁       GLM kernel/adapter 草稿
                                  │
                                  ▼
                         下一阶段：integrate/qualify/activate/rescan

BLOCKED：动态或未建模表面，结构化记录原因并停止
```

`migrate` 默认且推荐 ordinary script。`build_inputs + forward` 只有显式 `--mode contract` 才
进入 legacy 严格通道；`--mode auto` 只保留兼容性，新的 Demo 与比赛脚本不得依赖自动探测。

普通迁移不执行用户代码，也不发布数值或性能结论。产品内 ordinary 通用 benchmark 尚未实现；
本轮真实 NPU 数值/性能由外部受信、与原文件/GLM artifact/runtime/harness 哈希绑定的定向流程
验证。`check-suite`、`wizard`、`bench-cpu`、`bench-npu` 仍是 legacy contract 工具。

## 2. 十天交付目标

1. `/home/ma-user/work/demotest/demo` 的 10 组案例都得到稳定、可解释的五类结果；
2. direct 案例由 `glm-5.3` 生成完整迁移文件，并通过 source-bound AST、注释、模型和回执门；
3. fallback 案例具有清楚的 accept/reject 交互，非交互环境绝不默认同意；
4. missing 与 rejected audited fallback 能产生 canonical GapSpec；callable + `UNKNOWN` 不被盲目
   路由到算子生成；
5. 在新 kernel 生成前增加 ops-nn 官方复用决策；本轮至少完成一个
   “GapSpec → 官方候选审计或未验证 GLM 算子/adapter 草稿”的安全纵切，并明确展示 resolver
   与自动恢复编排尚未完成；
6. 依赖只能通过显式 bootstrap 计划与应用，校验锁定的 official
   AsNumpy/CANNBot 身份；只有获取或复验 CANNBot 内容时记录显式 CANN OSL 接受；
7. 形成评委可运行的 CLI、draw.io 架构图、测试报告、演示视频和发布包；
8. 所有**新编写**的案例、迁移、算子和 adapter 代码由 GLM 生成；复用的官方 ops-nn 源码必须
   保持 official provenance/hash/license，不得冒充 GLM 生成。NumPilot 只做扫描、门禁、执行和记录。

## 3. Bootstrap 交付约束

已落地的命令面为：

```bash
# 默认与 --plan 都是只读；--json 可供自动化消费
python -m numpilot bootstrap
python -m numpilot bootstrap --plan
python -m numpilot bootstrap --plan --json

# 全新机器默认：只克隆 NumPilot，从锁定源码构建 AsNumpy 并准备 CANNBot
./install.sh --allow-network \
  --with-cannbot --accept-cannbot-license

# 等价手工路径
python -m numpilot bootstrap --apply --build-asnumpy \
  --with-cannbot --accept-cannbot-license

# 可选历史路径：可信本地 wheel + expected hash
python -m numpilot bootstrap --apply \
  --asnumpy-wheel /trusted/path/asnumpy.whl \
  --asnumpy-wheel-sha256 "$TRUSTED_ASNUMPY_SHA256"

# 安装后 AsNumpy smoke；可选固定本地 CANNBot Skill 清单
python -m numpilot bootstrap --verify
python -m numpilot bootstrap --verify --with-cannbot --accept-cannbot-license
```

- 无模式或 `--plan` 只读环境和动作计划，不 clone、fetch、build 或 install；
- `--apply` 必须在“`--asnumpy-wheel PATH` + `--asnumpy-wheel-sha256 HEX`”和
  `--build-asnumpy` 之间二选一；
- official AsNumpy source commit/tree 和 CANNBot commit/bundle 身份固定在实现中；
  `--asnumpy-cache PATH`/`--cannbot-cache PATH` 只显式改变 checkout 缓存位置；
- 实现不内置 wheel URL/hash。本地 wheel 的 expected SHA-256 由可信发布物/用户
  输入提供，校验后将实际 hash 写入 receipt；资格 profile 仍在 capability 层绑定；
- 全新机器 Demo 默认只 clone NumPilot，由 `install.sh` 在项目 venv 安装
  `.[bootstrap]`、从锁定源码构建 AsNumpy，并拉取/校验 CANNBot；不使用
  wheelhouse，不运行 sudo，不修改 shell 配置；
- 源码构建子进程尚无 OS 级断网沙箱，plan 必须保守标记 build 为 `network=true`；
- 当前锁定官方源码的公开 `matmul` 是 host NumPy fallback；接受该路径时只验收
  安装、兼容性和数值，必须标记 `strict_npu=false`、`performance_valid=false`，不复用
  历史 P0 wheel 的性能结论；
- `--verify` 必须真实执行 AsNumpy `ap.ones((1000, 1000), dtype=ap.float32)` smoke；带
  `--with-cannbot` 时还要从固定本地 bundle 输出“现在写 Ascend C 算子有哪些能参考的 Skill？”
  的已校验清单，不联网、不调用模型；
- 只有显式 `--with-cannbot` 获取或复验 CANNBot 内容时才必须同时给出
  `--accept-cannbot-license`；单独安装 AsNumpy 不要求 CANN OSL；
- pip 安装钩子、`import`、`doctor`、`migrate`、`complete-gaps` 均不得隐式下载依赖；ordinary
  `migrate` 调用已配置 GLM 是明确的生成动作，必须留存回执；
- 不在计划或文档中写尚未存在的 wheel URL。

## 4. 每日安排

| 日次 | 日期 | 重点 | 当日交付与验收 |
|---|---|---|---|
| D1 | 08-25 | 简化架构与主通道 | ordinary scanner、版本绑定 capability、GLM 精确迁移、fallback 同意门、三页 draw.io；测试以结构化报告为准，不在计划中固化易漂移总数 |
| D2 | 08-26 | 入口与依赖收口 | 默认 ordinary、显式 legacy contract、bootstrap plan/apply/receipt + 锁定身份；CANNBot 许可证条件门；禁止隐式联网 |
| D3 | 08-27 | 10 组测试集首轮 | 用户逐例确认；先扫描/分类，再迁移 direct；保存 GLM 回执、source/response/artifact hash |
| D4 | 08-28 | 修 direct 主路径 | 只修测试集需要的 operator/method/错误提示或 capability 误判；不增加通用图 IR |
| D5 | 08-29 | fallback 与 capability audit | 验证 ask/accept/reject；分开 GAP_REQUIRED 与 callable+UNKNOWN 的审计路径 |
| D6 | 08-30 | 一个 gap 安全纵切 | 只选一个缺口，先对照匹配版本的 ops-nn；精确匹配则复用官方 kernel、由 GLM 生成 adapter，否则才由 GLM 生成 operator/adapter candidate；状态保持 unverified |
| D7 | 08-31 | 恢复闭环下一阶段 | 尝试把现有原语串成可恢复的 integrate/qualify/activate/rescan；未完成则如实保留阶段状态，不宣称自动闭环 |
| D8 | 09-01 | 端到端回归与 CodeArts | 10 例分类/迁移矩阵、无静默 fallback、全量测试；在真实 CodeArts 保存任务→GLM→Diff→测试证据 |
| D9 | 09-02 | RC 与演示冻结 | README、架构图、依赖 lock、许可证、AI 生成说明、报告、PPT/视频候选；只修阻塞问题 |
| D10 | 09-03 | 最终提交 | 空目录复现、ZIP 回读与 SHA-256、最终 PPT/视频、双份备份和上传校验 |

## 5. 测试集执行顺序

用户允许开始真实迁移后，每个 NumPy 文件固定按以下顺序处理：

1. 只读源码与惰性静态扫描；
2. 输出 API、capability provenance 及五类结果；
3. `DIRECT` 才调用迁移 GLM，并检查完整回执与精确 AST；
4. `FALLBACK_CONSENT` 等待用户策略，不代替用户选择；
5. `GAP_REQUIRED` 只写 gap request，先不调用迁移 GLM；后续优先查询匹配版本的 ops-nn，
   resolver 未实现前只记录候选，不自动下载、构建或安装；
6. `CAPABILITY_AUDIT_REQUIRED` 保存运行时身份和审计理由，不盲目生成替代算子；
7. `BLOCKED` 保存静态原因并停止；
8. 每例保存 source/model/response/artifact hash、状态和失败原因；
9. 当日最后跑定向测试、全量测试与 `git diff --check`。

当前进度以 `docs/AI-HANDOFF.md` 的最新检查点和仓库外 source-bound run 为准；不要在计划中
复制易漂移的测试总数或把旧 run 当成当前产物。

当前已有一个可公开的 ordinary 定向 evidence 示例：
[Demo v1 Case 01 精选证据](../evidence/demotest-case01-20260825/README.md)。它是外部受信、
哈希绑定 harness 的结果，不是产品内 ordinary 通用 benchmark；稳态和冷启动口径分开，且
没有把 ACLNN 日志写成 kernel-profiler 证明。

## 6. 范围与止损线

### 本轮做

- 顶层 `import numpy as np` 的普通纯 NumPy文件；
- 精确 namespace 迁移与少量 source-bound 必需改写；
- 五类 capability/fallback 决策；
- 显式、lock-bound bootstrap；
- 一个 gap 的安全纵切；
- ops-nn 只作为显式、版本/许可证/语义绑定的 P1 复用候选，不是默认 bootstrap 依赖；
- 外部受信、哈希绑定的 ordinary artifact 定向 NPU evidence；
- legacy contract 工具仅作为显式兼容/证据工具。

### 本轮不做

- Torch、JAX、Pandas、SciPy 或第三方 import 的混合迁移；
- 通用控制流图、CPU/NPU 自动切图、任意 Python 动态特性；
- 为所有 ndarray dunder/method 一次性建立通用 IR；
- 同时生成多个新算子；
- 自动整库拉取/全量构建 ops-nn，或直接写系统 `${ASCEND_HOME_PATH}/opp/vendors`；
- 在普通脚本迁移阶段执行不可信用户源码；
- 产品内普通脚本通用 benchmark；
- 把 shim、API 存在、GLM 回复、unverified draft、adapter candidate 或“编译通过”单独写成
  NPU 可用/加速证据。

若 D4 结束时 10 组案例仍不能稳定分类，立即停止算子扩张，只保证 direct + fallback 同意门。
若 D7 未串起自动恢复流，演示只展示现有原语和准确阶段状态，绝不伪装成一键闭环。

## 7. 每日看板

| 指标 | 目标 |
|---|---|
| 静态扫描覆盖 | 10/10 有确定结果 |
| direct 迁移 | 所有 direct 案例均有 GLM 回执 + AST pass |
| 静默 fallback | 0 |
| callable+UNKNOWN 被盲目生成为 gap | 0 |
| 隐式依赖联网 | 0 |
| 人工改写案例/迁移/算子代码 | 0 |
| 不实 strict-NPU/性能/自动闭环声明 | 0 |
| 全量测试 | 每日通过，结果写入当日结构化记录 |
| 演示阻断问题 | D9 前为 0 |

## 8. 每日收工命令

```bash
git status --short --untracked-files=all
python -m pytest -q
git diff --check
```

真实 GLM 调用只在运行进程临时配置用户指定代理；Token、代理设置和运行产物不得写入仓库。
