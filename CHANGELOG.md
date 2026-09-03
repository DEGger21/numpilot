# 变更记录

## 0.3.0a1.dev0 - Unreleased

- 增加 6 个显式 `OperatorSpec` 自定义算子案例（逐元素、稳定分支、归约、LayerNorm），并补充
  AsNumpy 固定源码的 facade → pybind `_core` → C++/ACLNN 封装底稿；文档明确 lazy export
  不是 kernel registry、host-assisted provenance 与 `generated_unverified` 边界；
- 增加 LLM 手写算子批测脱敏报告：6 个真实远程 case 的完整 draft 成功率为 `0/6`，单 case
  约 `60.6–61.0s`、HTTP 504；同步补齐 CodeArts Agent 重跑口径和截图要求，避免把连通性
  探针或模型片段包装成算子接入成功；
- 增加显式 `--allow-custom-operator-generation` draft-only 路由：仅静态绑定的 custom
  `OperatorSpec` 可用，收据保留 `official_equivalence=not_proven`，不伪装官方 clean no-match，
  且仍受 LLM/Skill/代理/六门资格化门禁；
- `complete-gaps` 新增显式 `--continue-on-error` 多缺口调度模式：单个 Gap 的模型/本地校验失败
  不再丢失同批次其余 Gap 的准备结果；`gap-report.json` 保留脱敏 `generation_failures`、远端
  调用状态和 HTTP 状态，失败项只保留 GapSpec/Generation Request，默认行为仍为 fail-fast，
  且含失败时 CLI 保持非零退出；
- 增加统一的 TTY 交互层：空命令显示产品入口，ordinary 分析/迁移/验证/性能阶段显示可清除的
  spinner 与持久耗时结果，`doctor`/`paths` 使用宽屏表格/窄屏列表的自适应 Rich 视图；JSON、单路径、重定向、`NO_COLOR`
  和 `TERM=dumb` 均保持无动态控制符的纯文本。模型检查新增脱敏代理路由状态，强制代理缺失或
  命中 bypass 时在联网前给出可操作错误；demo 失败会报告阶段、退出码、耗时和保留的运行目录；
- 比赛就绪度对 hosted video 页面链接（含常见视频域名、Markdown/HTML 视频标签）做保守识别，
  仍将未提供 `accessible=true` 回执的链接标为未独立验证，不把普通文档链接误当视频证明；
- 将公开首页整理为英文 `README.md` 与中文 `README.zh-CN.md`，增加双语安装/验收指南、按代码
  边界更新的双语 SVG 架构主图，以及保持既有视觉风格且待 PowerPoint/WPS 人工视觉核查的
  `docs/numpilot-en.pptx` / `docs/numpilot-zh.pptx`；同步收敛双分支 PR 发布约定；
- 增加可选的 `NUMPILOT_LLM_REQUIRE_PROXY=1` fail-closed 门禁：真实模型请求必须命中与
  endpoint 协议匹配的标准 HTTP(S) 代理，缺失代理或命中 `NO_PROXY` 时在联网前终止；每次
  请求使用单次环境快照完成 proxy/bypass 判定并构造固定代理 opener，代理地址继续不进入配置、
  日志与回执；
- 长期分支治理规范收敛为 `main`（发布、比赛交付）与 `develop`（日常集成），完善无 force
  push 的 `develop → main → annotated tag` 发布门禁；服务端保护规则仍需管理员验收；
- 增加 ordinary `benchmark-script`（`bench-script` 短别名）：只接受真实 configured-model
  `migrated_strict_npu` 产物，以显式 marker 划分一次输入准备和 steady resident 计算区；CPU/NPU
  独立 worker 在导入 NumPy 前固定到同一逻辑核，实测 CPU BLAS 单线程，并以 runtime identity、
  profile target、精确调用次数、逐样本前后同步、最终一次 D2H 和 named-output 数值门失败关闭。
  combined receipt 保留全部样本、中位/p10/p90、保守比与 bootstrap 95% 区间，并把输入准备、
  resident、最终 D2H 组件和明确标为非直接端到端采样；
- 将根 `run_demo.sh` 从已经失配的 legacy/AST 路径改成 ordinary 分阶段演示链：
  `doctor → analyze → configured model migrate → verify-script → benchmark-script`，新增默认 9M
  `float32` Poisson-Ritz 源项/鲁棒残差阶段、唯一 run 目录、未知参数失败和 shell 编排回归；
  五步结束后只读四份 JSON 回执生成一屏评委摘要，明确分开 resident headline 与非直接
  E2E 的一次组件耗时；
- 增加两份可随源码复核的 ordinary 真机证据：应用形态 Poisson 阶段在同核、CPU BLAS
  单线程、20 样本下 resident 中位加速 `2.567×`（保守 `2.471×`），但一次组件总耗时仅
  `0.789×`；合成逐元素链 resident 中位 `8.079×`（保守 `7.503×`），组件和 `1.220×`。
  两者均由实际 `glm-5.3` 生成并通过 exact AST、数值、runtime 与 16/16 性能门；组件和均
  明确不是直接 E2E，Poisson 阶段也不冒充完整求解器或公开数据负载；
- 增加标准库只读证据审计器，以及 wheel/sdist 发布物 manifest 工具；后者校验安全归档路径、
  禁带文件、完整逐文件哈希、包元数据一致性和 Git provenance，脏树只标记
  `development_only`，`--require-clean` 才能通过 RC 门；sdist 显式排除内部 todo 与历史 PDF；
- 修复 OpenAI-compatible SSE 在网关发送完整 `data: [DONE]` 后保持 HTTP/1.1
  keep-alive 时仍等待 EOF、最终误报传输超时的问题；流式读取现在逐行、有界、常量状态识别
  完整终止事件，并把 `NUMPILOT_LLM_TIMEOUT` 落实为包含连接与整个流读取的单请求 wall-clock
  deadline。非流式响应合同不变，回执继续绑定实际消费到完整终止事件的响应体 SHA-256；
- 真实 configured-model 回归已通过轻量 probe、模型身份/回执校验以及
  `elementwise_forward` 的 strict-NPU 迁移与数值对拍；linkage-v2 MatMul Adapter 长请求不再
  因 keep-alive 假超时，但上游仍可能无视关闭 thinking 的参数、耗尽输出预算并返回空
  content，因此恢复链继续停在 `CANDIDATE_READY`，没有发布未完整的新 Adapter；
- 将模型配置与回执合同泛化为精确身份绑定的 OpenAI-compatible Chat Completions 最小协议；
  provider 仅作为审计标签，迁移、缺口生成、repair 和 adapter 路径共用同一验证器；
- 增加 `NUMPILOT_HOME`、持久路径解析和无凭据 `activate-numpilot.sh`，支持临时 HOME、重启后
  复用、CANN 路径去冲突和 direct lane 不依赖 CANNBot；
- 增加 canonical capability catalog、精确 runtime identity、调用 profile、独立 `analyze`
  命令与 qualified ordinary migration，未知/漂移/未资格化合同默认失败关闭；
- 增加 installed CANN ACLNN 与固定 ops-nn v9.0.0 的 official-first 只读 resolver、write-once
  收据和 configured-model 新 kernel 前置门禁；
- 增加 `numpy.matmul` 缺口案例：复用 ops-nn `matmul/mat_mul_v3` 对应的已安装 ACLNN 实现，
  经限定范围 AsNumPy 接入、真实模型迁移和 Ascend 910B4 数值资格化后，证明原生调用一次且
  宿主 fallback 为零；该案例不发布性能结论；
- 增加 ordinary `verify-script`：重新绑定 source/migration report/migrated hash，在两个独立
  `python -I` worker 中通过无 pickle、有界 named-output 合同做数值对拍，记录解释器及
  NumPy/AsNumPy 模块身份并发布 write-once 回执；固定不宣称 OS sandbox、strict-NPU 或性能；
- 将 gap 恢复控制面升级为 schema-2 typed stage receipt：以私有 write-once SHA-256 状态链冻结
  source/target/runtime/policy，每次 phase `+1` 必须有 parent/binding/kind/coverage 精确匹配的
  新生产者收据；新增 `recover-gaps ingest-official/status`，严格消费 migration、GapSpec 和
  official-resolution，幂等推进到 `RESOLVED`；新增三种冻结授权 profile、installed
  `official_binary` fresh candidate materializer 和 reviewed MatMul adapter receipt；官方二进制
  通用链到 `CANDIDATE_READY`，仅 reviewed MatMul 切片可继续到 `ADAPTER_CANDIDATE`，
  `candidate-generation` profile 不能途中升级。仅初始冻结的 `full-local-recovery` 可授权未来本地
  integration/build/execution/activation producer，这些 producer 尚未自动接线；
- 增加 `official_aclnn_existing_module` AsNumPy planner：精确绑定 installed candidate、五类
  official refs、目标源码 context/overlay allow-list 和 route-specific Skill；schema-2
  `prompt_dependencies` 将每个目标绑定到最小 reference/signed/generated 投影，并以
  `projection_sha256` 和 remote `request_body_sha256` 绑定请求。configured model 返回
  `NUMPILOT_EXACT_EDIT_V1` 语义编辑，本地校验 base hash/唯一 OLD/非重叠区间后精确
  重建完整文件；新增 official target edit windows，`product.cpp` 只允许 Matmul 函数、
  `test_product.py` 只允许 imports/helpers 与 matmul 段，OLD 不得跨窗口。以现有
  schema-4 report/receipt 绑定；GLM exact route 使用严格 SSE 聚合、thinking disabled/
  reasoning minimal，仅聚合后空 content 才从 8192 升到 16384。真实端点已产生
  5 份成功 remote receipts，状态仅到 `ADAPTER_CANDIDATE`，未 build/execute/activate；
  普通 operator draft 与 adapter 写入白名单完全分离；
- 修复 ordinary 直接发现 missing API 时只写报告、不发布 `gap-request.json` 的断点；现在
  missing 与用户拒绝 audited fallback 两条路径都会生成 canonical GapSpec hand-off，且均不在
  缺口未解决时调用迁移模型；
- 重写可渲染 HTML 徽章首页与新机器 quickstart；退役曾使用的
  `release/competition-submission` 方案并统一由 `main` 发布；删除历史 PDF、WIP/AI 交接稿、
  旧 profiler/数组/日志和已否决展示材料；
- 公开 evidence 收敛为 Poisson 应用阶段与逐元素优势区间两案，新增发布树只读门禁，自动检查
  文件白名单、大小/后缀、版本、证据清单、PPTX 完整性和本机绝对路径。

## 0.2.0a2 - 2026-08-25

- 新增顶层 `install.sh` 源码安装入口：全新机器只需克隆 NumPilot，即可在项目虚拟环境中
  安装构建工具、从固定官方源码编译 AsNumpy，并拉取/校验固定 CANNBot Skills；联网和
  CANNBot 许可证都必须由用户显式确认，脚本不运行 sudo、不修改 shell 配置；
- 源码安装器会将项目 venv 的工具目录前置到 `PATH`，确保现场安装的 CMake 不会被
  过旧系统版本遮蔽；新增显式 `--pip-index-url`，默认镜像缺包时不会悄悄换源；
- 一键安装的默认 bootstrap receipt 改用 NumPilot XDG 状态路径，不污染 Git 工作树；
- NPU capability 层已接入 pinned-source bootstrap receipt：全部已审计 Python wrapper 继续使用
  内置 hash，现场编译且不可重现的 `_core.so` 必须同时匹配 wheel 安装记录和 README
  verification hash；source/trust/path/interpreter/package 任一漂移即回到 unknown；
- 同时保留用户手工 checkout/安装路径；bootstrap 会校验手工 AsNumpy/CANNBot 的精确
  origin、commit、tree 和工作树，完全自行安装的 AsNumpy 只能获得基础 smoke 结论；
- 将源码构建依赖同步到 AsNumpy 上游要求：`scikit-build-core>=1.0.0`、`cmake>=3.26`，
  并在一键入口前置检查 Python 开发头、系统 fmt/spdlog 和 CANN 头库；
- CANNBot 精确 pin 更新到 `1ef6372a538f39b0b7ed34666a3e1f465fb4936a`；NumPilot 使用的
  36 个 Skill/模板文件与上一 pin 内容逐文件一致；
- 全新机器 Quickstart、Notebook 和 AI Prompt 改为 clone-only 源码流，明确配置文件需在
  新机器单独创建，不再把 wheelhouse 作为默认验收输入；
- 明确本轮若接受官方 AsNumpy `matmul` host fallback，只能验证兼容性和数值，报告保持
  `strict_npu=false`、`performance_valid=false`，不得复用历史 P0 wheel 加速结论。

## 0.2.0a1 - 2026-08-25

- 将普通纯 NumPy 脚本设为默认迁移通道：先做不执行源码的静态扫描，再按能力结论选择
  直接迁移、询问是否接受 NumPy fallback，或交接到缺口补全；统一自动完成
  `integrate → qualify → activate → rescan` 的恢复编排尚未实现；
- 将 callable 但 provenance 为 `UNKNOWN` 的 API 单列为 `CAPABILITY_AUDIT_REQUIRED`，避免把
  尚未审计的已有接口误判为缺口并生成同名替代算子；
- 普通脚本迁移代码只能由配置并校验身份的 GLM 生成；报告绑定模型回执、源码与产物
  SHA-256，生成失败时不写入候选迁移文件；
- 增加普通脚本安全门：限制 NumPy-only 输入，审计 import/别名、API、签名、profile、
  fallback、输出边界和迁移 AST，并保持静态扫描、LLM 生成与代码执行相互隔离；
- 保留 `build_inputs` / `forward` contract 作为可选严格验证与 benchmark 通道，不再把它们
  作为普通脚本迁移的前置条件；
- 新增显式 `numpilot bootstrap`：默认 plan 只读且不联网；apply 必须选择可信本地
  AsNumpy wheel + SHA-256 或固定官方源码构建，CANNBot 获取另有许可证确认与原子 receipt；
  `bootstrap --verify` 可真实复验 AsNumpy smoke 与已固定的本地 CANNBot Skill 清单；
- 基础运行依赖保持 Python 3.10+、NumPy 1.26+ 和 Loguru；Demo v1 P0 wheel 的已审计 profile
  固定为 CPython 3.10/aarch64、NumPy 1.26.4、Loguru 0.7.3；源码构建依赖通过
  `numpilot[bootstrap]` 显式安装，不在迁移或 import 阶段补取；
- 记录 P1 待定的 ops-nn 官方算子复用层：缺口优先做版本/许可证/语义匹配，命中后只构建
  选中官方算子并由 GLM 生成 AsNumpy adapter，无精确匹配才生成新 kernel；默认安装不拉取；
- 新增可直接打开的全新 Notebook 验收模板，覆盖固定 tag、`numpilot[demo]`、P0 wheel、
  CANNBot、GLM-5.3 ordinary migration 与 NPU 实跑；当前验收明确不接入 ops-nn；
- 新增 Demo Case 01 精选证据：2048×2048 float32 稳态计算中 CPU 单线程中位数
  `809.784 ms`、NPU `4.499 ms`，NPU 约快 `180.004×`；冷启动成本单独报告；
- 将默认报告、benchmark、向导和缺口补全产物移到仓库外的 XDG 状态目录；
- 新增 `numpilot paths` 与 `NUMPILOT_STATE_HOME`、`NUMPILOT_RUNS_DIR`、
  `NUMPILOT_REPORTS_DIR` 覆盖；
- 新增不执行案例源码的 `numpilot check-suite` 静态准入门，区分直接迁移、单个简单缺口和
  阻断案例，并可严格核对 5+5 测试集构成；
- 静态准入门绑定真实 AsNumpy Python/C++ 签名与返回语义，检查创建器 dtype、axis、
  scalar operand、einsum 形式、只读 ndarray metadata 和 `build_inputs` 参数类型；
- 新增模型生成/修复 AsNumpy adapter 的独立命令及内置 custom-op integration Skill；
  adapter、operator repair 与 seal/reload 均绑定完整 Skill、诊断、父链和文件身份；
- 未实现的 `--operator-goal performance` 现在会在联网、读取 Skill 或写输出前明确拒绝；
- 整理可公开的 heat-diffusion 与 Poisson strict-NPU 精选证据；
- 增加 CANN memcpy 方向探针、Poisson profiler 复制审计和哈希绑定的 AsNumpy move wheel
  capability；同一 10 轮 workload 的 `aclrtMemcpy` 从 1,235 降至 35；
- 增加 Poisson 原始 48×48/49 RBF/3000 轮独立 CPU/NPU 性能 runner、OpenBLAS 线程实测、
  同规模 profiler 发布门与精选证据；合格结果为 CPU 约快 `2.67×`，不宣称 NPU 加速；
- 增加 Poisson 广播/CSE 优化变体和哈希独立的性能 runner；每轮调用从 117 降到 101，
  原始规模 NPU 中位时延下降 `19.71%`，3000 轮最终输出与基线字节级相同；
- 增加 `run_poisson_demo.sh` 客户体验入口：先校验 135 份公开证据，再展示固定
  CANNBot Skill、算子 qualification、实时 strict-NPU 迁移与驻留执行；
- 修复新版 `npu-smi` 仅显示 `910B4` 而无 `Ascend` 字样时的 doctor 设备误报；
- 增加 Poisson 单核 CPU 亲和性实验 runner：CPU 子进程在 Python 启动前固定到一个逻辑核，
  收据同时验证 affinity 与 BLAS 单线程；在缺少同身份 passing profiler audit 时明确保持
  `performance.published=false`，不发布或引用实验性能比值；
- 增加开源许可证、贡献、安全、第三方依赖和证据存储说明。

## 0.1.0

- NumPy → AsNumpy AST/LLM 迁移原型；
- shim 与真实 NPU 后端、capability provenance、严格数值比较和 benchmark 门禁；
- 缺口算子 GapSpec、LLM 草稿、repair、封存、qualification overlay 与 Poisson P0 纵切。
