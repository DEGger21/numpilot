# 变更记录

## Unreleased

- 暂无。

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
