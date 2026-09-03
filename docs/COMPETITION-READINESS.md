# 比赛就绪度检查

`competition-readiness` 是一个本地、只读的提交前审计器。它把
`XH-202618` 的三个评分维度和硬性材料要求转换成可核查的检查项：

```bash
python -m numpilot competition-readiness
python -m numpilot competition-readiness --json
# 回执目录必须是当前用户私有目录；不要直接把 --out 放在共享的 /tmp 根目录。
RUN=$(mktemp -d /tmp/numpilot-readiness.XXXXXX)
chmod 700 "$RUN"
python -m numpilot competition-readiness --strict --out "$RUN/readiness.json"
```

命令只读取仓库文件、脱敏的本地配置元数据和本地 Git 状态；不会导入或执行案例源码、调用
模型、联网、访问 NPU，也不会把 `.codeartsdoer/` 配置误认为 CodeArts 平台执行记录。`--out`
只会创建一个新的（0600）JSON 回执，若目标已存在则拒绝覆盖。

报告中的 `official_score` 始终为 `null`：检查器不能代替评委打分。`coverage_ratio` 只是本地
证据覆盖率，不是比赛得分。`--strict` 在任一关键检查不是 `pass`（包括 `partial`）时返回 1，
默认返回 0 以便先查看缺口。

`C03` 按 PDF 允许“视频或在线环境”二选一；具备完整的华为云环境回执时，即使没有本地视频文件也
可通过本地材料检查，但仍必须人工确认外部访问和录制内容。

注意：`pass` 表示“本地文件/字段存在且通过格式启发式”，不是平台、网络或业务事实的独立
证明。视频文件/托管链接、报名表和压缩包目前只做非空文件、视频语境链接或扩展名观察（报名表支持
常见 PDF/Office/表格/图片格式）；ZIP
内部可读性、比赛要求的“院校名称+队长姓名+手机号+队伍名称”命名、报名系统信息一致性，以及第三方 AI/组件来源和
许可证，仍须由提交人和评委人工复核。建议把最终 ZIP、报名表、SBOM、第三方清单和每项
CodeArts 导出记录放进一个冻结 tag 后再运行 `--strict`。
同样，PPT 文件存在只证明附件入口，不证明场景、需求、过程记录、测试效果和扩展性内容齐全；
仓库中的 `numpilot-zh.pptx`/`numpilot-en.pptx` 是单页架构源文件；7 分钟讲解初版及其截图
占位稿已移至仓库外的本地答辩归档。投影可读性、真实 CodeArts/云端画面和赛题材料覆盖仍需
人工补齐，不能把占位框当作真实性证据。

现场没有 Ascend 时可以用 `numpilot run ... --backend shim --fallback-policy accept` 做离线数值
诊断；它不会通过 `--require-direct`，也不会生成 strict-NPU/性能声明。比赛演示仍应使用 activated
NPU runtime 和 `run_demo.sh` 的 strict 主链。

## `run` 流水线回执定位

现场展示一键主链时，`numpilot run ... --out RUN_DIR` 会在全新目录写入
`workflow-report.json`。该文件只是编排索引，阶段回执仍以各自的 JSON 为准；不要把
`workflow-report` 单独当作 NPU 或业务真实性证明。

- `status` 是整条流水线结论：`passed` 表示所有请求阶段通过；`blocked` 表示模型/资格、
  源码绑定或可恢复前置条件未满足；`failed` 表示阶段执行失败。具体超时请查看
  `stages[*].status=timeout` 或 `timed_out=true`。
- `stages` 按顺序记录 `01-analyze`、`02-migrate`、`03-verify`（以及可选的
  `04-benchmark`），包括退出码、耗时、`stdout_path`/`stderr_path` 和日志 SHA-256。日志是
  有界的私有文件；失败时先查看对应阶段的 stderr，再打开其 `receipts` 中的
  `qualified-analysis.json`、`script-migration-report.json`、`script-verification-report.json`
  或 `script-benchmark-report.json`。
- `claims` 只汇总已通过阶段回执中的事实：`model_generated`、`numerically_verified` 和
  `strict_npu_performance_verified`；它们为 `false` 时不得在 PPT/视频中宣称相应能力。
- `source_guard` 在 admission、每个阶段边界和 final 重新计算源码摘要，并核对阶段回执的
  `source_sha256`。其状态为 `blocked` 或 `receipt_hashes_match=false` 时，源码可能在运行中漂移，
  应保留该目录用于审计并换一个全新 `--out` 重跑。

命令行非 JSON 模式会显示失败阶段的安全化 stderr 尾部；需要机器处理时使用
`--json`，再按 `stages[*].stderr_path` 和 `receipts` 精确定位。流水线不覆盖旧目录，也不
自动重试模型请求；修复配置、环境或源码后必须指定新的运行目录。

`run_demo.sh` 末尾的 `tools/render_demo_summary.py` 只读四份回执并打印一屏摘要。其中
`[跨回执绑定]` 会比较分析（A）、迁移（M）、数值对拍（V）和性能（B）回执携带的
`source`、`migrated`、`report` SHA-256：字段齐全且一致才是 `PASS`；字段缺失显示
`UNKNOWN（不可发布）`，不一致或格式非法显示 `FAIL（不可发布）`。`[设备驻留 headline]`
仍只表示性能回执的门禁，不能替代这条跨回执绑定结论；发布前应先处理 UNKNOWN/FAIL，
并保留原始回执供定位。

在真实交互终端中，摘要会自动显示为响应式 Rich 证据卡：宽终端使用阶段表格，窄终端改为
逐阶段门禁并折叠长路径；卡片明确标注“只读展示”，不会重新执行源码或联网。录屏、CI 或
重定向场景保持稳定纯文本；调用 `tools/render_demo_summary.py --plain` 可取得可复制的完整
绝对路径。`--quick`（2 次
预热、7 个样本）适合现场演示，`--full`（3 次预热、30 个样本）才适合冻结性能证据；两者的
数字应在答辩材料中分别标注协议，不要把现场 quick 数字与历史 full 回执混写。

## 视频演示文件：录制、复核与提交

视频应证明一次真实、可回放的工程过程，而不是把若干静态截图剪成“已完成”。建议在同一个
CodeArts 会话和同一个冻结 commit 上连续录制一条 5–8 分钟的 MP4（H.264、1080p、清晰收音）；
保留原始录屏在仓库外，公开提交包只放脱敏后的成片、封面和校验清单。

录制前先完成以下准备：

1. 从待交付的 `main` commit 打开 CodeArts IDE，确认项目、分支、Codebase 索引和侧边栏 AI
   对话可见；实际账号、套餐和区域若不可用，先记录 `BLOCKED`，不要用本地终端冒充平台画面。
2. 在目标 NPU 机 source CANN 环境，并用 `components/asnumpy/` 的本地组件通过
   `python tools/check_release_tree.py --require-asnumpy-component --json`；模型配置、Token、
   代理和运行回执放在仓库外的私有目录。
3. 预先测试麦克风、窗口缩放和字体；终端至少 120 列，录屏中能读清命令、阶段状态、
   `warmup/samples` 和回执路径，但不要让私有 URL、Token、手机号或完整模型响应出现在画面。
4. 为本次录制新建空的私有 `RUN_DIR`；不要复用旧回执，也不要在录制中修改历史 evidence。

建议按下面顺序录制并在视频字幕/章节中标出时间：

| 时间 | 画面与旁白 | 必须说明的边界 |
|---|---|---|
| 0:00–0:40 | 问题、普通 NumPy 输入和目标 Ascend NPU | 当前只承诺可静态分析的支持子集，不是任意 NumPy 项目 |
| 0:40–1:40 | CodeArts 侧边栏：Codebase 索引、知识问答、规范/任务拆解 | 五类平台记录必须有真实任务 ID；仓库里的 Agent/Skill 文件不等于平台执行 |
| 1:40–2:40 | CodeArts 代码生成与单元测试生成，展示 diff 和人工审批 | 模型只提出候选，不能把未审查草稿称为已接入算子 |
| 2:40–4:20 | Remote SSH/终端运行 `run_demo.sh --quick`，展示 analyze → migrate → verify | `verify` 数值通过不等于 strict-NPU 或性能通过；组件安装耗时也不是冷启动 E2E |
| 4:20–5:10 | 打开一个缺口/动态 dispatch 负例 | 明确展示 `GAP_REQUIRED`、`CAPABILITY_AUDIT_REQUIRED` 或 `BLOCKED`，并保留非零退出码 |
| 5:10–6:00 | `render_demo_summary.py --plain`、A/M/V/B 哈希绑定和运行时画像 | 只播放已有回执，不在摘要工具中重新执行源码或联网 |
| 6:00–结束 | 华为云外部访问、视频入口和提交材料清单（若已具备） | 本机 localhost、静态截图和未验证 URL 不能算华为云部署证据 |

录制结束后，从头到尾回放成片，核对画面中的项目/分支/commit、CodeArts 任务 ID、命令退出码、
运行时版本和失败边界与回执一致；同时用另一台设备或无登录状态检查视频权限和播放。为成片和
封面生成 SHA-256，填写脱敏的 `evidence/video.json`（`url`、`accessible`、时长、commit 和
`sha256`），再将它与 CodeArts 导出记录一起放入受限比赛提交包。无法外部回放、缺少真实平台记录
或只剩截图时，`accessible`/`actual_execution` 必须保持 `false`，并在报告中写明待补项。

视频推荐的最小验收清单：真实 CodeArts 侧边栏、一次成功主链、一次失败关闭、固定 AsNumpy
组件校验、证据摘要和明确非目标均出现；音频可听、文字可读、无凭据泄露、成片可下载且哈希与
提交清单一致。录屏本身不能替代真实数据来源、受益方确认、许可证/SBOM 或第二环境复现。

## 可选的脱敏证据合同

把下列文件放入最终交付包后，检查器才能把对应要求提升为“有本地证据”。字段值应脱敏，不要
放 Token、私钥或内部密码。

### CodeArts 实际使用

`evidence/codearts-usage.json`：

```json
{
  "platform": "Huawei CodeArts",
  "actual_execution": true,
  "features": [
    "code_generation",
    "knowledge_qa",
    "unit_test_generation",
    "codebase_index",
    "spec_driven_development"
  ]
}
```

上面的最小格式继续兼容历史材料。冻结提交建议增加 `records`，把五类功能逐项绑定到平台导出
文件；只有字段齐全、记录文件存在且 SHA-256 匹配时，检查器才把它标记为“可追溯”：

```json
{
  "records": {
    "code_generation": {
      "record_id": "脱敏后的 CodeArts 任务 ID",
      "record_path": "evidence/codearts-records/code-generation.json",
      "timestamp_utc": "2026-09-03T12:00:00Z",
      "sha256": "记录文件的 64 位十六进制摘要",
      "status": "passed"
    }
  }
}
```

`records` 也可以是带 `feature` 字段的对象数组。五个 feature 都应各有一条记录；`record_path`
必须是仓库相对路径（不允许绝对路径、`..`、反斜杠或符号链接），`timestamp_utc` 必须显式带
`Z`/`+00:00`。报告会给出 `records_status=verified|partial|blocked|legacy`、缺失 feature、
哈希漂移和不可读文件的原因；`records` 缺失时保留旧的最小 JSON 判定，并显示
`records_status=legacy`。因此“平台声明存在”与“每项记录可回放”不会被混为一谈。

截图、任务记录和导出文件应与该声明放在同一证据目录，并在仓库级清单中绑定 SHA-256。仓库
中的 Agent/Skill Markdown 只能证明“可加载配置”，不能证明目标账号真的执行过。

### 华为云部署

`evidence/cloud-deployment.json`：

```json
{
  "provider": "Huawei Cloud",
  "status": "verified",
  "verification": true,
  "endpoint": "https://脱敏域名或公开演示地址"
}
```

报告只显示 endpoint 的主机名，不读取凭据。在线演示必须在华为云完成外部访问验收。

也可以用 `evidence/video.json` 或 `evidence/demo-environment.json` 记录脱敏演示地址：

```json
{
  "url": "https://公开演示地址",
  "accessible": true
}
```

检查器只在 `accessible` 明确为 `true` 时计入材料，并且报告只显示主机名；它不会主动请求
该地址，因此仍需在视频和答辩材料中提供实际访问验证。

### 真实工程负载

`evidence/workload-manifest.json`：

```json
{
  "data_source": "可追溯来源（URL、内部授权编号或公开数据集）",
  "license": "数据/代码许可证",
  "beneficiary": "实际使用方或受益方（可脱敏）",
  "source_sha256": "64位十六进制摘要",
  "data_sha256": "64位十六进制摘要"
}
```

该文件证明提交材料具备来源和哈希字段，不替团队向评委证明业务关系；PPT 和答辩仍应解释
场景边界、受益方、验证环境和可扩展性。

## 提交前人工核对（PDF 硬要求）

本地审计器不能读取报名系统或 CodeArts 云端，因此以下项目必须在交付负责人清单中逐项
签字确认：

1. 最终源码 ZIP 名称严格为 `院校名称+队长姓名+队长手机号+队伍名称`，且与报名表/系统填报
   信息逐字符一致（手机号等敏感字段只放在受限交付包，不放公开仓库）。
2. 代码生成、知识问答、单元测试生成、Codebase 索引、规范驱动开发五类 CodeArts 记录均可
   回放，并能关联到同一个项目/时间线。
3. 在线演示地址确实位于华为云，评委从外部网络可访问；视频包含环境、输入、输出、失败
   关闭和回执核验，而非只展示静态截图。
4. 真实数据来源、授权/许可证、脱敏规则、受益方确认和独立验收记录齐全；不要把示例或
   合成数据写成真实业务证明。

## 状态含义

- `pass`：本地观察到符合合同的证据；
- `partial` / `warn`：有部分材料或只有声明，不能满足最高档；
- `blocked`：关键材料缺失，提交前必须补齐。

当 `codearts-usage.json` 含有 `records` 时，C05/C06 会把记录文件缺失、路径不安全、时间戳
不明确、状态非 `passed` 或哈希漂移降为 `partial/blocked`；检查器不会读取记录内容，因此不会
把记录中的账号、回答或凭据写进报告。

当前开发态证据若显示 `git_dirty=true`、没有云部署/CodeArts 实际记录或仅使用合成数据，报告
会保守地保留阻断项。这是防止把设计目标或历史开发态回执误写成已完成比赛要求。
