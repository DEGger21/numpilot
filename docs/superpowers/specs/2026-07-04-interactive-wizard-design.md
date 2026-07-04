# NumPilot 交互式迁移向导 — 设计

- 日期:2026-07-04
- 分支:feat/numpilot-demo
- 目标:把迁移流程"程序化 / 交互化",做一个终端向导:启动界面 → 让用户输入待迁移程序
  绝对路径 → 选引擎 → 带进度条+实时文字解释地跑迁移 → 最后给迁移反馈(性能对比、
  接入多少自定义算子等)。

## 决策(已与用户确认)
- UI:引入 **rich** 库(进度条/彩框/表格),写进 pyproject 依赖。
- "自定义算子接入"口径:**两者都显示** —— 总替换数(所有 np.*→ap.*)+ 其中 asnumpy
  自定义 NPU 算子(relu/softmax/gelu/angle)数量。
- 引擎:**向导内让用户选** ast / llm(glm-5.1);选 llm 失败按 strict 终止。
- 性能:**完整对比** —— 子进程 bench-cpu(numpy 钉 1 核)+ 向导内 NPU 测速,算加速比。

## 入口
- 新子命令 `python -m numpilot wizard`。
- 也支持带参非交互:`wizard <file> --engine ast --backend npu`(跳过对应提问;供脚本/测试)。
- 新模块 `numpilot/wizard.py`。

## 流程
1. **启动界面**:rich Panel,标题 + "系统启动·昇腾 NPU 代码迁移智能体"。
2. **选代码**:Prompt 输入绝对路径 → 校验(存在 / `.py` / 含 `build_inputs`+`forward`),非法红字重问。
3. **选引擎**:Prompt choices=[ast, llm];llm 时从 `.env.llm` 载入凭据到 os.environ,strict。
4. **实时迁移**(rich Progress:spinner+进度条+步号,每步一句解释):
   - [1/5] 分析算子(AnalysisAgent)
   - [2/5] 决策卸载(DecisionAgent)
   - [3/5] 迁移改写(MigrationAgent;llm 时显示"请求 glm-5.1…")
   - [4/5] 真机校验+NPU测速(VerificationAgent, backend=npu, measure=True → 正确性 + asnumpy 证据 + NPU 时延)
   - [5/5] numpy 单核基线(子进程 `python -m numpilot bench-cpu <file> --out <tmp>`,读 cpu_baseline.json 取 t_cpu)
5. **迁移反馈**(rich Table/Panel):网络名/源路径/引擎;分析(调用数/算子数/覆盖);
   迁移(总替换 K + 自定义 NPU 算子 M 个逐列);卸载(NPU/CPU);执行证据(asnumpy 版本+路径);
   校验(数值一致 + max_rel_diff);性能(numpy@1核 vs asnumpy@NPU + 加速比);产物 migrated.py。

## 关键复用
- 公平性能 = 子进程 bench-cpu(1 核,进程隔离不污染 CANN)+ 向导进程内 verify(measure=True)
  拿 NPU 时延(向导进程满线程),speedup = t_cpu / t_npu。
- 自定义算子:coverage.py 加 `CUSTOM_OPS = {relu, softmax, gelu, angle}` + `is_custom(name)`;
  向导对迁移后 ap.* 命中 CUSTOM_OPS 的算子计数(复用 engines.ast_rewrite.count_ap_calls)。

## 错误处理
- 路径非法 → 重问(最多不限,Ctrl-C 退出)。
- npu 环境不满足 → 复用 env_check,拦截并提示;询问是否降级 shim。
- LLM 失败 → 捕获 LlmError,红字终止(exit 4 语义)。

## 测试
- 单测 helper:路径校验、custom-op 计数、载入 .env.llm(用临时文件)。
- 非交互端到端:`wizard <file> --engine ast --backend shim` 跑通不抛异常(CI 安全,不需 NPU/网络)。
- 交互输入层不强测。

## 不做
- 不改已有 migrate/bench/doctor;不做多文件批量;不做 web UI。
