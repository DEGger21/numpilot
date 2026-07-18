"""交互式迁移向导:启动界面 → 选代码 → 选引擎 → 进度迁移 → 迁移反馈。

依赖 rich 做终端交互(进度条/彩框/表格)。向导只负责迁移、产物级正确性和
capability provenance；性能必须通过独立的 ``bench-cpu`` / ``bench-npu``
严格门禁发布，避免在交互流程中形成不可审计的加速比。
"""
import ast
import os
import re

from rich.console import Console
from rich.panel import Panel
from rich.progress import (BarColumn, Progress, SpinnerColumn, TextColumn,
                           TimeElapsedColumn)
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich import box

from numpilot.agents.analysis import AnalysisAgent
from numpilot.agents.decision import DecisionAgent
from numpilot.agents.migration import MigrationAgent
from numpilot.agents.verification import VerificationAgent
from numpilot.backends.base import select_backend
from numpilot.coverage import is_custom
from numpilot.engines.llm import LlmError
from numpilot.env_check import Env
from numpilot.orchestrator import _make_engine
from numpilot.report import save_migration_report

console = Console()
_OUT_DIR = "out_wizard"


# ───────────────────────── helper ─────────────────────────
def validate_task_file(path):
    """返回 (ok, 规范化路径, 错误信息)。要求:存在、.py、含 build_inputs+forward。"""
    if not path:
        return False, path, "路径为空"
    p = os.path.expanduser(path.strip())
    if not os.path.isfile(p):
        return False, p, f"文件不存在:{p}"
    if not p.endswith(".py"):
        return False, p, "不是 .py 文件"
    try:
        tree = ast.parse(open(p, encoding="utf-8").read())
    except SyntaxError as e:
        return False, p, f"Python 语法错误:{e}"
    funcs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    missing = {"build_inputs", "forward"} - funcs
    if missing:
        return False, p, f"缺少必需函数:{', '.join(sorted(missing))}(任务程序须定义 build_inputs() 与 forward())"
    return True, p, ""


def load_env_llm(path=".env.llm"):
    """把 .env.llm 里的 `export KEY="VAL"` 载入 os.environ。返回载入的 key 列表。"""
    if not os.path.isfile(path):
        return []
    loaded = []
    pat = re.compile(r'^\s*export\s+(\w+)\s*=\s*"?([^"\n]*)"?\s*$')
    for line in open(path, encoding="utf-8"):
        m = pat.match(line)
        if m:
            os.environ[m.group(1)] = m.group(2)
            loaded.append(m.group(1))
    return loaded


def custom_op_breakdown(rewrites):
    """挑出 AsNumpy 的扩展 API；这不代表本次新生成了自定义算子。"""
    return {op: n for op, n in rewrites.items() if is_custom(op)}


def agent_roster(engine_name):
    """返回 5 智能体阵容 [(名称, 职责, 模型/引擎)]。只有迁移智能体(可选)用大模型。"""
    mig_model = ("glm-5.1 大模型" if engine_name == "llm"
                 else "AST 确定性改写(规则,无模型)")
    return [
        ("编排智能体 Orchestrator", "统筹流水线 · 传递上下文 · 汇总报告", "规则编排(无模型)"),
        ("分析智能体 Analysis", "AST 静态解析算子 · 比对覆盖集", "Python AST(规则,无模型)"),
        ("决策智能体 Decision", "把整个 forward 划为单一 NPU 岛", "确定性边界规则(无模型)"),
        ("迁移智能体 Migration", "np.* → ap.* 代码改写", mig_model),
        ("验证智能体 Verification", "双端执行 · 数值对比 · 测速", "数值对比 + 真机执行(无模型)"),
    ]


def _print_roster(engine_name):
    tb = Table(title="多智能体阵容(本次任务)", box=box.SIMPLE_HEAVY, title_style="bold cyan")
    tb.add_column("智能体", style="bold")
    tb.add_column("职责")
    tb.add_column("模型 / 引擎", style="magenta")
    for name, role, model in agent_roster(engine_name):
        hi = "[bold magenta]" if "glm" in model else ""
        end = "[/]" if hi else ""
        tb.add_row(name, role, f"{hi}{model}{end}")
    console.print(tb)


# ───────────────────────── 各步骤 ─────────────────────────
def _banner():
    console.print(Panel.fit(
        "[bold cyan]NumPilot[/]  ·  昇腾 NPU 代码迁移智能体\n"
        "[dim]系统启动 · 把 NumPy 程序自动迁移到 Ascend NPU(np.* → ap.*)[/]",
        border_style="cyan", box=box.DOUBLE))


def _ask_file(file):
    if file is not None:
        ok, p, msg = validate_task_file(file)
        if not ok:
            console.print(f"[red]✗ {msg}[/]")
            return None
        console.print(f"[green]✓[/] 待迁移程序:[bold]{p}[/]")
        return p
    while True:
        raw = Prompt.ask("[bold]请输入待迁移程序的绝对路径[/]")
        ok, p, msg = validate_task_file(raw)
        if ok:
            console.print(f"[green]✓[/] 已选定:[bold]{p}[/]")
            return p
        console.print(f"[red]✗ {msg}[/],请重试。")


def _ask_engine(engine):
    if engine is None:
        engine = Prompt.ask("[bold]选择迁移引擎[/]  [dim](ast=确定性改写/离线, llm=glm-5.1 大模型智能体)[/]",
                            choices=["ast", "llm"], default="ast")
    if engine == "llm":
        loaded = load_env_llm()
        if not loaded:
            console.print("[red]✗ 未找到 .env.llm,无法启用大模型智能体。[/]")
            return None
        console.print(f"[green]✓[/] 已载入大模型凭据(model=[bold]{os.getenv('NUMPILOT_LLM_MODEL','?')}[/],"
                      "失败将按 strict 直接终止)")
    return engine


def _startup_check(backend, interactive):
    """进入系统即自检本机环境(asnumpy/CANN/NPU 是否就绪),快速失败。

    npu 后端:缺硬依赖(尤其 asnumpy)→ 交互询问降级 shim,非交互直接失败(返回 None)。
    shim 后端:缺失只是 warn,照常返回 shim。返回生效后端名或 None(终止)。
    """
    console.print("[bold]系统自检[/] · 检查本机 asnumpy / CANN / NPU 环境…")
    rep = Env.check(backend=backend)
    _COLOR = {"ok": "green", "warn": "yellow", "error": "red"}
    _MARK = {"ok": "✓", "warn": "⚠", "error": "✗"}
    tb = Table(box=box.SIMPLE, show_header=False)
    tb.add_column(no_wrap=True)
    tb.add_column()
    for it in rep.items:
        c = _COLOR.get(it.status, "white")
        tb.add_row(f"[{c}]{_MARK.get(it.status, '?')} {it.name}[/]", f"[dim]{it.detail}[/]")
    border = "red" if rep.errors else ("yellow" if rep.warnings else "green")
    console.print(Panel(tb, title="环境自检", border_style=border))

    if rep.errors:
        names = ", ".join(i.name for i in rep.errors)
        console.print(f"[red]✗ 环境不满足 {backend} 硬依赖:{names}"
                      + ("(asnumpy 未就绪则无法在真机 NPU 上执行)" if any(
                          i.name == "asnumpy" for i in rep.errors) else "") + "[/]")
        if backend == "npu" and interactive and Confirm.ask(
                "是否降级到 shim 仿真后端继续?[dim](无真机性能,仅演示流程)[/]", default=False):
            return "shim"
        return None
    console.print(f"[green]✓ 环境就绪[/](后端={backend})")
    return backend


def _run_pipeline(source, path, engine_name, backend_name):
    """跑分析→整段决策→迁移→产物校验→证据门禁。"""
    data = {"path": path, "engine": engine_name, "backend": backend_name}
    mig_tag = "glm-5.1" if engine_name == "llm" else "AST"
    # 每步归属的智能体(名称, 该步用的模型/引擎), 让用户看清是谁在动作
    steps = [
        ("分析智能体", "分析算子", "规则AST"),
        ("决策智能体", "划分 NPU 岛", "整段 forward"),
        ("迁移智能体", "迁移改写", mig_tag),
        ("验证智能体", "产物级语义校验", "严格比较"),
        ("验证智能体", "capability 证据门禁", "provenance"),
    ]
    def sdesc(i, extra=""):
        a, act, m = steps[i]
        return f"[{i+1}/5] {a} · {act}({m}){extra}"

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                  BarColumn(bar_width=24), TextColumn("{task.completed}/{task.total}"),
                  TimeElapsedColumn(), console=console) as prog:
        t = prog.add_task("准备中", total=len(steps))

        # [1] 分析智能体
        prog.update(t, description=sdesc(0))
        analysis = AnalysisAgent().analyze(source, path)
        data["analysis"] = analysis
        distinct = sorted({o.name for o in analysis.op_uses})
        if not analysis.migratable:
            prog.console.print(f"   ↳ [red]发现 {len(analysis.unsupported)} 个覆盖集缺口:"
                               f"{', '.join(analysis.unsupported)} —— 无法迁移[/]")
            prog.advance(t)
            data["stopped"] = "unsupported"
            return data
        prog.console.print(f"   ↳ [cyan]分析智能体[/] {len(analysis.op_uses)} 处 numpy 调用 · {len(distinct)} 个算子 · 全部在覆盖集 ✓")
        prog.advance(t)

        # [2] 决策智能体
        prog.update(t, description=sdesc(1))
        plan = DecisionAgent().plan(analysis)
        data["plan"] = plan
        npu_ops = sorted(o for o, p in plan.placement.items() if p == "NPU")
        prog.console.print(
            f"   ↳ [cyan]决策智能体[/] 策略={plan.strategy}:"
            f"forward 内 {len(npu_ops)} 个算子统一进入 NPU 岛"
        )
        prog.advance(t)

        # [3] 迁移智能体(LLM 时较慢)
        prog.update(t, description=sdesc(2, " · 请求中…" if engine_name == "llm" else ""))
        migration = MigrationAgent(_make_engine(engine_name)).migrate(source, plan)
        data["migration"] = migration
        os.makedirs(_OUT_DIR, exist_ok=True)
        mig_path = os.path.join(_OUT_DIR, "migrated.py")
        with open(mig_path, "w", encoding="utf-8") as f:
            f.write(migration.migrated_source)
        data["migrated_path"] = mig_path
        model_tag = "glm-5.1 大模型" if migration.engine == "llm" else "AST 规则改写"
        prog.console.print(f"   ↳ [magenta]迁移智能体·{model_tag}[/] 替换 {migration.total_rewrites} 处 np.* → ap.*")
        prog.advance(t)

        # [4] 验证智能体：运行迁移产物自己的 build_inputs + forward。
        prog.update(t, description=sdesc(3))
        report = VerificationAgent(select_backend(backend_name)).verify(
            source,
            migration.migrated_source,
            measure=False,
            required_ops=migration.expected_rewrites,
        )
        data["report"] = report
        mark = "✓" if report.numerically_consistent else "✗"
        detail = (
            f"异常 {report.exception_phase}/{report.exception_type}: {report.exception_message}"
            if report.exception_type
            else f"max_abs={report.max_abs_diff:.2e}, max_rel={report.max_rel_diff:.2e}"
        )
        prog.console.print(
            f"   ↳ [cyan]验证智能体[/] 产物级数值一致 {mark} · {detail} · "
            f"自身 build_inputs={'是' if report.artifact_inputs_executed else '否'}"
        )
        prog.advance(t)

        # [5] capability/provenance 只判资格，不在向导中顺手发布性能。
        prog.update(t, description=sdesc(4))
        provenance = report.backend_provenance or {}
        data["performance_valid"] = False
        if provenance:
            ok = provenance.get("ok")
            audit_id = provenance.get("audit_id", "?")
            gate = (
                ("通过" if ok else "未通过")
                if provenance.get("strict_npu")
                else "未执行(shim/非 strict)"
            )
            prog.console.print(
                f"   ↳ [cyan]验证智能体[/] capability audit={audit_id} · "
                f"strict-NPU 门禁={gate}"
            )
        else:
            prog.console.print("   ↳ [yellow]无 capability provenance，不具备性能发布资格[/]")
        prog.advance(t)
    return data


def _feedback(data):
    """渲染迁移反馈:总表 + 性能高亮面板。"""
    if data.get("stopped") == "unsupported":
        console.print(Panel(f"缺口算子:[red]{', '.join(data['analysis'].unsupported)}[/]\n"
                            "这些算子不在 AsNumpy 覆盖集,本 demo 不做缺口算子生成,迁移终止。",
                            title="[red]迁移未完成[/]", border_style="red"))
        return

    a, plan, mig, rep = data["analysis"], data["plan"], data["migration"], data["report"]
    custom = custom_op_breakdown(mig.rewrites)
    npu_ops = sorted(o for o, p in plan.placement.items() if p == "NPU")
    cpu_ops = sorted(o for o, p in plan.placement.items() if p == "CPU")

    tb = Table(title="迁移反馈", box=box.ROUNDED, show_header=False, title_style="bold cyan")
    tb.add_column(style="bold", no_wrap=True)
    tb.add_column()
    net = os.path.splitext(os.path.basename(data["path"]))[0]
    tb.add_row("网络", net)
    tb.add_row("源程序", data["path"])
    tb.add_row("迁移引擎", data["engine"] + (" (glm-5.1 智能体)" if data["engine"] == "llm" else " (确定性改写)"))
    tb.add_row(
        "算子分析",
        f"{len(a.op_uses)} 处 forward 调用 · {len({o.name for o in a.op_uses})} 个算子 · "
        "静态名称/签名门禁通过",
    )
    replaced = "  ".join(f"{op}×{n}" for op, n in
                         sorted(mig.rewrites.items(), key=lambda kv: (-kv[1], kv[0])))
    tb.add_row("算子替换", f"共 [bold]{mig.total_rewrites}[/] 处 np.* → ap.*\n{replaced}")
    if custom:
        clist = "  ".join(f"{op}×{n}" for op, n in sorted(custom.items(), key=lambda kv: (-kv[1], kv[0])))
        tb.add_row("AsNumpy 扩展 API", f"命中 [bold green]{len(custom)}[/] 种 / {sum(custom.values())} 处:{clist}\n"
                                  "[dim](表示复用已有扩展 API，不等于本次新生成自定义算子)[/]")
    else:
        tb.add_row("AsNumpy 扩展 API", "无(本网络未命中 relu/softmax/gelu/angle)")
    tb.add_row(
        "卸载方案",
        f"策略: {plan.strategy}\nNPU: {', '.join(npu_ops) or '—'}\nCPU: {', '.join(cpu_ops) or '—'}",
    )
    if rep.backend_version:
        tb.add_row("执行证据", f"[green]真机 asnumpy {rep.backend_version}[/]\n{rep.backend_module}")
    else:
        tb.add_row("执行证据", "shim 仿真(numpy 承载,非真机)")
    cmark = "[green]一致 ✓[/]" if rep.numerically_consistent else "[red]不一致 ✗[/]"
    if rep.exception_type:
        tb.add_row("数值校验", f"{cmark}  {rep.exception_phase}: {rep.exception_type}: {rep.exception_message}")
    else:
        tb.add_row(
            "数值校验",
            f"{cmark}  max_abs={rep.max_abs_diff:.2e}, max_rel={rep.max_rel_diff:.2e}  "
            f"(atol={rep.atol:.0e}, rtol={rep.rtol:.0e})",
        )
    console.print(tb)

    console.print(Panel(
        "向导不发布加速比。请在独立进程依次运行:\n"
        f"python -m numpilot bench-cpu {data['path']} --out out_demo\n"
        f"python -m numpilot bench-npu {data['path']} --out out_demo\n"
        "[dim]只有 strict-NPU capability、数值和基线一致性门禁全部通过，"
        "benchmark_report.md 才会给出合格加速比。[/]",
        title="性能证据", border_style="cyan"))
    console.print(f"[dim]迁移代码已写入:{data['migrated_path']}[/]")


def _report_info(data):
    """把向导 data 规范化为 report.build_report_md 需要的 info dict。"""
    net = os.path.splitext(os.path.basename(data["path"]))[0]
    info = {"net": net, "source_path": data["path"], "engine": data["engine"],
            "backend": data["backend"], "status": "unsupported" if data.get("stopped") else None,
            "agents": agent_roster(data["engine"])}
    a = data.get("analysis")
    if data.get("stopped") == "unsupported":
        info["unsupported"] = a.unsupported
        return info
    plan, mig, rep = data["plan"], data["migration"], data["report"]
    info["status"] = "pass" if rep.numerically_consistent else "fail"
    info["n_calls"] = len(a.op_uses)
    info["n_ops"] = len({o.name for o in a.op_uses})
    info["rewrites"] = mig.rewrites
    info["total_rewrites"] = mig.total_rewrites
    info["extension_apis"] = custom_op_breakdown(mig.rewrites)
    info["strategy"] = plan.strategy
    info["npu_ops"] = sorted(o for o, p in plan.placement.items() if p == "NPU")
    info["cpu_ops"] = sorted(o for o, p in plan.placement.items() if p == "CPU")
    info["backend_version"] = rep.backend_version
    info["backend_module"] = rep.backend_module
    info["numerically_consistent"] = rep.numerically_consistent
    info["max_rel_diff"] = rep.max_rel_diff
    info["max_abs_diff"] = rep.max_abs_diff
    info["tol"] = rep.tol
    info["atol"] = rep.atol
    info["rtol"] = rep.rtol
    info["failure_path"] = rep.failure_path
    info["failure_index"] = rep.failure_index
    info["failure_reason"] = rep.failure_reason
    info["exception_phase"] = rep.exception_phase
    info["exception_type"] = rep.exception_type
    info["exception_message"] = rep.exception_message
    info["artifact_inputs_executed"] = rep.artifact_inputs_executed
    info["input_consistent"] = rep.input_consistent
    info["kernel_consistent"] = rep.kernel_consistent
    info["artifact_consistent"] = rep.artifact_consistent
    info["comparisons"] = rep.comparisons
    info["backend_provenance"] = rep.backend_provenance
    info["performance_valid"] = False
    info["migrated_path"] = data.get("migrated_path")
    return info


# ───────────────────────── 入口 ─────────────────────────
def run_wizard(file=None, engine=None, backend="npu"):
    interactive = file is None or engine is None
    _banner()
    # 一进系统先做环境自检:asnumpy / CANN / NPU 装好没,没装好当场失败
    backend_name = _startup_check(backend, interactive)
    if backend_name is None:
        console.print("[red]✗ 环境不满足且未降级,向导终止。[/]")
        return 3
    path = _ask_file(file)
    if path is None:
        return 1
    engine_name = _ask_engine(engine)
    if engine_name is None:
        return 1

    _print_roster(engine_name)      # 让用户看清 5 个智能体各自用什么模型/引擎
    source = open(path, encoding="utf-8").read()
    try:
        data = _run_pipeline(source, path, engine_name, backend_name)
    except LlmError as e:
        console.print(Panel(f"{e}", title="[red]LLM 迁移终止(strict)[/]", border_style="red"))
        return 4
    _feedback(data)
    # 每次迁移落地一份带时间戳的 Markdown 报告(默认 reports/ 目录)
    try:
        report_path = save_migration_report(_report_info(data))
        console.print(f"[green]✓[/] 迁移报告已落盘:[bold]{report_path}[/]")
    except Exception as e:
        console.print(f"[yellow]⚠ 报告落盘失败(不影响迁移):{e}[/]")
    if data.get("stopped"):
        return 1
    return 0 if data["report"].numerically_consistent else 2
