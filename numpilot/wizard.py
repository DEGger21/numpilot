"""交互式迁移向导:启动界面 → 选代码 → 选引擎 → 进度迁移 → 迁移反馈。

依赖 rich 做终端交互(进度条/彩框/表格)。向导只负责迁移、产物级正确性和
capability provenance；性能必须通过独立的 ``bench-cpu`` / ``bench-npu``
严格门禁发布，避免在交互流程中形成不可审计的加速比。
"""
import ast
import os
import sys
import uuid

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich import box

from numpilot.backends.base import select_backend
from numpilot.cli_ui import AgentProgress, compact_diagnostic
from numpilot.contracts import MigrationResult
from numpilot.coverage import is_custom
from numpilot.engines.llm import LlmError
from numpilot.env_check import Env
from numpilot.llm_config import (LlmConfigError, configure_llm_interactive,
                                 load_llm_config)
from numpilot.orchestrator import Orchestrator, _make_engine
from numpilot.paths import default_benchmark_dir, default_wizard_dir
from numpilot.report import agent_roster, save_migration_report

console = Console()


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


def custom_op_breakdown(rewrites):
    """挑出 AsNumpy 的扩展 API；这不代表本次新生成了自定义算子。"""
    return {op: n for op, n in rewrites.items() if is_custom(op)}


def _print_roster(engine_name):
    tb = Table(title="多智能体阵容(本次任务)", box=box.SIMPLE_HEAVY, title_style="bold cyan")
    tb.add_column("智能体", style="bold")
    tb.add_column("职责")
    tb.add_column("模型 / 引擎", style="magenta")
    for name, role, model in agent_roster(engine_name):
        tb.add_row(name, role, model)
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
    interactive_choice = engine is None
    if engine is None:
        engine = Prompt.ask("[bold]选择迁移引擎[/]  [dim](ast=确定性改写/离线, llm=已配置大模型智能体)[/]",
                            choices=["ast", "llm"], default="ast")
    if engine == "llm":
        status = load_llm_config()
        if not status.configured:
            can_prompt = interactive_choice and sys.stdin.isatty() and sys.stdout.isatty()
            if not can_prompt:
                raise LlmConfigError(
                    status.summary() + "；请先运行 python -m numpilot configure-llm"
                )
            if not Confirm.ask("尚未配置 LLM，是否现在在 NumPilot 内填写?", default=True):
                return None
            status = configure_llm_interactive()
        console.print(f"[green]✓[/] 已载入大模型配置(model=[bold]{status.model or '?'}[/],"
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
        detail = compact_diagnostic(it.name, it.detail)
        tb.add_row(f"[{c}]{_MARK.get(it.status, '?')} {it.name}[/]", f"[dim]{detail}[/]")
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


def _run_pipeline(source, path, engine_name, backend_name, out_dir):
    """跑六角色主链:Analyzer → Planner → [Gap Filler] → Code Migrator →
    全程序验证 → Diagnostics → Reporter。"""
    data = {"path": path, "engine": engine_name, "backend": backend_name}
    orchestrator = Orchestrator(
        backend=select_backend(backend_name),
        engine=_make_engine(engine_name),
    )
    with AgentProgress(
        stream=console.file,
        context={
            "source": path,
            "output": str(out_dir),
            "backend": backend_name,
            "engine": engine_name or "llm",
            "benchmark": False,
            "mode": "wizard",
        },
    ) as progress:
        result = orchestrator.run_source(
            source, path, out_dir, on_event=progress.on_event
        )
        if result["status"] not in {"pass", "unsupported"}:
            progress.error()

    data["analysis"] = result["analysis"]
    data["plan"] = result.get("plan")
    data["route"] = result.get("route")
    data["roles"] = result.get("roles", [])
    if result["status"] == "unsupported":
        data["stopped"] = "unsupported"
        return data
    data["migration"] = MigrationResult(
        engine=result["engine"],
        migrated_source=result["migrated_source"],
        rewrites=result["rewrites"],
        expected_rewrites=result["expected_rewrites"],
        model_receipt=result["model_receipt"],
        model_identity=result["model_identity"],
    )
    data["report"] = result["report"]
    data["migrated_path"] = result["migrated_path"]
    data["validation_evidence"] = result.get("validation_evidence")

    # capability/provenance 只判资格，不在向导中顺手发布性能。
    provenance = result["report"].backend_provenance or {}
    data["performance_valid"] = False
    if provenance:
        ok = provenance.get("ok")
        audit_id = provenance.get("audit_id", "?")
        gate = (
            ("通过" if ok else "未通过")
            if provenance.get("strict_npu")
            else "未执行(shim/非 strict)"
        )
        console.print(
            f"   ↳ [cyan]验证步骤[/] capability audit={audit_id} · "
            f"strict-NPU 门禁={gate}"
        )
    else:
        console.print("   ↳ [yellow]无 capability provenance，不具备性能发布资格[/]")
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
    model_label = os.getenv("NUMPILOT_LLM_MODEL", "LLM")
    tb.add_row("迁移引擎", data["engine"] +
               (f" ({model_label} 智能体)" if data["engine"] == "llm" else " (确定性改写)"))
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
        f"python -m numpilot bench-cpu {data['path']} --out {data['benchmark_dir']}\n"
        f"python -m numpilot bench-npu {data['path']} --out {data['benchmark_dir']}\n"
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
    info["model_receipt"] = mig.model_receipt
    info["model_identity"] = mig.model_identity
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
def run_wizard(
    file=None,
    engine=None,
    backend="npu",
    out_dir=None,
    reports_dir=None,
):
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
    try:
        engine_name = _ask_engine(engine)
    except LlmConfigError as e:
        console.print(Panel(f"{e}", title="[red]LLM 配置失败[/]", border_style="red"))
        return 4
    if engine_name is None:
        return 1

    _print_roster(engine_name)      # 让用户看清六角色各自使用的实际执行方式
    source = open(path, encoding="utf-8").read()
    try:
        data = _run_pipeline(
            source,
            path,
            engine_name,
            backend_name,
            out_dir or str(default_wizard_dir() / ("run-" + uuid.uuid4().hex)),
        )
    except LlmError as e:
        console.print(Panel(f"{e}", title="[red]LLM 迁移终止(strict)[/]", border_style="red"))
        return 4
    data["benchmark_dir"] = str(default_benchmark_dir())
    _feedback(data)
    # 每次迁移落地一份带时间戳的 Markdown 报告。
    try:
        report_path = save_migration_report(
            _report_info(data), reports_dir=reports_dir
        )
        console.print(f"[green]✓[/] 迁移报告已落盘:[bold]{report_path}[/]")
    except Exception as e:
        console.print(f"[yellow]⚠ 报告落盘失败(不影响迁移):{e}[/]")
    if data.get("stopped"):
        return 1
    return 0 if data["report"].numerically_consistent else 2
