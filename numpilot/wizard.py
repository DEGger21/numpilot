"""交互式迁移向导:启动界面 → 选代码 → 选引擎 → 进度迁移 → 迁移反馈。

依赖 rich 做终端交互(进度条/彩框/表格)。核心步骤复用已有智能体与 bench:
- 迁移/校验在向导进程内跑(满线程, 用于正确性 + asnumpy 真机证据 + NPU 测速);
- numpy 单核基线用**子进程** `python -m numpilot bench-cpu`(进程隔离, 干净钉 1 核, 不污染 CANN);
- speedup = t_cpu(numpy@1核) / t_npu(asnumpy@NPU)。
"""
import ast
import json
import os
import re
import subprocess
import sys

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
    """从 {op:count} 里挑出 asnumpy 自定义 NPU 算子(relu/softmax/gelu/angle)。"""
    return {op: n for op, n in rewrites.items() if is_custom(op)}


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


def _resolve_backend(backend, interactive):
    """npu 后端做前置环境自检;不满足时(交互)询问降级 shim,(非交互)直接失败。"""
    if backend != "npu":
        return backend
    rep = Env.check(backend="npu")
    if not rep.errors:
        return "npu"
    console.print(Panel(rep.render(), title="[red]环境自检未通过[/]", border_style="red"))
    if interactive and Confirm.ask("是否降级到 shim 仿真后端继续?[dim](无真机性能,仅演示流程)[/]",
                                    default=False):
        return "shim"
    return None


def _run_pipeline(source, path, engine_name, backend_name):
    """跑 分析→决策→迁移→校验→CPU基线,返回收集的数据 dict。中途失败抛异常。"""
    data = {"path": path, "engine": engine_name, "backend": backend_name}
    steps = ["分析算子", "决策卸载", "迁移改写", "真机校验 + NPU 测速", "numpy 单核基线"]
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                  BarColumn(bar_width=24), TextColumn("{task.completed}/{task.total}"),
                  TimeElapsedColumn(), console=console) as prog:
        t = prog.add_task("准备中", total=len(steps))

        # [1] 分析
        prog.update(t, description=f"[1/5] {steps[0]}")
        analysis = AnalysisAgent().analyze(source, path)
        data["analysis"] = analysis
        distinct = sorted({o.name for o in analysis.op_uses})
        if not analysis.migratable:
            prog.console.print(f"   ↳ [red]发现 {len(analysis.unsupported)} 个覆盖集缺口:"
                               f"{', '.join(analysis.unsupported)} —— 无法迁移[/]")
            prog.advance(t)
            data["stopped"] = "unsupported"
            return data
        prog.console.print(f"   ↳ {len(analysis.op_uses)} 处 numpy 调用 · {len(distinct)} 个算子 · 全部在覆盖集 ✓")
        prog.advance(t)

        # [2] 决策
        prog.update(t, description=f"[2/5] {steps[1]}")
        plan = DecisionAgent().plan(analysis)
        data["plan"] = plan
        npu_ops = sorted(o for o, p in plan.placement.items() if p == "NPU")
        prog.console.print(f"   ↳ 按规模阈值卸载:NPU {len(npu_ops)} 个算子,其余留 CPU")
        prog.advance(t)

        # [3] 迁移(LLM 时较慢)
        prog.update(t, description=f"[3/5] {steps[2]}"
                    + (" · 请求 glm-5.1…" if engine_name == "llm" else ""))
        migration = MigrationAgent(_make_engine(engine_name)).migrate(source, plan)
        data["migration"] = migration
        os.makedirs(_OUT_DIR, exist_ok=True)
        mig_path = os.path.join(_OUT_DIR, "migrated.py")
        with open(mig_path, "w", encoding="utf-8") as f:
            f.write(migration.migrated_source)
        data["migrated_path"] = mig_path
        prog.console.print(f"   ↳ 引擎={migration.engine} · 替换 {migration.total_rewrites} 处 np.* → ap.*")
        prog.advance(t)

        # [4] 校验 + NPU 测速
        prog.update(t, description=f"[4/5] {steps[3]}")
        report = VerificationAgent(select_backend(backend_name)).verify(
            source, migration.migrated_source, measure=True)
        data["report"] = report
        mark = "✓" if report.numerically_consistent else "✗"
        prog.console.print(f"   ↳ 数值一致 {mark} max_rel_diff={report.max_rel_diff:.2e} · "
                           f"asnumpy@{report.backend} 时延 {report.t_mig_ms:.1f} ms")
        prog.advance(t)

        # [5] numpy 单核基线(子进程, 干净钉 1 核)
        prog.update(t, description=f"[5/5] {steps[4]}")
        t_cpu = _cpu_baseline_subprocess(path)
        data["t_cpu_ms"] = t_cpu
        if t_cpu is not None:
            prog.console.print(f"   ↳ numpy@CPU(1 核)基线 {t_cpu:.1f} ms")
        else:
            prog.console.print("   ↳ [yellow]CPU 基线获取失败,跳过加速比[/]")
        prog.advance(t)
    return data


def _cpu_baseline_subprocess(path):
    """子进程跑 bench-cpu(钉 1 核), 读 cpu_baseline.json 取 t_cpu_ms。失败返回 None。"""
    try:
        subprocess.run([sys.executable, "-m", "numpilot", "bench-cpu", path, "--out", _OUT_DIR],
                       capture_output=True, text=True, timeout=600, check=True)
        with open(os.path.join(_OUT_DIR, "cpu_baseline.json"), encoding="utf-8") as f:
            return json.load(f)["t_cpu_ms"]
    except Exception:
        return None


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
    tb.add_row("算子分析", f"{len(a.op_uses)} 处调用 · {len({o.name for o in a.op_uses})} 个算子 · 全部在覆盖集 ✓")
    replaced = "  ".join(f"{op}×{n}" for op, n in
                         sorted(mig.rewrites.items(), key=lambda kv: (-kv[1], kv[0])))
    tb.add_row("算子替换", f"共 [bold]{mig.total_rewrites}[/] 处 np.* → ap.*\n{replaced}")
    if custom:
        clist = "  ".join(f"{op}×{n}" for op, n in sorted(custom.items(), key=lambda kv: (-kv[1], kv[0])))
        tb.add_row("自定义 NPU 算子", f"接入 [bold green]{len(custom)}[/] 种 / {sum(custom.values())} 处:{clist}\n"
                                  "[dim](relu/softmax/gelu/angle 等 numpy 没有、由 NPU 提供的算子)[/]")
    else:
        tb.add_row("自定义 NPU 算子", "无(本网络未用到 relu/softmax/gelu/angle)")
    tb.add_row("卸载方案", f"NPU: {', '.join(npu_ops) or '—'}\nCPU: {', '.join(cpu_ops) or '—'}")
    if rep.backend_version:
        tb.add_row("执行证据", f"[green]真机 asnumpy {rep.backend_version}[/]\n{rep.backend_module}")
    else:
        tb.add_row("执行证据", "shim 仿真(numpy 承载,非真机)")
    cmark = "[green]一致 ✓[/]" if rep.numerically_consistent else "[red]不一致 ✗[/]"
    tb.add_row("数值校验", f"{cmark}  max_rel_diff={rep.max_rel_diff:.2e}  (tol={rep.tol:.0e})")
    console.print(tb)

    # 性能高亮
    t_cpu, t_npu = data.get("t_cpu_ms"), rep.t_mig_ms
    if data["backend"] == "npu" and t_cpu and t_npu:
        speedup = t_cpu / t_npu
        color = "green" if speedup > 1 else "yellow"
        verdict = "asnumpy@NPU 更快" if speedup > 1 else "numpy 更快"
        console.print(Panel(
            f"numpy@CPU(1 核)  [bold]{t_cpu:.1f} ms[/]\n"
            f"asnumpy@NPU(单卡) [bold]{t_npu:.1f} ms[/]\n"
            f"加速比  [bold {color}]{speedup:.2f}×[/]  ({verdict})\n"
            f"[dim]{rep.n_elements:,} 元素 · 公平口径:CPU 限 1 核 vs NPU 单卡[/]",
            title="[bold]整网性能对比[/]", border_style=color))
    else:
        console.print(Panel("[dim]未做真机性能对比(非 npu 后端或基线缺失)[/]",
                            title="整网性能对比", border_style="dim"))
    console.print(f"[dim]迁移代码已写入:{data['migrated_path']}[/]")


# ───────────────────────── 入口 ─────────────────────────
def run_wizard(file=None, engine=None, backend="npu"):
    interactive = file is None or engine is None
    _banner()
    path = _ask_file(file)
    if path is None:
        return 1
    engine_name = _ask_engine(engine)
    if engine_name is None:
        return 1
    backend_name = _resolve_backend(backend, interactive)
    if backend_name is None:
        console.print("[red]✗ 环境不满足且未降级,向导终止。[/]")
        return 3

    source = open(path, encoding="utf-8").read()
    try:
        data = _run_pipeline(source, path, engine_name, backend_name)
    except LlmError as e:
        console.print(Panel(f"{e}", title="[red]LLM 迁移终止(strict)[/]", border_style="red"))
        return 4
    _feedback(data)
    if data.get("stopped"):
        return 1
    return 0 if data["report"].numerically_consistent else 2
