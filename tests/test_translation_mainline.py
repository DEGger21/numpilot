"""Planner hand-off and configured-model prompt contract tests.

These tests never contact a model: they assert what the framework *hands to*
the model and how the Planner routes translation work.
"""
import json

from numpilot.agents.analysis import AnalysisAgent
from numpilot.agents.gap_filler import collect_gap_inventory
from numpilot.agents.planner import ROUTE_DIRECT, PlannerAgent
from numpilot.asnumpy_guide import render_asnumpy_guide
from numpilot.engines.llm import LlmEngine
from numpilot.program_preparation import prepare_program
from numpilot.translation_taskbook import (
    SCHEMA,
    build_translation_taskbook,
    render_translation_taskbook_markdown,
    validate_translation_taskbook,
)

STRUCTURED_PROGRAM = """
import numpy as np


def helper(x):
    return np.abs(x)


def forward(x):
    for _ in range(2):
        x = helper(x)
    return {'y': x}
"""


def _plan(source):
    preparation = prepare_program(source)
    analysis = AnalysisAgent().analyze(
        preparation.analysis_source, "program.py"
    )
    inventory = collect_gap_inventory(analysis)
    return preparation, analysis, inventory, PlannerAgent().plan(
        analysis,
        inventory,
        source=source,
        prepared_source=preparation.analysis_source,
        preparation=preparation,
        source_path="program.py",
    )


def test_preparation_is_explicit_and_keeps_original_as_oracle():
    preparation, analysis, _, plan = _plan(STRUCTURED_PROGRAM)

    assert preparation.applied
    assert preparation.analysis_source != preparation.original_source
    kinds = {item.kind for item in preparation.rewrites}
    assert "inline_helper" in kinds
    assert "unroll_for" in kinds
    assert analysis.migratable
    assert plan.route == ROUTE_DIRECT
    assert plan.translation_tasks == ()

    taskbook = plan.translation_taskbook
    assert taskbook["preparation"]["applied"] is True
    assert taskbook["source"]["sha256"] != taskbook["source"]["prepared_sha256"]
    assert taskbook["acceptance"]["oracle_source_sha256"] == taskbook["source"]["sha256"]


def test_surface_only_ssa_normalization_is_reported_as_applied():
    source = (
        "import numpy as np\n"
        "\n"
        "def forward(x, y):\n"
        "    z = np.add(x, y)\n"
        "    return {'out': np.sqrt(np.square(z))}\n"
    )
    preparation = prepare_program(source)

    assert preparation.analysis_source != preparation.original_source
    assert preparation.rewrites == ()
    assert preparation.normalized
    assert preparation.applied

    payload = preparation.to_dict()
    assert payload["applied"] is True
    assert payload["analysis_source_changed"] is True
    assert payload["normalized"] is True
    assert any("SSA" in note for note in payload["notes"])


def test_taskbook_is_sealed_deterministic_and_renderable():
    preparation = prepare_program(STRUCTURED_PROGRAM)
    analysis = AnalysisAgent().analyze(
        preparation.analysis_source, "program.py"
    )
    kwargs = dict(
        analysis=analysis,
        route=ROUTE_DIRECT,
        source=STRUCTURED_PROGRAM,
        prepared_source=preparation.analysis_source,
        preparation=preparation,
        source_path="program.py",
    )
    first = build_translation_taskbook(**kwargs)
    second = build_translation_taskbook(**kwargs)

    assert first == second
    assert first["schema"] == SCHEMA
    validate_translation_taskbook(first)
    assert first["taskbook_id"].startswith("translation-taskbook-")
    assert first["callsites"]
    assert all(
        site["target_api"] and site["target_api"].startswith("asnumpy.")
        for site in first["callsites"]
    )
    markdown = render_translation_taskbook_markdown(first)
    assert first["taskbook_id"] in markdown
    assert "逐调用点任务" in markdown


def test_code_migrator_tasks_stay_on_the_direct_route():
    source = (
        "import numpy as np\n"
        "def forward(x):\n"
        "    return np.abs(x)\n"
    )
    # Analyze the unprepared spelling on purpose: the alias is a translation
    # task, not a missing runtime capability.
    analysis = AnalysisAgent().analyze(source, "program.py")
    inventory = collect_gap_inventory(analysis)
    plan = PlannerAgent().plan(
        analysis, inventory, source=source, source_path="program.py"
    )

    assert not analysis.migratable
    assert inventory.blocking
    assert plan.route == ROUTE_DIRECT
    assert plan.translation_tasks
    assert all(task["route"] == "code_migrator" for task in plan.translation_tasks)


def test_planner_taskbook_marks_alias_translation():
    source = (
        "import numpy as np\n"
        "def forward(x):\n"
        "    return np.abs(x)\n"
    )
    analysis = AnalysisAgent().analyze(source, "program.py")
    inventory = collect_gap_inventory(analysis)
    plan = PlannerAgent().plan(
        analysis, inventory, source=source, source_path="program.py"
    )
    site = plan.translation_taskbook["callsites"][0]

    assert site["name"] == "abs"
    assert site["status"] == "alias_required"
    assert site["target_api"] == "asnumpy.absolute"


def test_taskbook_names_target_api_for_python_operator():
    source = (
        "import numpy as np\n"
        "def forward(x):\n"
        "    y = x + 1.0\n"
        "    return {'out': np.sqrt(y)}\n"
    )
    _, _, _, plan = _plan(source)

    assert plan.route == ROUTE_DIRECT
    site = next(
        site
        for site in plan.translation_taskbook["callsites"]
        if site["name"] == "python_operator.Add"
    )
    assert site["target_api"] == "asnumpy.add"
    assert any("asnumpy.add" in note for note in site["semantics"])
    assert any("device arrays" in note for note in site["semantics"])

    task = next(
        task
        for task in plan.translation_taskbook["gap_tasks"]
        if task["name"] == "python_operator.Add"
    )
    assert task["target"] == "add"


def test_llm_prompt_carries_guide_and_taskbook(monkeypatch):
    preparation, analysis, inventory, plan = _plan(STRUCTURED_PROGRAM)
    engine = LlmEngine()
    captured = {}

    def fake_request(system, user_msg):
        captured["system"] = system
        captured["user"] = user_msg
        return "import numpy as np\nimport asnumpy as ap\ndef forward(x):\n    return ap.absolute(x)\n"

    monkeypatch.setattr(engine, "_request_code", fake_request)
    code = engine.rewrite(
        preparation.analysis_source,
        type("Plan", (), {"placement": {"absolute": "NPU"}})(),
        taskbook=plan.translation_taskbook,
    )

    assert "ap.absolute" in code
    assert render_asnumpy_guide() in captured["system"]
    assert "AsNumpy 代码格式" in captured["system"]
    assert plan.translation_taskbook["taskbook_id"] in captured["user"]
    assert "迁移任务书" in captured["user"]


def test_llm_repair_prompt_carries_diagnosis_and_previous_candidate(monkeypatch):
    preparation, analysis, inventory, plan = _plan(STRUCTURED_PROGRAM)
    engine = LlmEngine()
    captured = {}

    def fake_request(system, user_msg):
        captured["system"] = system
        captured["user"] = user_msg
        return "import numpy as np\nimport asnumpy as ap\ndef forward(x):\n    return ap.absolute(x)\n"

    monkeypatch.setattr(engine, "_request_code", fake_request)
    previous = "import numpy as np\ndef forward(x):\n    return x\n"
    diagnosis = {
        "category": "numeric_mismatch",
        "suggested_role": "code_migrator",
        "summary": "absolute was not rewritten",
    }
    engine.repair(
        preparation.analysis_source,
        type("Plan", (), {"placement": {"absolute": "NPU"}})(),
        previous,
        diagnosis,
        taskbook=plan.translation_taskbook,
    )

    assert previous in captured["user"]
    assert "numeric_mismatch" in captured["user"]
    assert "上一版迁移没有通过确定性验证" in captured["user"]
    assert plan.translation_taskbook["taskbook_id"] in captured["user"]
    assert json.loads(json.dumps(diagnosis))["summary"] in captured["user"]
