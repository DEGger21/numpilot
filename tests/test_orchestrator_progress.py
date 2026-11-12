"""CPU-only lifecycle truth: real operation boundaries, failures and repairs."""
from dataclasses import FrozenInstanceError
import json

import pytest

from numpilot import orchestrator as orchestration
from numpilot.backends.shim import ShimBackend
from numpilot.engines.ast_rewrite import AstRewriteEngine
from numpilot.orchestrator import Orchestrator
from numpilot.progress import ProgressEmitter, RoleProgressEvent
from numpilot.roles import ROLE_ORDER, Role


SOURCE = (
    "import numpy as np\n"
    "def build_inputs(): return (np.array([1, 2, 3, 4], dtype=np.float32),)\n"
    "def forward(x): return np.sqrt(x)\n"
)
GAP_SOURCE = SOURCE.replace("np.sqrt(x)", "np.nextafter(x, x)")


def assert_closed(events):
    pending = set()
    for event in events:
        key = event.component, event.attempt
        if event.state == "running":
            assert key not in pending
            pending.add(key)
        else:
            pending.discard(key)
    assert not pending


def test_lifecycle_starts_before_real_operations_and_preserves_final_role_contract(tmp_path, monkeypatch):
    events, records, operations, completions = [], [], [], []
    inventories = []
    original_inventory = orchestration.collect_gap_inventory

    def collect(analysis):
        assert events[-1].role is Role.ANALYZER and events[-1].state == "running"
        inventory = original_inventory(analysis)
        inventories.append(inventory)
        return inventory

    monkeypatch.setattr(orchestration, "collect_gap_inventory", collect)
    targets = [
        (orchestration.AnalysisAgent, "analyze", "analyzer"),
        (orchestration.GapFillerAgent, "plan", "gap_filler"),
        (orchestration.PlannerAgent, "plan", "planner"),
        (orchestration.MigrationAgent, "migrate", "code_migrator"),
        (orchestration.VerificationAgent, "verify", "verification"),
        (orchestration.DiagnosticsAgent, "diagnose", "diagnostics"),
    ]
    for cls, name, component in targets:
        original = getattr(cls, name)

        def checked(*args, _original=original, _component=component, **kwargs):
            assert events[-1].component == _component and events[-1].state == "running"
            if _component == "planner":
                assert len(inventories) == 1 and args[2] is inventories[0]
            operations.append(_component)
            return _original(*args, **kwargs)

        monkeypatch.setattr(cls, name, checked)

    out = tmp_path / "output"

    def observe(event):
        events.append(event)
        if event.role is Role.REPORTER and event.state == "completed":
            completions.append({name: (out / name).is_file() for name in ("migrated.py", "report.json", "report.md")})

    result = Orchestrator(backend=ShimBackend()).run_source(
        SOURCE, "example.py", out_dir=str(out), on_event=observe, on_record=records.append)
    assert result["status"] == "pass"
    assert completions == [{name: True for name in ("migrated.py", "report.json", "report.md")}]
    assert json.loads((out / "report.json").read_text())["roles"][-1]["status"] == "ok"
    assert operations == ["analyzer", "planner", "gap_filler", "code_migrator", "verification", "diagnostics"]
    starts = [event.role for event in events if event.state == "running" and event.role is not None]
    assert starts == list(ROLE_ORDER)
    assert [record.role for record in records] == list(ROLE_ORDER)
    assert [record["role"] for record in result["roles"]] == [role.value for role in ROLE_ORDER]
    assert [event.sequence for event in events] == list(range(1, len(events) + 1))
    assert [event.elapsed_seconds for event in events] == sorted(event.elapsed_seconds for event in events)
    assert all(event.execution_mode == "static" for event in events)
    gates = [event for event in events if event.role is None]
    assert len(gates) == 2 and all(event.kind == "validation" and event.component == "verification" for event in gates)
    assert {event.role for event in events if event.role is not None} == set(ROLE_ORDER)
    assert_closed(events)


def test_gap_inventory_drift_after_planning_stops_before_migration(monkeypatch):
    from numpilot.agents.gap_filler import GapPlan, GapTask

    def changed_inventory(_self, analysis):
        return GapPlan((GapTask("unexpected", "api_gap", "gap_filler", "gap_required"),))

    monkeypatch.setattr(orchestration.GapFillerAgent, "plan", changed_inventory)
    events = []
    with pytest.raises(RuntimeError, match="gap inventory changed after planning"):
        Orchestrator(backend=ShimBackend()).run_source(SOURCE, "example.py", on_event=events.append)
    starts = [event.role for event in events if event.state == "running"]
    assert starts == [Role.ANALYZER, Role.PLANNER, Role.GAP_FILLER]
    assert events[-1].role is Role.GAP_FILLER and events[-1].state == "failed"
    assert_closed(events)


def test_observer_errors_do_not_change_computation_or_final_records(tmp_path):
    def broken(_):
        raise RuntimeError("display unavailable")

    orchestrator = Orchestrator(backend=ShimBackend())
    expected = orchestrator.run_source(SOURCE, "example.py", str(tmp_path / "expected"))
    actual = orchestrator.run_source(SOURCE, "example.py", str(tmp_path), on_event=broken, on_record=broken)
    for key in ("status", "route", "migrated_source", "rewrites", "roles", "validation_evidence"):
        assert actual[key] == expected[key]


@pytest.mark.parametrize("component", ["analyzer", "gap_filler", "planner", "code_migrator", "verification", "diagnostics", "reporter"])
def test_each_operation_closes_on_exception_without_exposing_error_text(tmp_path, monkeypatch, component):
    targets = {
        "analyzer": (orchestration.AnalysisAgent, "analyze"),
        "gap_filler": (orchestration.GapFillerAgent, "plan"),
        "planner": (orchestration.PlannerAgent, "plan"),
        "code_migrator": (orchestration.MigrationAgent, "migrate"),
        "verification": (orchestration.VerificationAgent, "verify"),
        "diagnostics": (orchestration.DiagnosticsAgent, "diagnose"),
        "reporter": (Orchestrator, "_maybe_write"),
    }

    def fail(*args, **kwargs):
        raise OSError("provider-secret-123 request body must not reach progress")

    cls, name = targets[component]
    monkeypatch.setattr(cls, name, fail)
    events, records = [], []
    with pytest.raises(OSError, match="provider-secret"):
        Orchestrator(backend=ShimBackend()).run_source(
            SOURCE, "example.py", str(tmp_path), on_event=events.append, on_record=records.append)
    assert events[-1].component == component and events[-1].state == "failed"
    assert "provider-secret" not in json.dumps([event.to_dict() for event in events])
    assert not any(event.role is Role.REPORTER and event.state == "completed" for event in events)
    assert not any(record.role is Role.REPORTER for record in records)
    assert_closed(events)


def test_unsupported_route_skips_unexecuted_roles_and_does_not_run_verifier(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("no candidate exists")

    monkeypatch.setattr(orchestration.MigrationAgent, "migrate", forbidden)
    monkeypatch.setattr(orchestration.VerificationAgent, "verify", forbidden)
    events = []
    result = Orchestrator(backend=ShimBackend()).run_source(GAP_SOURCE, "gap.py", on_event=events.append)
    assert result["status"] == "unsupported"
    states = {event.component: event.state for event in events}
    assert states["code_migrator"] == states["verification"] == states["diagnostics"] == "skipped"
    assert states["gap_filler"] == states["planner"] == "blocked"
    assert states["reporter"] == "completed"
    assert_closed(events)


class RepairEngine:
    # A fake local engine exercises mode reporting without any model request.
    name = "llm"

    def __init__(self, succeeds):
        self.succeeds = succeeds

    def rewrite(self, source, plan):
        return AstRewriteEngine().rewrite(source, plan).replace("ap.sqrt", "ap.square")

    def repair(self, source, plan, previous_source, diagnosis):
        return AstRewriteEngine().rewrite(source, plan) if self.succeeds else self.rewrite(source, plan)


@pytest.mark.parametrize("succeeds,attempts,status", [(True, 2, "pass"), (False, 4, "fail")])
def test_repair_events_show_every_real_attempt_without_changing_final_records(succeeds, attempts, status):
    events, records = [], []
    result = Orchestrator(backend=ShimBackend(), engine=RepairEngine(succeeds)).run_source(
        SOURCE, "repair.py", on_event=events.append, on_record=records.append)
    assert result["status"] == status and result["repair_attempts"] == attempts - 1
    for component in ("code_migrator", "verification", "diagnostics"):
        starts = [event for event in events if event.component == component and event.state == "running"]
        assert [event.attempt for event in starts] == list(range(1, attempts + 1))
    assert all(event.execution_mode == ("llm" if event.role is Role.CODE_MIGRATOR else "static") for event in events)
    assert [record.role for record in records] == list(ROLE_ORDER)
    assert_closed(events)


def test_run_forwards_both_observer_interfaces(tmp_path):
    source = tmp_path / "example.py"
    source.write_text(SOURCE)
    events, records = [], []
    result = Orchestrator(backend=ShimBackend()).run(str(source), on_event=events.append, on_record=records.append)
    assert result["status"] == "pass" and events and len(records) == 6


def test_progress_event_is_immutable_json_roundtrippable_and_gate_is_not_a_role():
    emitter = ProgressEmitter()
    for role, component, mode in [(Role.ANALYZER, None, "static"), (None, "verification", "static"),
                                   (None, "workflow", "external")]:
        event = emitter.emit(role, "running", "处理阶段", component=component, execution_mode=mode)
        assert RoleProgressEvent.from_dict(json.loads(json.dumps(event.to_dict()))) == event
        with pytest.raises(FrozenInstanceError):
            event.state = "failed"


@pytest.mark.parametrize("field,value", [
    ("schema", "foreign"), ("schema_version", True), ("schema_version", 2), ("type", "receipt"),
    ("role", "seventh_agent"), ("role", 1), ("component", "unknown"), ("kind", "system"),
    ("state", "qualified"), ("execution_mode", "magic"), ("message", ""), ("message", "x" * 513),
    ("message", "secret\ntext"), ("message", "\x1b[1mtext"), ("attempt", True), ("attempt", 0),
    ("attempt", 10_001), ("sequence", 0), ("sequence", 10_000_001), ("elapsed_seconds", True),
    ("elapsed_seconds", -1), ("elapsed_seconds", float("inf")), ("elapsed_seconds", float("nan")),
    ("elapsed_seconds", 8_640_001),
])
def test_transport_rejects_invalid_event_values(field, value):
    data = ProgressEmitter().emit(Role.ANALYZER, "running", "分析源码").to_dict()
    data[field] = value
    with pytest.raises(ValueError):
        RoleProgressEvent.from_dict(data)


def test_transport_rejects_extra_missing_fields_and_llm_verification():
    data = ProgressEmitter().emit(None, "running", "执行验证").to_dict()
    for candidate in (dict(data, unexpected="field"), {k: v for k, v in data.items() if k != "role"},
                      dict(data, execution_mode="llm")):
        with pytest.raises(ValueError):
            RoleProgressEvent.from_dict(candidate)


def test_cancelled_operation_closes_and_preserves_control_signal():
    events = []
    with pytest.raises(KeyboardInterrupt):
        with ProgressEmitter(events.append).operation(Role.CODE_MIGRATOR, "执行改写"):
            raise KeyboardInterrupt()
    assert [event.state for event in events] == ["running", "failed"]


def test_reporter_completion_callback_sees_taskbook_and_report_on_gap_route(tmp_path):
    events, completions = [], []

    def observe(event):
        events.append(event)
        if event.role is Role.REPORTER and event.state == "completed":
            completions.append({name: (tmp_path / name).is_file() for name in
                                ("report.json", "GAP_TASKBOOK.json", "GAP_TASKBOOK.md")})

    result = Orchestrator(backend=ShimBackend()).run_source(GAP_SOURCE, "gap.py", str(tmp_path), on_event=observe)
    assert result["status"] == "unsupported" and result["gap_taskbook"]
    assert completions == [{name: True for name in ("report.json", "GAP_TASKBOOK.json", "GAP_TASKBOOK.md")}]
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["gap_taskbook"]["schema"] == "numpilot.gap-taskbook-ref"
    assert events[-1].role is Role.REPORTER and events[-1].state == "completed"
    assert_closed(events)


OBJECT_DTYPE_SOURCE = (
    "import numpy as np\n"
    "def build_inputs():\n"
    "    return {'x': np.array([1, 2], dtype=object)}\n"
    "def forward(x):\n"
    "    return {'out': np.multiply(x, x)}\n"
)


def test_contract_run_forwards_strict_input_types(tmp_path):
    path = tmp_path / "object_dtype.py"
    path.write_text(OBJECT_DTYPE_SOURCE, encoding="utf-8")
    result = Orchestrator(
        backend=ShimBackend(), engine=AstRewriteEngine()
    ).run(str(path), strict_input_types=True)

    assert result["status"] == "unsupported"
    assert "multiply" in result["unsupported"]
