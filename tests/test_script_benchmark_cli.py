import json
from types import SimpleNamespace

import numpilot.script_benchmark as script_benchmark
from numpilot.__main__ import main


def _payload(status="passed"):
    return {
        "status": status,
        "performance": {
            "performance_valid": status == "passed",
            "cpu_median_ms": 4.0,
            "npu_median_ms": 1.0,
            "median_speedup": 4.0 if status == "passed" else None,
            "conservative_cpu_p10_over_npu_p90": (
                3.0 if status == "passed" else None
            ),
        },
    }


class _Receipt:
    def __init__(self, status="passed"):
        self.status = status
        self.path = "/tmp/benchmark.json"
        self.payload = _payload(status)

    def to_dict(self):
        return json.loads(json.dumps(self.payload))


def _arguments(command="benchmark-script"):
    return [
        command,
        "source.py",
        "--migration-report",
        "run/script-migration-report.json",
        "--output",
        "result",
        "--out",
        "run/benchmark.json",
    ]


def test_benchmark_script_cli_passes_defaults_and_prints_scope(monkeypatch, capsys):
    observed = {}

    def fake(*args, **kwargs):
        observed["args"] = args
        observed["kwargs"] = kwargs
        return _Receipt()

    monkeypatch.setattr(script_benchmark, "benchmark_script", fake)

    assert main(_arguments()) == 0
    output = capsys.readouterr().out
    assert "performance_valid=true" in output
    assert "device-resident" in output
    assert observed["kwargs"]["warmup"] == 3
    assert observed["kwargs"]["samples"] == 20


def test_bench_script_alias_and_json_output(monkeypatch, capsys):
    monkeypatch.setattr(script_benchmark, "benchmark_script", lambda *a, **k: _Receipt())

    assert main(_arguments("bench-script") + ["--json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "passed"


def test_benchmark_cli_failure_with_receipt_never_prints_speedup(monkeypatch, capsys):
    receipt = _Receipt("failed")

    def fail(*args, **kwargs):
        raise script_benchmark.ScriptBenchmarkError(
            "performance_gate_failed", receipt=receipt
        )

    monkeypatch.setattr(script_benchmark, "benchmark_script", fail)

    assert main(_arguments()) == 1
    captured = capsys.readouterr()
    assert "performance_valid=false" in captured.out
    assert "speedup=null" in captured.out


def test_benchmark_cli_preflight_failure_is_exit_two(monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise script_benchmark.ScriptBenchmarkError("invalid_report")

    monkeypatch.setattr(script_benchmark, "benchmark_script", fail)

    assert main(_arguments()) == 2
    assert "benchmark_not_admitted" in capsys.readouterr().err

