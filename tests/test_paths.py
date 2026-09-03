from pathlib import Path

from numpilot.paths import (default_benchmark_dir, default_bootstrap_receipt,
                            default_gap_dir, default_wizard_dir, reports_dir,
                            runs_dir, state_home)


def test_paths_follow_xdg_state_home(monkeypatch, tmp_path):
    monkeypatch.delenv("NUMPILOT_HOME", raising=False)
    monkeypatch.delenv("NUMPILOT_STATE_HOME", raising=False)
    monkeypatch.delenv("NUMPILOT_BOOTSTRAP_RECEIPT", raising=False)
    monkeypatch.delenv("NUMPILOT_RUNS_DIR", raising=False)
    monkeypatch.delenv("NUMPILOT_REPORTS_DIR", raising=False)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))

    assert state_home() == tmp_path / "numpilot"
    assert runs_dir() == tmp_path / "numpilot" / "runs"
    assert reports_dir() == tmp_path / "numpilot" / "reports"
    assert default_benchmark_dir() == runs_dir() / "benchmark"
    assert default_wizard_dir() == runs_dir() / "wizard"
    assert default_gap_dir() == runs_dir() / "gaps"
    assert default_bootstrap_receipt() == state_home() / "bootstrap" / "receipt.json"


def test_explicit_paths_override_xdg(monkeypatch, tmp_path):
    state = tmp_path / "state"
    runs = tmp_path / "external-runs"
    reports = tmp_path / "external-reports"
    monkeypatch.setenv("NUMPILOT_STATE_HOME", str(state))
    monkeypatch.setenv("NUMPILOT_RUNS_DIR", str(runs))
    monkeypatch.setenv("NUMPILOT_REPORTS_DIR", str(reports))

    assert state_home() == state
    assert runs_dir() == runs
    assert reports_dir() == reports


def test_runtime_root_is_descriptive_alias_for_persistent_home(
    monkeypatch, tmp_path
):
    monkeypatch.delenv("NUMPILOT_HOME", raising=False)
    monkeypatch.setenv("NUMPILOT_RUNTIME_ROOT", str(tmp_path))

    assert state_home() == tmp_path / ".numpilot-state"
    assert runs_dir() == tmp_path / ".numpilot-state" / "runs"


def test_numpilot_home_takes_precedence_over_runtime_root(
    monkeypatch, tmp_path
):
    primary = tmp_path / "primary"
    secondary = tmp_path / "secondary"
    monkeypatch.setenv("NUMPILOT_HOME", str(primary))
    monkeypatch.setenv("NUMPILOT_RUNTIME_ROOT", str(secondary))

    assert state_home() == primary / ".numpilot-state"


def test_path_resolution_does_not_create_directories(monkeypatch, tmp_path):
    target = tmp_path / "not-created"
    monkeypatch.setenv("NUMPILOT_STATE_HOME", str(target))

    assert state_home() == target
    assert not Path(target).exists()
