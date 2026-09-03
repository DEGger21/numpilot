import hashlib
import json
import stat
import types

import pytest

import numpilot.__main__ as cli
from numpilot.backends.npu import NpuBackend


DIRECT_SOURCE = """\
import numpy as np
x = np.full((2,), 1.0, dtype=np.float32)
y = np.exp(x)
print(y)
"""

FALLBACK_SOURCE = """\
import numpy as np
x = np.full((2, 2), 1.0, dtype=np.float32)
y = np.matmul(x, x)
print(y)
"""


@pytest.fixture(autouse=True)
def _matched_metadata_only_runtime(monkeypatch):
    runtime = types.SimpleNamespace(
        status="matched",
        identity_sha256="a" * 64,
        audit_id="test-runtime-audit-v1",
    )
    monkeypatch.setattr(NpuBackend, "runtime_identity", lambda _self: runtime)

    def forbidden_get_module(_self):
        raise AssertionError("analyze must not load AsNumpy")

    monkeypatch.setattr(NpuBackend, "get_module", forbidden_get_module)

    def forbidden_model_load():
        raise AssertionError("analyze must not load or contact a model")

    monkeypatch.setattr(cli, "_load_llm_or_report", forbidden_model_load)


def test_analyze_json_is_hash_bound_static_and_profile_complete(
    tmp_path, capsys
):
    source = tmp_path / "case.py"
    source.write_text(DIRECT_SOURCE, encoding="utf-8")

    rc = cli.main(["analyze", str(source), "--json"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["analysis_mode"] == "STATIC_AST_ONLY"
    assert payload["source_path"] == str(source.absolute())
    assert payload["source_sha256"] == hashlib.sha256(
        DIRECT_SOURCE.encode("utf-8")
    ).hexdigest()
    assert payload["imports"] == ["import numpy as np"]
    assert payload["decision"] == "DIRECT"
    assert payload["runtime_identity_status"] == "matched"
    assert payload["runtime_identity_sha256"] == "a" * 64
    assert payload["runtime_audit_id"] == "test-runtime-audit-v1"
    assert payload["actual_remote_call"] is False
    assert payload["execution_performed"] is False
    assert payload["performance_valid"] is False
    assert payload["speedup"] is None
    assert [item["source_api"] for item in payload["calls"]] == [
        "numpy.full",
        "numpy.exp",
    ]
    assert payload["calls"][0]["line"] == 2
    assert payload["calls"][0]["facts"]["dtype"] == "float32"
    assert payload["calls"][0]["contract"]["dtype"] == "QUALIFIED"
    assert payload["calls"][0]["decision"] == "DIRECT"
    assert payload["next_action"]


def test_analyze_never_executes_source(tmp_path, capsys):
    marker = tmp_path / "must-not-exist"
    source = tmp_path / "hostile.py"
    source.write_text(
        DIRECT_SOURCE
        + "\nopen(%r, 'w').write('executed')\n" % str(marker),
        encoding="utf-8",
    )

    rc = cli.main(["analyze", str(source), "--json"])

    assert rc == 0
    assert not marker.exists()
    payload = json.loads(capsys.readouterr().out)
    assert payload["decision"] == "BLOCKED"
    assert payload["actual_remote_call"] is False
    assert payload["execution_performed"] is False


def test_analyze_non_direct_succeeds_unless_direct_is_required(
    tmp_path, capsys
):
    source = tmp_path / "fallback.py"
    source.write_text(FALLBACK_SOURCE, encoding="utf-8")

    assert cli.main(["analyze", str(source), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["decision"] == "FALLBACK_CONSENT"
    assert payload["calls"][-1]["route"] == "CPU_FALLBACK"

    assert cli.main([
        "analyze", str(source), "--json", "--require-direct"
    ]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["decision"] == "FALLBACK_CONSENT"


def test_analyze_human_summary_is_concise_and_does_not_dump_runtime(
    tmp_path, capsys, monkeypatch
):
    source = tmp_path / "case.py"
    source.write_text(DIRECT_SOURCE, encoding="utf-8")
    runtime = types.SimpleNamespace(
        status="audit_required",
        identity_sha256="b" * 64,
        audit_id="public-audit-id",
        python_executable="/private/interpreter/path",
        issues=("private runtime detail",),
    )
    monkeypatch.setattr(NpuBackend, "runtime_identity", lambda _self: runtime)

    rc = cli.main(["analyze", str(source)])

    assert rc == 0
    rendered = capsys.readouterr().out
    assert str(source.absolute()) in rendered
    assert "CAPABILITY_AUDIT_REQUIRED" in rendered
    assert "actual_remote_call=false" in rendered
    assert "execution_performed=false" in rendered
    assert "/private/interpreter/path" not in rendered
    assert "private runtime detail" not in rendered


def test_analyze_runtime_probe_failure_fails_closed_without_failing_analysis(
    tmp_path, capsys, monkeypatch
):
    source = tmp_path / "case.py"
    source.write_text(DIRECT_SOURCE, encoding="utf-8")

    def failed_probe(_self):
        raise LookupError("private runtime probe detail")

    monkeypatch.setattr(NpuBackend, "runtime_identity", failed_probe)

    rc = cli.main(["analyze", str(source), "--json"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["decision"] == "CAPABILITY_AUDIT_REQUIRED"
    assert payload["runtime_identity_status"] == "audit_required"
    assert payload["runtime_identity_sha256"] is None
    assert "private runtime probe detail" not in json.dumps(payload)


def test_analyze_accepts_read_only_runtime_identity_mapping(
    tmp_path, capsys, monkeypatch
):
    source = tmp_path / "case.py"
    source.write_text(DIRECT_SOURCE, encoding="utf-8")
    runtime = types.MappingProxyType({
        "status": "matched",
        "identity_sha256": "c" * 64,
        "audit_id": "mapping-audit-v1",
    })
    monkeypatch.setattr(NpuBackend, "runtime_identity", lambda _self: runtime)

    assert cli.main(["analyze", str(source), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["runtime_audit_id"] == "mapping-audit-v1"


def test_analyze_out_is_atomic_write_once_and_private(tmp_path, capsys):
    source = tmp_path / "case.py"
    source.write_text(DIRECT_SOURCE, encoding="utf-8")
    output = tmp_path / "private" / "analysis.json"

    rc = cli.main(["analyze", str(source), "--out", str(output)])

    assert rc == 0
    first = output.read_bytes()
    assert json.loads(first)["decision"] == "DIRECT"
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert stat.S_IMODE(output.parent.stat().st_mode) == 0o700
    assert str(output.absolute()) in capsys.readouterr().out

    rc = cli.main(["analyze", str(source), "--out", str(output)])

    captured = capsys.readouterr()
    assert rc == 2
    assert output.read_bytes() == first
    assert "already exists" in captured.err


@pytest.mark.parametrize("alias_kind", ["same_path", "symlink"])
def test_analyze_out_rejects_source_alias_without_modification(
    tmp_path, capsys, alias_kind
):
    source = tmp_path / "case.py"
    source.write_text(DIRECT_SOURCE, encoding="utf-8")
    before = source.read_bytes()
    if alias_kind == "same_path":
        output = source
    else:
        output = tmp_path / "analysis.json"
        output.symlink_to(source)

    rc = cli.main(["analyze", str(source), "--out", str(output)])

    assert rc == 2
    assert source.read_bytes() == before
    assert "source" in capsys.readouterr().err.lower()


def test_analyze_rejects_insecure_existing_output_parent(tmp_path, capsys):
    source = tmp_path / "case.py"
    source.write_text(DIRECT_SOURCE, encoding="utf-8")
    parent = tmp_path / "shared"
    parent.mkdir(mode=0o755)
    parent.chmod(0o755)
    output = parent / "analysis.json"

    rc = cli.main(["analyze", str(source), "--out", str(output)])

    assert rc == 2
    assert not output.exists()
    assert stat.S_IMODE(parent.stat().st_mode) == 0o755
    assert "0700" in capsys.readouterr().err


@pytest.mark.parametrize("kind", ["missing", "directory", "invalid_utf8"])
def test_analyze_rejects_invalid_source_without_output(tmp_path, capsys, kind):
    source = tmp_path / "case.py"
    if kind == "directory":
        source.mkdir()
    elif kind == "invalid_utf8":
        source.write_bytes(b"import numpy as np\n\xff")
    output = tmp_path / "private" / "analysis.json"

    rc = cli.main([
        "analyze", str(source), "--json", "--out", str(output)
    ])

    assert rc == 2
    assert not output.exists()
    assert capsys.readouterr().err


def test_analyze_refuses_output_parent_symlink(tmp_path, capsys):
    source = tmp_path / "case.py"
    source.write_text(DIRECT_SOURCE, encoding="utf-8")
    real_parent = tmp_path / "real-private"
    real_parent.mkdir(mode=0o700)
    alias_parent = tmp_path / "alias-private"
    alias_parent.symlink_to(real_parent, target_is_directory=True)

    rc = cli.main([
        "analyze", str(source), "--out", str(alias_parent / "analysis.json")
    ])

    assert rc == 2
    assert not (real_parent / "analysis.json").exists()
    assert "real directory" in capsys.readouterr().err


def test_analyze_rejects_source_dotdot_path_that_crosses_symlink(
    tmp_path, capsys
):
    real_source = tmp_path / "case.py"
    real_source.write_text(DIRECT_SOURCE, encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(outside, target_is_directory=True)

    rc = cli.main(["analyze", str(linked / ".." / real_source.name), "--json"])

    assert rc == 2
    assert "symlink" in capsys.readouterr().err


def test_analyze_rejects_output_dotdot_path_that_crosses_symlink(
    tmp_path, capsys
):
    source = tmp_path / "case.py"
    source.write_text(DIRECT_SOURCE, encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(outside, target_is_directory=True)

    rc = cli.main([
        "analyze",
        str(source),
        "--out",
        str(linked / ".." / "analysis.json"),
    ])

    assert rc == 2
    assert not (outside / "analysis.json").exists()
    assert "symlink" in capsys.readouterr().err


def test_analyze_accepts_ordinary_nested_dotdot_paths(tmp_path, capsys):
    source_dir = tmp_path / "inputs"
    source_dir.mkdir()
    source = source_dir / "case.py"
    source.write_text(DIRECT_SOURCE, encoding="utf-8")
    output = tmp_path / "reports" / "nested" / ".." / "analysis.json"

    rc = cli.main(["analyze", str(source_dir / ".." / "inputs" / "case.py"),
                   "--out", str(output)])

    assert rc == 0
    assert output.resolve().is_file()
    assert "analysis.json" in capsys.readouterr().out
