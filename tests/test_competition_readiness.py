import hashlib
import json
from datetime import datetime, timedelta, timezone
import io
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from numpilot.competition_readiness import (
    BLOCKED,
    PASS,
    assess_competition_readiness,
    render_readiness,
    write_readiness_report,
)


def _minimal_checkout(root: Path) -> None:
    (root / "numpilot").mkdir()
    (root / "tests").mkdir()
    (root / "docs").mkdir()
    (root / ".codeartsdoer" / "agents").mkdir(parents=True)
    (root / ".codeartsdoer" / "skills" / "numpilot-migration-gate").mkdir(
        parents=True
    )
    (root / "pyproject.toml").write_text("[project]\nname='numpilot'\n")
    (root / "README.md").write_text("README\n")
    (root / "README.zh-CN.md").write_text("README\n")
    (root / "docs" / "architecture.md").write_text("失败关闭 五态 official-first CodeArts 证据\n")
    (root / "docs" / "numpilot-architecture.drawio").write_text("diagram\n")
    (root / "docs" / "installation.md").write_text("install\n")
    (root / "docs" / "installation.zh-CN.md").write_text("安装\n")
    (root / "run_demo.sh").write_text("#!/bin/sh\n")
    (root / "LICENSE").write_text("Apache-2.0\n")
    (root / "THIRD_PARTY_NOTICES.md").write_text("notices\n")
    (root / "docs" / "numpilot-zh.pptx").write_bytes(b"ppt")
    (root / ".codeartsdoer" / "agents" / "numpilot-orchestrator.md").write_text("agent\n")
    (root / ".codeartsdoer" / "skills" / "numpilot-migration-gate" / "SKILL.md").write_text("skill\n")


def _verified_evidence(root: Path) -> None:
    evidence = root / "evidence"
    evidence.mkdir()
    (evidence / "codearts-usage.json").write_text(
        json.dumps(
            {
                "platform": "Huawei CodeArts",
                "actual_execution": True,
                "features": [
                    "code_generation",
                    "knowledge_qa",
                    "unit_test_generation",
                    "codebase_index",
                    "spec_driven_development",
                ],
            }
        )
    )
    (evidence / "cloud-deployment.json").write_text(
        json.dumps(
            {
                "provider": "Huawei Cloud",
                "status": "verified",
                "verification": True,
                "endpoint": "https://demo.example.invalid/app",
            }
        )
    )
    (evidence / "workload-manifest.json").write_text(
        json.dumps(
            {
                "data_source": "https://data.example.invalid/dataset-v1",
                "license": "CC-BY-4.0",
                "beneficiary": "某实验室 PDE 研究团队",
                "source_sha256": "a" * 64,
                "data_sha256": "b" * 64,
            }
        )
    )


def _verified_codearts_records(root: Path, *, as_list: bool = False) -> dict[str, dict[str, str]]:
    """Add full, hash-bound per-feature CodeArts records to a fixture."""

    evidence = root / "evidence"
    evidence.mkdir(exist_ok=True)
    records_dir = evidence / "codearts-records"
    records_dir.mkdir()
    records: dict[str, dict[str, str]] = {}
    for feature in (
        "code_generation",
        "knowledge_qa",
        "unit_test_generation",
        "codebase_index",
        "spec_driven_development",
    ):
        record_path = records_dir / (feature + ".json")
        payload = ("CodeArts record " + feature + "\n").encode("utf-8")
        record_path.write_bytes(payload)
        records[feature] = {
            "record_id": "task-" + feature,
            "record_path": "evidence/codearts-records/" + feature + ".json",
            "timestamp_utc": "2026-09-03T12:00:00Z",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "status": "passed",
        }
    evidence_payload = {
        "platform": "Huawei CodeArts",
        "actual_execution": True,
        "features": list(records),
        "records": list(
            {"feature": feature, **entry} for feature, entry in records.items()
        )
        if as_list
        else records,
    }
    (evidence / "codearts-usage.json").write_text(
        json.dumps(evidence_payload), encoding="utf-8"
    )
    return records


def test_current_style_checkout_reports_conservative_blockers(tmp_path):
    _minimal_checkout(tmp_path)
    report = assess_competition_readiness(
        tmp_path, now=datetime(2026, 9, 2, tzinfo=timezone.utc)
    )

    assert report.read_only is True
    assert report.source_execution_performed is False
    assert report.network_accessed is False
    assert report.overall_status == BLOCKED
    ids = {item.check_id: item for item in report.checks}
    assert ids["C05"].status == BLOCKED  # repository config is not CodeArts proof
    assert ids["C07"].status == BLOCKED  # no Huawei Cloud deployment receipt
    assert ids["A01"].status == BLOCKED  # no real-data manifest
    assert report.deadline["days_remaining"] == 3
    actions = "\n".join(report.next_actions)
    assert "报名表" in actions
    assert "data_source" in actions
    assert "受益方" in actions


def test_deadline_uses_utc_date_for_non_utc_aware_now(tmp_path):
    _minimal_checkout(tmp_path)
    # 00:30 on Sep 5 at UTC+14 is still Sep 4 in UTC; the UTC-bound report
    # therefore has one day remaining rather than treating the deadline as
    # already reached.
    report = assess_competition_readiness(
        tmp_path,
        now=datetime(
            2026,
            9,
            5,
            0,
            30,
            tzinfo=timezone(timedelta(hours=14)),
        ),
    )

    assert report.deadline["days_remaining"] == 1
    assert report.generated_at_utc.startswith("2026-09-04T10:30:00")


@pytest.mark.parametrize("bad_now", (0, False, ""))
def test_readiness_rejects_falsy_non_datetime_now(tmp_path, bad_now):
    _minimal_checkout(tmp_path)

    with pytest.raises(ValueError, match="检查时间"):
        assess_competition_readiness(tmp_path, now=bad_now)


def test_verified_local_evidence_is_recognized_without_running_code(tmp_path):
    _minimal_checkout(tmp_path)
    _verified_evidence(tmp_path)
    # A malicious source must never be imported/executed by the auditor.
    marker = tmp_path / "executed"
    (tmp_path / "examples").mkdir()
    (tmp_path / "examples" / "evil.py").write_text(
        "from pathlib import Path\nPath('executed').write_text('bad')\n"
    )
    report = assess_competition_readiness(
        tmp_path, now=datetime(2026, 8, 1, tzinfo=timezone.utc)
    )
    ids = {item.check_id: item for item in report.checks}

    assert ids["C03"].status == PASS  # verified Huawei Cloud environment is an allowed demo
    assert ids["C05"].status == PASS
    assert ids["C06"].status == PASS
    assert ids["C07"].status == PASS
    assert ids["A01"].status == PASS
    assert ids["A02"].status == PASS
    assert not marker.exists()


def test_codearts_records_are_hash_bound_and_traceable(tmp_path):
    _minimal_checkout(tmp_path)
    _verified_codearts_records(tmp_path)

    report = assess_competition_readiness(tmp_path)
    checks = {item.check_id: item for item in report.checks}
    codearts = checks["C05"].details["codearts"]

    assert checks["C05"].status == PASS
    assert checks["C06"].status == PASS
    assert codearts["records_present"] is True
    assert codearts["records_complete"] is True
    assert codearts["records_status"] == "verified"
    assert codearts["records_verified_features"] == [
        "code_generation",
        "knowledge_qa",
        "unit_test_generation",
        "codebase_index",
        "spec_driven_development",
    ]
    assert codearts["records"]["entries"]["code_generation"]["record_path"].startswith(
        "evidence/"
    )


def test_codearts_records_accept_exported_list_shape(tmp_path):
    _minimal_checkout(tmp_path)
    _verified_codearts_records(tmp_path, as_list=True)

    report = assess_competition_readiness(tmp_path)
    check = next(item for item in report.checks if item.check_id == "C06")

    assert check.status == PASS
    assert check.details["records"]["status"] == "verified"


def test_codearts_record_hash_drift_is_partial_and_blocks_strict_readiness(tmp_path):
    _minimal_checkout(tmp_path)
    _verified_codearts_records(tmp_path)
    drifted = tmp_path / "evidence" / "codearts-records" / "knowledge_qa.json"
    drifted.write_text("changed after export\n", encoding="utf-8")

    report = assess_competition_readiness(tmp_path)
    checks = {item.check_id: item for item in report.checks}
    records = checks["C06"].details["records"]

    assert checks["C05"].status == "partial"
    assert checks["C06"].status == "partial"
    assert report.overall_status == BLOCKED
    assert records["hash_mismatched_features"] == ["knowledge_qa"]
    assert records["complete"] is False


def test_codearts_record_path_traversal_and_invalid_timestamp_are_not_read(tmp_path):
    _minimal_checkout(tmp_path)
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    outside = tmp_path.parent / "outside-codearts-record.json"
    outside.write_text(
        json.dumps({"token": "do-not-leak"}), encoding="utf-8"
    )
    payload = {
        "platform": "Huawei CodeArts",
        "actual_execution": True,
        "features": ["code_generation"],
        "records": {
            "code_generation": {
                "record_id": "task-1",
                "record_path": "../outside-codearts-record.json",
                "timestamp_utc": "2026-09-03T12:00:00",  # naive, invalid
                "sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
                "status": "passed",
            }
        },
    }
    (evidence / "codearts-usage.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    report = assess_competition_readiness(tmp_path)
    checks = {item.check_id: item for item in report.checks}
    records = checks["C06"].details["records"]
    rendered = json.dumps(report.to_dict(), ensure_ascii=False)

    assert checks["C05"].status == BLOCKED
    assert checks["C06"].status == "partial"
    assert records["status"] == "blocked"
    assert records["invalid_features"] == ["code_generation"]
    assert "do-not-leak" not in rendered
    assert "outside-codearts-record" not in rendered


def test_cloud_generic_passed_status_is_not_deployment_proof(tmp_path):
    """A generic stage status must not satisfy the cloud evidence contract."""

    _minimal_checkout(tmp_path)
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "cloud-deployment.json").write_text(
        json.dumps(
            {
                "provider": "Huawei Cloud",
                "status": "passed",
                "verification": True,
                "endpoint": "https://demo.example.invalid/app",
            }
        ),
        encoding="utf-8",
    )

    report = assess_competition_readiness(tmp_path)
    checks = {item.check_id: item for item in report.checks}

    assert checks["C07"].status == BLOCKED
    assert checks["C07"].details["cloud"]["verified"] is False


def test_hosted_video_page_link_is_detected_but_not_marked_verified(tmp_path):
    _minimal_checkout(tmp_path)
    (tmp_path / "README.zh-CN.md").write_text(
        "比赛演示视频：https://www.bilibili.com/video/BV1demo123\n",
        encoding="utf-8",
    )

    report = assess_competition_readiness(tmp_path)
    check = next(item for item in report.checks if item.check_id == "C03")

    assert check.status == "partial"
    video = check.details["video"]
    assert video["present"] is True
    assert video["verified"] is False
    assert any(host.endswith("bilibili.com") for host in video["links"])
    assert video["unverified_link_count"] >= 1


def test_spreadsheet_registration_form_is_detected_without_reading_contents(tmp_path):
    _minimal_checkout(tmp_path)
    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "报名表-已盖章.xlsx").write_bytes(b"opaque-office-form")

    report = assess_competition_readiness(tmp_path)
    check = next(item for item in report.checks if item.check_id == "C12")

    assert check.status == PASS
    assert check.details["registration"]["files"] == [
        "submission/报名表-已盖章.xlsx"
    ]


def test_labeled_markdown_video_link_is_detected_on_an_unknown_host(tmp_path):
    _minimal_checkout(tmp_path)
    (tmp_path / "README.md").write_text(
        "[Demo video](https://demo.example.invalid/review/run-42)\n",
        encoding="utf-8",
    )

    report = assess_competition_readiness(tmp_path)
    check = next(item for item in report.checks if item.check_id == "C03")

    assert check.status == "partial"
    assert "demo.example.invalid" in check.details["video"]["links"]


def test_manifest_is_checked_without_shelling_out(tmp_path):
    _minimal_checkout(tmp_path)
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    tracked = evidence / "note.txt"
    tracked.write_text("hello\n")
    import hashlib

    digest = hashlib.sha256(tracked.read_bytes()).hexdigest()
    (evidence / "MANIFEST.sha256").write_text(f"{digest}  evidence/note.txt\n")
    report = assess_competition_readiness(tmp_path)
    check = next(item for item in report.checks if item.check_id == "C09")
    assert check.status == PASS
    assert check.details["manifest"]["complete"] is True


def test_manifest_rejects_unicode_format_names_without_echoing_them(tmp_path):
    _minimal_checkout(tmp_path)
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    # The name is deliberately not created; it exercises manifest parsing and
    # report redaction without allowing a control/format mark into status text.
    bad_name = "evidence/noted\u202e.txt"
    (evidence / "MANIFEST.sha256").write_text(
        "%s  %s\n" % ("a" * 64, bad_name), encoding="utf-8"
    )

    report = assess_competition_readiness(tmp_path)
    check = next(item for item in report.checks if item.check_id == "C09")
    manifest = check.details["manifest"]

    assert manifest["complete"] is False
    assert manifest["invalid"] >= 1
    rendered = json.dumps(report.to_dict(), ensure_ascii=False)
    assert "\u202e" not in rendered


def test_readiness_bounded_reader_does_not_use_unbounded_path_read(tmp_path, monkeypatch):
    payload = b"small evidence\n"
    path = tmp_path / "evidence.txt"
    path.write_bytes(payload)
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda _path: pytest.fail("readiness used an unbounded file read"),
    )

    from numpilot import competition_readiness as readiness

    assert readiness._bounded_bytes(path, len(payload)) == payload


def test_readiness_bounded_reader_rejects_growth_beyond_limit(tmp_path, monkeypatch):
    path = tmp_path / "evidence.txt"
    path.write_bytes(b"x")
    from numpilot import competition_readiness as readiness

    real_read = readiness.os.read

    def oversized_read(fd, size):
        return real_read(fd, size) + b"x" * (len(b"x") + 1)

    monkeypatch.setattr(readiness.os, "read", oversized_read)
    assert readiness._bounded_bytes(path, 1) is None


def test_partial_critical_requirement_remains_a_strict_blocker(tmp_path):
    _minimal_checkout(tmp_path)
    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "numpilot-source.zip").write_bytes(b"development-package")

    report = assess_competition_readiness(tmp_path)
    check = next(item for item in report.checks if item.check_id == "C08")

    assert check.status == "partial"  # package/notices exist, SBOM is missing
    assert any(item.startswith("C08 ") for item in report.blockers)


def test_report_render_and_write_are_redacted_and_write_once(tmp_path):
    _minimal_checkout(tmp_path)
    config = tmp_path / "llm.json"
    config.write_text(
        json.dumps(
            {
                "base_url": "https://private.invalid/v1",
                "auth_token": "super-secret-token",
                "model": "glm-5.3",
            }
        )
    )
    report = assess_competition_readiness(tmp_path, llm_config=config)
    rendered = render_readiness(report)
    assert "super-secret-token" not in rendered
    output = tmp_path / "out" / "readiness.json"
    write_readiness_report(report, output)
    payload = output.read_text()
    assert "super-secret-token" not in payload
    assert json.loads(payload)["read_only"] is True
    with pytest.raises(FileExistsError):
        write_readiness_report(report, output)
    # Publication uses a no-clobber hard-link and removes its staging name.
    assert not list(output.parent.glob(".readiness.json.*.tmp"))


@pytest.mark.parametrize(
    "bad_path",
    (
        "bad\npath",
        "bad\rpath",
        "bad\x00path",
        "bad\x1bpath",
        "bad\u202epath",  # RIGHT-TO-LEFT OVERRIDE (Unicode Cf)
        "x" * 5000,
    ),
)
def test_readiness_api_redacts_invalid_llm_config_path(tmp_path, bad_path):
    _minimal_checkout(tmp_path)
    report = assess_competition_readiness(tmp_path, llm_config=bad_path)
    assert report.llm_config["path_valid"] is False
    assert "bad" not in report.llm_config["path"]
    assert "\n" not in report.to_dict()["llm_config"]["path"]


def test_readiness_preserves_pathlike_config_boundary(tmp_path):
    """A PathLike must be validated by fspath, not stringified by the caller."""

    _minimal_checkout(tmp_path)

    class BidiPath:
        def __fspath__(self):
            return "bad\u202epath"

        def __str__(self):  # pragma: no cover - proves assess does not use it
            raise AssertionError("config path was stringified")

    report = assess_competition_readiness(tmp_path, llm_config=BidiPath())
    assert report.llm_config["path_valid"] is False
    assert report.llm_config["path"] == "<invalid-llm-config-path>"


def test_readiness_uses_same_default_config_path_as_llm_loader(tmp_path, monkeypatch):
    _minimal_checkout(tmp_path)
    monkeypatch.delenv("NUMPILOT_LLM_CONFIG_FILE", raising=False)
    monkeypatch.delenv("NUMPILOT_CONFIG_HOME", raising=False)
    monkeypatch.delenv("NUMPILOT_HOME", raising=False)
    monkeypatch.delenv("NUMPILOT_RUNTIME_ROOT", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg-config"))

    report = assess_competition_readiness(tmp_path)

    assert report.llm_config["path"] == str(
        tmp_path / "xdg-config" / "numpilot" / "llm.json"
    )


def test_report_write_rejects_unicode_format_path(tmp_path):
    _minimal_checkout(tmp_path)
    report = assess_competition_readiness(tmp_path)

    with pytest.raises(ValueError, match="报告路径无效"):
        write_readiness_report(report, tmp_path / "readiness\u202e.json")


def test_report_write_rejects_symlinked_parent(tmp_path):
    _minimal_checkout(tmp_path)
    report = assess_competition_readiness(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    linked_parent = tmp_path / "linked-out"
    linked_parent.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="符号链接"):
        write_readiness_report(report, linked_parent / "readiness.json")
    assert not (outside / "readiness.json").exists()


def test_report_write_rejects_dotdot_symlinked_parent(tmp_path):
    _minimal_checkout(tmp_path)
    report = assess_competition_readiness(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    linked_parent = tmp_path / "linked-out"
    linked_parent.symlink_to(outside, target_is_directory=True)

    # ``abspath`` would collapse this to ``tmp_path/readiness.json`` even
    # though filesystem resolution follows ``linked-out`` before ``..``.
    with pytest.raises(ValueError, match="符号链接"):
        write_readiness_report(report, linked_parent / ".." / "readiness.json")
    assert not (tmp_path / "readiness.json").exists()
    assert not (outside / "readiness.json").exists()


def test_report_write_rejects_shared_parent_permissions(tmp_path):
    _minimal_checkout(tmp_path)
    report = assess_competition_readiness(tmp_path)
    shared = tmp_path / "shared"
    shared.mkdir(mode=0o755)
    # ``mkdir(mode=...)`` is subject to the process umask; make the assertion
    # deterministic for the permission gate under test.
    shared.chmod(0o755)

    with pytest.raises(ValueError, match="父目录.*私有权限"):
        write_readiness_report(report, shared / "readiness.json")
    assert not (shared / "readiness.json").exists()


def test_codearts_negative_feature_label_is_not_counted(tmp_path):
    _minimal_checkout(tmp_path)
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "codearts-usage.json").write_text(
        json.dumps(
            {
                "platform": "Huawei CodeArts",
                "actual_execution": True,
                "features": [
                    "not code_generation",
                    "knowledge_qa",
                    "unit_test_generation",
                    "codebase_index",
                    "spec_driven_development",
                ],
            }
        ),
        encoding="utf-8",
    )

    report = assess_competition_readiness(tmp_path)
    check = next(item for item in report.checks if item.check_id == "C06")

    assert check.status == "partial"
    assert any("code_generation" in gap for gap in check.gaps)


def test_readiness_rejects_symlinked_checkout_root(tmp_path):
    real_root = tmp_path / "checkout"
    real_root.mkdir()
    _minimal_checkout(real_root)
    linked_root = tmp_path / "checkout-link"
    linked_root.symlink_to(real_root, target_is_directory=True)

    with pytest.raises(ValueError, match="仓库路径.*符号链接"):
        assess_competition_readiness(linked_root)


def test_readiness_rejects_dotdot_checkout_path_that_crosses_symlink(tmp_path):
    real_root = tmp_path / "checkout"
    real_root.mkdir()
    _minimal_checkout(real_root)
    linked_root = tmp_path / "checkout-link"
    linked_root.symlink_to(real_root, target_is_directory=True)

    with pytest.raises(ValueError, match="仓库路径.*符号链接"):
        assess_competition_readiness(linked_root / ".." / "checkout")


def test_readiness_does_not_borrow_enclosing_git_checkout(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _minimal_checkout(repo)
    subprocess.run(
        ["git", "init", "--quiet"],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    nested = repo / "nested"
    nested.mkdir()

    report = assess_competition_readiness(nested)

    assert report.git["git_root"] == str(repo)
    assert report.git["root_matches"] is False
    assert report.git["is_repository"] is False
    c10 = next(item for item in report.checks if item.check_id == "C10")
    assert c10.status == BLOCKED


def test_git_status_output_overflow_is_not_reported_as_clean(tmp_path, monkeypatch):
    from numpilot import competition_readiness as readiness

    def fake_run(command, **kwargs):
        del kwargs
        if "status" in command:
            return SimpleNamespace(
                returncode=0,
                stdout="x" * (readiness._MAX_GIT_OUTPUT_BYTES + 1),
            )
        if "--show-toplevel" in command:
            return SimpleNamespace(returncode=0, stdout=str(tmp_path))
        if command[-1] == "HEAD":
            return SimpleNamespace(returncode=0, stdout="a" * 40)
        if "symbolic-ref" in command:
            return SimpleNamespace(returncode=0, stdout="main")
        # ``describe --exact-match`` is optional and normally exits non-zero.
        return SimpleNamespace(returncode=1, stdout="")

    monkeypatch.setattr(readiness.subprocess, "run", fake_run)
    snapshot = readiness._git_snapshot(tmp_path)

    assert snapshot["is_repository"] is True
    assert snapshot["status_entries"] is None
    assert snapshot["dirty"] is None


def test_git_status_failure_is_a_strict_cleanliness_blocker(tmp_path, monkeypatch):
    from numpilot import competition_readiness as readiness

    def fail_status(command, **kwargs):
        del kwargs
        if "--show-toplevel" in command:
            return SimpleNamespace(returncode=0, stdout=str(tmp_path))
        if command[-1] == "HEAD":
            return SimpleNamespace(returncode=0, stdout="a" * 40)
        if "symbolic-ref" in command:
            return SimpleNamespace(returncode=0, stdout="main")
        # Both ``describe`` and ``status`` are allowed to fail, but status
        # failure must remain distinguishable from a clean empty result.
        return SimpleNamespace(returncode=1, stdout="")

    monkeypatch.setattr(readiness.subprocess, "run", fail_status)
    snapshot = readiness._git_snapshot(tmp_path)
    assert snapshot["is_repository"] is True
    assert snapshot["dirty"] is None


def test_report_write_does_not_clobber_racing_target(tmp_path, monkeypatch):
    _minimal_checkout(tmp_path)
    report = assess_competition_readiness(tmp_path)
    output = tmp_path / "out" / "readiness.json"

    def create_target_then_fail(source, destination, *args, **kwargs):
        Path(destination).write_text("sentinel\n", encoding="utf-8")
        raise FileExistsError(destination)

    monkeypatch.setattr("os.link", create_target_then_fail)
    with pytest.raises(FileExistsError):
        write_readiness_report(report, output)
    assert output.read_text(encoding="utf-8") == "sentinel\n"


def test_cli_json_is_one_document_and_strict_is_actionable(tmp_path, monkeypatch):
    _minimal_checkout(tmp_path)
    output = io.StringIO()
    diagnostic = io.StringIO()
    monkeypatch.setattr(sys, "stdout", output)
    monkeypatch.setattr(sys, "stderr", diagnostic)
    from numpilot import __main__ as cli

    assert cli.main(["competition-readiness", "--repo-root", str(tmp_path), "--json"]) == 0
    payload = json.loads(output.getvalue())
    assert payload["producer"] == "numpilot-competition-readiness-v1"
    assert payload["network_accessed"] is False
    assert diagnostic.getvalue() == ""

    output.seek(0)
    output.truncate(0)
    assert cli.main(
        ["readiness", "--repo-root", str(tmp_path), "--strict", "--json"]
    ) == 1
    assert json.loads(output.getvalue())["overall_status"] == BLOCKED


def test_cli_readiness_rejects_explicit_empty_output_path(tmp_path, capsys):
    _minimal_checkout(tmp_path)
    from numpilot import __main__ as cli

    rc = cli.main(
        [
            "competition-readiness",
            "--repo-root",
            str(tmp_path),
            "--out",
            "",
        ]
    )

    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    assert "[比赛就绪度] 失败" in captured.err


def test_cli_readiness_displays_normalized_pathlike_once(tmp_path, monkeypatch):
    _minimal_checkout(tmp_path)
    output = io.StringIO()
    monkeypatch.setattr(sys, "stdout", output)

    target = tmp_path / "out" / "readiness.json"

    class OneShotPath:
        calls = 0

        def __fspath__(self):
            self.calls += 1
            if self.calls > 1:
                raise RuntimeError("path converted more than once")
            return str(target)

    path = OneShotPath()
    args = SimpleNamespace(
        repo_root=tmp_path,
        llm_config=None,
        out=path,
        json=False,
        strict=False,
    )

    from numpilot import __main__ as cli

    assert cli._cmd_competition_readiness(args) == 0
    assert path.calls == 1
    assert target.is_file()
    assert str(target) in output.getvalue()
