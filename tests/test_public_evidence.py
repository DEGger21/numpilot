from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence"
CASES = {
    "competition-ordinary-20260828",
    "competition-poisson-stage-20260828",
}
FORBIDDEN_SUFFIXES = {".db", ".log", ".npy", ".npz", ".pyc", ".tmp"}


def _manifest_entries(path: Path) -> list[tuple[str, str]]:
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        entries.append((digest, relative))
    return entries


def test_public_evidence_has_only_the_two_current_competition_cases():
    assert {path.name for path in EVIDENCE.iterdir()} == {
        "README.md",
        "MANIFEST.sha256",
        *CASES,
    }


def test_public_evidence_manifest_is_complete_sorted_and_valid():
    manifest = EVIDENCE / "MANIFEST.sha256"
    entries = _manifest_entries(manifest)
    relatives = [relative for _, relative in entries]
    expected = sorted(
        str(path.relative_to(ROOT))
        for path in EVIDENCE.rglob("*")
        if path.is_file() and path != manifest
    )

    assert relatives == expected
    assert len(relatives) == len(set(relatives))
    for digest, relative in entries:
        target = ROOT / relative
        assert target.is_file()
        assert hashlib.sha256(target.read_bytes()).hexdigest() == digest


def test_each_competition_case_has_a_self_contained_valid_manifest():
    for case_name in CASES:
        case = EVIDENCE / case_name
        manifest = case / "MANIFEST.sha256"
        entries = _manifest_entries(manifest)
        expected = sorted(
            "./" + str(path.relative_to(case))
            for path in case.rglob("*")
            if path.is_file() and path != manifest
        )

        assert [relative for _, relative in entries] == expected
        for digest, relative in entries:
            assert hashlib.sha256((case / relative).read_bytes()).hexdigest() == digest


def test_public_evidence_contains_no_intermediate_binary_or_profiler_output():
    files = [path for path in EVIDENCE.rglob("*") if path.is_file()]
    assert not [path for path in files if path.suffix.lower() in FORBIDDEN_SUFFIXES]
    assert not [path for path in files if "profiler" in path.parts]


def test_public_evidence_index_marks_receipts_as_development_evidence():
    index = (EVIDENCE / "README.md").read_text(encoding="utf-8")
    assert "开发态证据" in index
    assert "最终提交前" in index
    assert "git_dirty=true" in index
