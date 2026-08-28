from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = (
    REPOSITORY_ROOT
    / "docs"
    / "evidence"
    / "day2-demo-golden-profile-20260826.json"
)
DEMO_ROOT = Path("/home/ma-user/work/demo")

DECISION_VOCABULARY = [
    "DIRECT",
    "FALLBACK_CONSENT",
    "GAP_REQUIRED",
    "CAPABILITY_AUDIT_REQUIRED",
    "BLOCKED",
]
ROUTE_VOCABULARY = {
    "NATIVE_NPU",
    "COMPOSITE_NPU",
    "CPU_FALLBACK",
    "CONDITIONAL_CPU_FALLBACK",
    "UNAVAILABLE",
}
QUALIFICATION_VOCABULARY = {
    "EXACT_PROFILE",
    "AUDITED_FALLBACK",
    "UNSUPPORTED_CALL_PROFILE",
    "UNAVAILABLE",
}
EXPECTED_DECISIONS = {
    "01": "FALLBACK_CONSENT",
    "02": "DIRECT",
    "03": "DIRECT",
    "04": "DIRECT",
    "05": "BLOCKED",
    "06": "DIRECT",
    "07": "BLOCKED",
    "08": "FALLBACK_CONSENT",
    "09": "BLOCKED",
    "10": "DIRECT",
}
EXPECTED_REFERENCES = {
    "01": "01_asnumpy_pure.py",
    "02": "02_asnumpy_one_numpy.py",
    "03": "03_subtract_clip.py",
    "04": "04_eye_tanh_det.py",
    "05": "05_cumsum_prod.py",
    "06": "06_log_exp_power.py",
    "07": "07_one_numpy_normal.py",
    "08": "08_one_numpy_ones.py",
    "09": "09_one_numpy_arange.py",
    "10": "10_one_numpy_zeros.py",
}
EXPECTED_CALLS = {
    "01": [
        (25, 4, "numpy.full"),
        (26, 4, "numpy.full"),
        (27, 9, "numpy.matmul"),
        (30, 13, "numpy.exp"),
        (31, 21, "numpy.sum"),
        (34, 11, "numpy.sum"),
    ],
    "02": [
        (25, 4, "numpy.linspace"),
        (28, 9, "numpy.add"),
        (28, 16, "numpy.sin"),
        (28, 27, "numpy.cos"),
        (31, 13, "numpy.absolute"),
        (32, 10, "numpy.amax"),
    ],
    "03": [
        (25, 4, "numpy.full"),
        (26, 4, "numpy.full"),
        (29, 7, "numpy.subtract"),
        (32, 10, "numpy.clip"),
        (35, 9, "numpy.sqrt"),
    ],
    "04": [
        (25, 4, "numpy.eye"),
        (28, 4, "numpy.multiply"),
        (28, 19, "numpy.full"),
        (31, 9, "numpy.tanh"),
        (34, 10, "numpy.linalg.det"),
    ],
    "05": [
        (25, 4, "numpy.full"),
        (28, 5, "numpy.cumsum"),
        (31, 9, "numpy.cumprod"),
        (34, 8, "numpy.maximum"),
    ],
    "06": [
        (25, 7, "numpy.full"),
        (26, 10, "numpy.full"),
        (29, 8, "numpy.power"),
        (32, 9, "numpy.log"),
        (35, 9, "numpy.exp"),
        (38, 12, "numpy.equal"),
    ],
    "07": [
        (25, 7, "numpy.random.normal"),
        (28, 5, "numpy.expm1"),
        (31, 11, "numpy.log1p"),
        (34, 8, "numpy.sign"),
        (37, 6, "numpy.subtract"),
        (38, 9, "numpy.square"),
        (39, 15, "numpy.sum"),
    ],
    "08": [
        (25, 4, "numpy.ones"),
        (28, 4, "numpy.full"),
        (31, 4, "numpy.dot"),
        (34, 8, "numpy.floor"),
        (35, 8, "numpy.ceil"),
        (38, 6, "numpy.full"),
        (39, 8, "numpy.floor"),
        (40, 8, "numpy.ceil"),
    ],
    "09": [
        (25, 4, "numpy.arange"),
        (28, 11, "numpy.maximum"),
        (33, 34, "numpy.sqrt"),
        (36, 11, "numpy.maximum"),
        (39, 10, "numpy.minimum"),
    ],
    "10": [
        (25, 7, "numpy.zeros"),
        (28, 4, "numpy.full"),
        (29, 4, "numpy.full"),
        (32, 13, "numpy.mod"),
        (35, 13, "numpy.remainder"),
        (38, 11, "numpy.equal"),
        (41, 9, "numpy.any"),
    ],
}

TOP_LEVEL_KEYS = {
    "schema",
    "schema_version",
    "profile_date",
    "analysis_mode",
    "demo_root_hint",
    "source_policy",
    "decision_vocabulary",
    "route_vocabulary",
    "qualification_vocabulary",
    "actual_remote_call",
    "execution_performed",
    "performance_valid",
    "speedup",
    "cases",
}
CASE_KEYS = {
    "case_id",
    "source",
    "reference",
    "numpy_calls",
    "qualified_attributes",
    "unmodeled_surfaces",
    "exact_qualification",
    "decision",
    "decision_reason",
    "next_action",
    "actual_remote_call",
    "execution_performed",
    "performance_valid",
    "speedup",
}
SOURCE_KEYS = {
    "relative_path",
    "role",
    "sha256",
    "imports",
    "einsum_equations",
}
REFERENCE_KEYS = {"relative_path", "role", "sha256"}
IMPORT_KEYS = {"kind", "module", "name", "alias", "line", "col"}
CALL_KEYS = {
    "line",
    "col",
    "qualified_name",
    "static_facts",
    "route",
    "qualification",
    "reason",
}
ATTRIBUTE_KEYS = {"line", "col", "qualified_name", "use", "reason"}
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
SOURCE_PATTERN = re.compile(r"([0-9]{2})_numpy\.py")
FORBIDDEN_KEY_FRAGMENTS = {
    "response",
    "credential",
    "api_key",
    "access_token",
    "authorization",
    "secret",
}


def _load_evidence() -> dict[str, Any]:
    return json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))


def _all_keys(value: Any):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _all_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _all_keys(child)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_day2_demo_golden_profile_has_strict_schema_and_vocabulary():
    evidence = _load_evidence()

    assert set(evidence) == TOP_LEVEL_KEYS
    assert evidence["schema"] == "numpilot.day2_demo_golden_profile"
    assert evidence["schema_version"] == 1
    assert evidence["profile_date"] == "2026-08-26"
    assert evidence["analysis_mode"] == "STATIC_AST_ONLY"
    assert evidence["source_policy"] == {
        "migration_source_pattern": "NN_numpy.py",
        "migration_source_count": 10,
        "reference_role": "REFERENCE_ONLY_NOT_MIGRATION_SOURCE",
        "reference_files_are_migration_inputs": False,
    }
    assert evidence["decision_vocabulary"] == DECISION_VOCABULARY
    assert set(evidence["route_vocabulary"]) == ROUTE_VOCABULARY
    assert set(evidence["qualification_vocabulary"]) == QUALIFICATION_VOCABULARY
    assert evidence["actual_remote_call"] is False
    assert evidence["execution_performed"] is False
    assert evidence["performance_valid"] is False
    assert evidence["speedup"] is None


def test_day2_demo_golden_profile_records_exactly_ten_migration_sources():
    evidence = _load_evidence()
    cases = evidence["cases"]

    assert len(cases) == 10
    assert [case["case_id"] for case in cases] == list(EXPECTED_DECISIONS)
    assert {case["decision"] for case in cases} == {
        "DIRECT",
        "FALLBACK_CONSENT",
        "BLOCKED",
    }

    for case in cases:
        case_id = case["case_id"]
        assert set(case) == CASE_KEYS
        assert case["decision"] == EXPECTED_DECISIONS[case_id]
        assert case["decision"] in DECISION_VOCABULARY
        assert case["decision_reason"].strip()
        assert case["next_action"].strip()
        assert case["actual_remote_call"] is False
        assert case["execution_performed"] is False
        assert case["performance_valid"] is False
        assert case["speedup"] is None

        source = case["source"]
        assert set(source) == SOURCE_KEYS
        assert source["role"] == "MIGRATION_SOURCE"
        match = SOURCE_PATTERN.fullmatch(source["relative_path"])
        assert match is not None
        assert match.group(1) == case_id
        assert SHA256_PATTERN.fullmatch(source["sha256"])
        assert source["einsum_equations"] == []
        assert source["imports"]
        for imported in source["imports"]:
            assert set(imported) == IMPORT_KEYS
            assert imported["kind"] in {"import", "from"}
            assert imported["module"]
            assert imported["line"] >= 1
            assert imported["col"] >= 0

        reference = case["reference"]
        assert set(reference) == REFERENCE_KEYS
        assert reference["relative_path"] == EXPECTED_REFERENCES[case_id]
        assert reference["role"] == "REFERENCE_ONLY_NOT_MIGRATION_SOURCE"
        assert SHA256_PATTERN.fullmatch(reference["sha256"])

        assert isinstance(case["qualified_attributes"], list)
        for attribute in case["qualified_attributes"]:
            assert set(attribute) == ATTRIBUTE_KEYS
            assert attribute["qualified_name"].startswith("numpy.")
            assert attribute["line"] >= 1
            assert attribute["col"] >= 0
            assert attribute["use"].strip()
            assert attribute["reason"].strip()
        assert all(
            isinstance(surface, str) and surface.strip()
            for surface in case["unmodeled_surfaces"]
        )
        assert all(
            isinstance(predicate, str) and predicate.strip()
            for predicate in case["exact_qualification"]
        )


def test_day2_demo_golden_profile_has_every_numpy_callsite_and_profile_fact():
    evidence = _load_evidence()
    cases = {case["case_id"]: case for case in evidence["cases"]}

    assert sum(len(case["numpy_calls"]) for case in cases.values()) == 59
    for case_id, expected_calls in EXPECTED_CALLS.items():
        calls = cases[case_id]["numpy_calls"]
        actual_calls = [
            (call["line"], call["col"], call["qualified_name"])
            for call in calls
        ]
        assert actual_calls == expected_calls
        assert len(actual_calls) == len(set(actual_calls))
        for call in calls:
            assert set(call) == CALL_KEYS
            assert call["qualified_name"].startswith("numpy.")
            assert call["line"] >= 1
            assert call["col"] >= 0
            assert isinstance(call["static_facts"], dict)
            assert call["static_facts"]
            assert call["route"] in ROUTE_VOCABULARY
            assert call["qualification"] in QUALIFICATION_VOCABULARY
            assert call["reason"].strip()

    for case_id in {"02", "03", "06", "10"}:
        assert cases[case_id]["decision"] == "DIRECT"
        assert {call["route"] for call in cases[case_id]["numpy_calls"]} == {
            "NATIVE_NPU"
        }
    assert cases["04"]["decision"] == "DIRECT"
    assert {call["route"] for call in cases["04"]["numpy_calls"]} == {
        "NATIVE_NPU",
        "COMPOSITE_NPU",
    }
    assert any(
        "numpy.pi" in predicate
        for predicate in cases["02"]["exact_qualification"]
    )
    assert any(
        "numpy.amax" in predicate and "axis=None" in predicate
        for predicate in cases["02"]["exact_qualification"]
    )
    assert any(
        "linalg.det" in predicate and "rank=2" in predicate
        for predicate in cases["04"]["exact_qualification"]
    )
    assert (
        "must not be reported as an operator gap"
        in cases["04"]["decision_reason"]
    )

    for case_id in {"01", "08"}:
        assert cases[case_id]["decision"] == "FALLBACK_CONSENT"
        assert any(
            call["route"] in {"CPU_FALLBACK", "CONDITIONAL_CPU_FALLBACK"}
            for call in cases[case_id]["numpy_calls"]
        )
    for case_id in {"05", "07", "09"}:
        assert cases[case_id]["decision"] == "BLOCKED"
        assert any(
            call["qualification"]
            in {"UNSUPPORTED_CALL_PROFILE", "UNAVAILABLE"}
            for call in cases[case_id]["numpy_calls"]
        )


def test_day2_demo_golden_profile_contains_no_response_or_credential_fields():
    evidence = _load_evidence()

    for key in _all_keys(evidence):
        normalized = key.lower()
        assert all(fragment not in normalized for fragment in FORBIDDEN_KEY_FRAGMENTS)


def test_day2_demo_golden_profile_hashes_match_demo_files_when_present():
    if not DEMO_ROOT.is_dir():
        return

    evidence = _load_evidence()
    for case in evidence["cases"]:
        source = case["source"]
        source_path = DEMO_ROOT / source["relative_path"]
        assert source_path.is_file()
        assert _sha256(source_path) == source["sha256"]

        reference = case["reference"]
        reference_path = DEMO_ROOT / reference["relative_path"]
        assert reference_path.is_file()
        assert _sha256(reference_path) == reference["sha256"]
