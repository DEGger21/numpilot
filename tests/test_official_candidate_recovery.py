import json
import subprocess
import sys

import pytest

from numpilot.gaps.official_candidate_recovery import (
    OfficialCandidateRecoveryError,
    materialize_official_candidate_recovery,
)


def _write_inputs(tmp_path):
    paths = {}
    for name in ("handoff", "preflight", "resolution", "gate"):
        path = tmp_path / (name + ".json")
        path.write_text("{}\n", encoding="utf-8")
        paths[name] = str(path)
    source = tmp_path / "source.py"
    source.write_text("import numpy as np\n", encoding="utf-8")
    paths["source"] = str(source)
    return paths


def test_official_recovery_rejects_unbound_inputs(tmp_path):
    paths = _write_inputs(tmp_path)
    with pytest.raises(OfficialCandidateRecoveryError) as error:
        materialize_official_candidate_recovery(
            paths["handoff"],
            paths["preflight"],
            paths["source"],
            paths["resolution"],
            paths["gate"],
            str(tmp_path / "state"),
            runtime_backend="ascend-npu",
            runtime_audit_id="audit",
            runtime_identity_sha256="0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        )
    assert error.value.code == "unsafe_gate"


def test_official_recovery_requires_receipt_only_profile(tmp_path):
    paths = _write_inputs(tmp_path)
    with pytest.raises(OfficialCandidateRecoveryError) as error:
        materialize_official_candidate_recovery(
            paths["handoff"],
            paths["preflight"],
            paths["source"],
            paths["resolution"],
            paths["gate"],
            str(tmp_path / "state"),
            runtime_backend="ascend-npu",
            runtime_audit_id="audit",
            runtime_identity_sha256="0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            profile="candidate-generation",
        )
    assert error.value.code == "unsafe_profile"


def test_official_recovery_cli_help_is_explicitly_fail_closed():
    result = subprocess.run(
        [sys.executable, "-m", "numpilot", "materialize-official-candidate-recovery", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "CANDIDATE_READY" in result.stdout
    assert "不调用模型" in result.stdout
