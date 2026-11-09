import json
import subprocess
import sys

import pytest

from numpilot.gaps.official_adapter_context_gate import (
    OfficialAdapterContextGateError,
    validate_official_adapter_context,
)

def _bound_payload():
    return {
        "schema": "numpilot.official-adapter-context-bound.v1",
        "source": "/tmp/source.py",
        "reference_root": "/tmp/asnumpy",
        "operator_manifest": {
            "model_invocation_allowed": True,
            "required_next_gate": "invoke_configured_model_for_official_adapter",
            "qualified_api": "numpy.concatenate",
        },
        "adapter_contract": {},
        "context_plan": {},
        "side_effects": {
            "model_invocation_performed": False,
            "network_access_performed": False,
            "system_write_performed": False,
            "asnumpy_import_performed": False,
            "build_performed": False,
            "execution_performed": False,
            "activation_performed": False,
        },
    }

def test_context_gate_rejects_unsafe_side_effect_claim(tmp_path):
    payload = _bound_payload()
    payload["side_effects"]["build_performed"] = True
    bound = tmp_path / "bound.json"
    bound.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(OfficialAdapterContextGateError):
        validate_official_adapter_context(
            str(bound), str(tmp_path / "handoff"), str(tmp_path / "preflight"),
            str(tmp_path / "source.py"), str(tmp_path / "resolution"),
        )

def test_context_gate_rejects_closed_model_gate(tmp_path):
    payload = _bound_payload()
    payload["operator_manifest"]["model_invocation_allowed"] = False
    bound = tmp_path / "bound.json"
    bound.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(OfficialAdapterContextGateError):
        validate_official_adapter_context(
            str(bound), str(tmp_path / "handoff"), str(tmp_path / "preflight"),
            str(tmp_path / "source.py"), str(tmp_path / "resolution"),
        )


def test_official_adapter_generate_cli_is_preflight_only():
    result = subprocess.run(
        [sys.executable, "-m", "numpilot", "official-adapter-generate", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "contract preflight" in result.stdout
    assert "不调用模型" in result.stdout
