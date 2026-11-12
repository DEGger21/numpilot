"""CPU input observation feeds observed profiles into the gap taskbook."""

import dataclasses

import pytest

from numpilot.gap_requirements import (
    build_gap_taskbook,
    render_gap_taskbook_markdown,
    validate_gap_taskbook,
)
from numpilot.gaps.input_observation import (
    OBSERVATION_SCHEMA,
    observe_gap_inputs,
)
from numpilot.gaps.pipeline import prepare_gap_completion


_CONTRACT = '''\
import numpy as np


def build_inputs():
    rng = np.random.default_rng(7)
    return {"a": rng.standard_normal((6, 18), dtype=np.float32)}


def forward(a):
    return {"da": np.diff(a, n=1, axis=1)}
'''

_UNSUPPORTED = _CONTRACT.replace("np.float32", "np.float64")


def _prepare(tmp_path, text):
    path = tmp_path / "gap_contract.py"
    path.write_text(text, encoding="utf-8")
    preparation = prepare_gap_completion(
        str(path), target_soc="Ascend910B4", cann_version="9.0.0"
    )
    assert len(preparation.specs) == 1
    return path, preparation


def test_observation_records_concrete_operands_and_output(tmp_path):
    path, preparation = _prepare(tmp_path, _CONTRACT)
    observed = observe_gap_inputs(
        str(path), gap_apis=[preparation.specs[0].qualified_api], timeout=120.0
    )
    assert observed.status == "observed"
    assert observed.mode == "contract"
    assert observed.deterministic is True
    assert observed.executed is True
    assert len(observed.callsites) == 1
    call = observed.callsites[0]
    assert call["qualified_api"] == "numpy.diff"
    assert call["positional_args"] == 1
    assert call["keyword_args"] == ["axis", "n"]
    assert call["inputs"] == [{
        "name": "arg[0]", "dtype": "float32", "shape": [6, 18],
        "rank": 2, "c_contiguous": True,
    }]
    assert call["output"] == {
        "dtype": "float32", "shape": [6, 17], "rank": 2, "c_contiguous": True,
    }
    payload = observed.to_dict()
    assert payload["schema"] == OBSERVATION_SCHEMA
    assert payload["source"]["sha256"] == preparation.specs[0].source_sha256


def test_observation_rejects_a_foreign_dtype(tmp_path):
    path, preparation = _prepare(tmp_path, _UNSUPPORTED)
    observed = observe_gap_inputs(
        str(path), gap_apis=[preparation.specs[0].qualified_api], timeout=120.0
    )
    assert observed.status == "unsupported_dtype"
    assert observed.callsites[0]["inputs"][0]["dtype"] == "float64"


def test_taskbook_carries_observed_profiles_and_scope(tmp_path):
    path, preparation = _prepare(tmp_path, _CONTRACT)
    observed = observe_gap_inputs(
        str(path), gap_apis=[preparation.specs[0].qualified_api], timeout=120.0
    )
    taskbook = build_gap_taskbook(
        preparation.specs[0], source=preparation.source, observed=observed
    )
    value = validate_gap_taskbook(taskbook)
    contract = value["call_contract"]
    assert contract["observation_status"] == "observed"
    assert len(contract["observed_profiles"]) == 1
    operand = contract["observed_profiles"][0]["operands"][0]
    assert (operand["dtype"], operand["shape"], operand["rank"]) == (
        "float32", [6, 18], 2,
    )
    facts = contract["tensor_facts"]
    assert facts["input_dtype"] == {
        "status": "observed", "value": "float32",
        "evidence": "observed-inputs.json#callsites",
    }
    assert facts["input_shape"]["value"] == [6, 18]
    assert facts["output_shape"]["value"] == [6, 17]
    assert value["support_scope"]["dtype_source"] == "observed_cpu_execution"
    assert value["acceptance"]["oracle"]["status"] == "inputs_observed_cpu"
    assert value["status"] == "requirements_prepared_unverified"
    markdown = render_gap_taskbook_markdown(value)
    assert "已观测运行时 profiles" in markdown
    assert "| 行 10 `numpy.diff` | `arg[0]` | `float32` | `[6, 18]` | 2 | True |" in markdown
    assert "定向特化算子" in markdown
    assert "Planner 没有执行源程序" not in markdown


def test_taskbook_rejects_an_observation_from_another_source(tmp_path):
    path, preparation = _prepare(tmp_path, _CONTRACT)
    observed = observe_gap_inputs(
        str(path), gap_apis=[preparation.specs[0].qualified_api], timeout=120.0
    )
    forged = dataclasses.replace(observed, source_sha256="0" * 64)
    with pytest.raises(ValueError, match="do not match the GapSpec source"):
        build_gap_taskbook(
            preparation.specs[0], source=preparation.source, observed=forged
        )


def test_taskbook_records_a_blocked_foreign_dtype(tmp_path):
    path, preparation = _prepare(tmp_path, _UNSUPPORTED)
    observed = observe_gap_inputs(
        str(path), gap_apis=[preparation.specs[0].qualified_api], timeout=120.0
    )
    taskbook = build_gap_taskbook(
        preparation.specs[0], source=preparation.source, observed=observed
    )
    value = validate_gap_taskbook(taskbook)
    assert value["status"] == "requirements_blocked"
    assert value["generation"]["allowed"] is False
    assert any(
        "outside the float32 development scope" in blocker
        for blocker in value["generation"]["blockers"]
    )
    assert value["call_contract"]["observation_status"] == "unsupported_dtype"
