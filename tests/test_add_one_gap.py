# This file is part of the NumPilot project at Harbin Institute of Technology (HIT).
#
# Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
# SPDX-License-Identifier: Apache-2.0
#
# Authors (accounts):
# - Su Tonghua <@sutonghua>
# - Li Xing <@li-xingHIT>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import hashlib
import json

import pytest

from numpilot.add_one_gap import (
    AddOneGapError,
    AddOneLoweringVerificationError,
    add_one_gap_summary,
    detect_add_one_calls,
    lower_add_one_calls,
    materialize_add_one_operator_spec,
    verify_add_one_lowering,
)


SOURCE = """import numpy as np

x = np.linspace(0.0, 1.0, 16, dtype=np.float32)
# NUMPILOT_DEVICE_INPUTS_END
result = np.add(x, 1.0)
"""


def test_detect_and_lower_registered_add_one_gap(tmp_path):
    source_path = tmp_path / "source.py"
    source_path.write_text(SOURCE, encoding="utf-8")
    calls = detect_add_one_calls(SOURCE)

    assert len(calls) == 1
    assert calls[0].input_name == "x"
    assert calls[0].lineno == 5

    output = tmp_path / "private"
    normalized, report_path = lower_add_one_calls(
        source_path=str(source_path),
        source=SOURCE,
        calls=calls,
        out_dir=output,
    )
    text = normalized.read_text(encoding="utf-8")
    assert "np.add(x, np.ones_like(x, dtype=np.float32))" in text
    assert "NUMPILOT_DEVICE_INPUTS_END" in text
    assert not detect_add_one_calls(text)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "lowered_unverified"
    assert report["active_route"] == "COMPOSITE_NPU"
    assert report["target_api_contract"] == "asnumpy.add_one"
    assert report["lowered_program_apis"] == ["numpy.ones_like", "numpy.add"]
    assert report["expected_migrated_program_apis"] == [
        "asnumpy.ones_like",
        "asnumpy.add",
    ]
    assert report["migrated_program_uses_target_api"] is False
    assert report["interface_activation_required_by_lowered_program"] is False
    assert report["source_sha256"] == hashlib.sha256(SOURCE.encode()).hexdigest()
    assert report["cannbot_custom_kernel_activated"] is False

    summary = add_one_gap_summary(calls)
    assert summary["target_api_contract"] == "asnumpy.add_one"
    assert summary["lowered_program_apis"] == ["numpy.ones_like", "numpy.add"]
    assert summary["expected_migrated_program_apis"] == [
        "asnumpy.ones_like",
        "asnumpy.add",
    ]
    assert summary["migrated_program_uses_target_api"] is False
    assert summary["interface_activation_required_by_lowered_program"] is False


def test_materialized_operator_spec_binds_original_source(tmp_path):
    source_path = tmp_path / "source.py"
    source_path.write_text(SOURCE, encoding="utf-8")
    calls = detect_add_one_calls(SOURCE)

    spec, relation = materialize_add_one_operator_spec(
        original_source_path=str(source_path),
        original_source=SOURCE,
        calls=calls,
        out_dir=tmp_path / "gap",
        target_soc="Ascend910B4",
        cann_version="9.0.0",
    )

    assert "@operator_spec" in spec.read_text(encoding="utf-8")
    payload = json.loads(relation.read_text(encoding="utf-8"))
    assert payload["original_source_sha256"] == hashlib.sha256(
        SOURCE.encode()
    ).hexdigest()
    assert payload["target_api"] == "asnumpy.add_one"
    assert payload["claims"]["interface_available"] is False

    from numpilot.gaps.pipeline import prepare_gap_completion

    preparation = prepare_gap_completion(
        str(spec), target_soc="Ascend910B4", cann_version="9.0.0"
    )
    assert len(preparation.specs) == 1
    assert preparation.specs[0].qualified_api == "operator.add_one"


@pytest.mark.parametrize(
    "expression",
    (
        "np.add(x, 2.0)",
        "np.add(x, one)",
        "np.add(1.0, x)",
        "np.add(x, 1.0, dtype=np.float32)",
        "other.add(x, 1.0)",
    ),
)
def test_detector_rejects_non_exact_patterns(expression):
    source = "import numpy as np\nx = np.ones(2, dtype=np.float32)\nresult = %s\n" % expression
    assert detect_add_one_calls(source) == ()


def test_lowering_rejects_source_drift(tmp_path):
    source_path = tmp_path / "source.py"
    source_path.write_text(SOURCE, encoding="utf-8")
    calls = detect_add_one_calls(SOURCE)
    changed = SOURCE.replace("1.0)", "1)")

    with pytest.raises(AddOneGapError, match="source-bound"):
        lower_add_one_calls(
            source_path=str(source_path),
            source=changed,
            calls=calls,
            out_dir=tmp_path / "out",
        )


@pytest.mark.parametrize("confirmation", (False, None, 1))
def test_lowering_verification_requires_literal_confirmation_without_side_effects(
    tmp_path, confirmation
):
    destination = tmp_path / "must-not-exist" / "receipt.json"

    with pytest.raises(
        AddOneLoweringVerificationError, match="explicit user confirmation"
    ) as caught:
        verify_add_one_lowering(
            tmp_path / "missing-source.py",
            tmp_path / "missing-normalized.py",
            tmp_path / "missing-report.json",
            ["result"],
            destination,
            user_confirmed=confirmation,
        )

    assert caught.value.code == "confirmation_required"
    assert caught.value.receipt is None
    assert not destination.parent.exists()


def _lowering_fixture(tmp_path):
    source_path = tmp_path / "source.py"
    source_path.write_text(SOURCE, encoding="utf-8")
    normalized, report = lower_add_one_calls(
        source_path=str(source_path),
        source=SOURCE,
        calls=detect_add_one_calls(SOURCE),
        out_dir=tmp_path / "lowering",
    )
    return source_path, normalized, report


def test_lowering_verification_publishes_bound_numerical_receipt(tmp_path):
    source, normalized, report = _lowering_fixture(tmp_path)
    destination = tmp_path / "receipts" / "add-one-lowering-verification.json"

    receipt = verify_add_one_lowering(
        source,
        normalized,
        report,
        ["result"],
        destination,
        user_confirmed=True,
    )

    payload = receipt.to_dict()
    assert receipt.path == str(destination.absolute())
    assert payload["status"] == "passed"
    assert payload["receipt_type"] == (
        "numpilot_add_one_lowering_numerical_verification"
    )
    assert payload["user_confirmation"] == {"required": True, "received": True}
    assert payload["source"]["sha256"] == hashlib.sha256(
        source.read_bytes()
    ).hexdigest()
    assert payload["normalized_source"]["sha256"] == hashlib.sha256(
        normalized.read_bytes()
    ).hexdigest()
    assert payload["lowering"]["report_sha256"] == hashlib.sha256(
        report.read_bytes()
    ).hexdigest()
    assert payload["lowering"]["target_api_contract"] == "asnumpy.add_one"
    assert payload["lowering"]["lowered_program_apis"] == [
        "numpy.ones_like",
        "numpy.add",
    ]
    assert payload["lowering"]["expected_migrated_program_apis"] == [
        "asnumpy.ones_like",
        "asnumpy.add",
    ]
    assert payload["lowering"]["migrated_program_uses_target_api"] is False
    assert (
        payload["lowering"]["interface_activation_required_by_lowered_program"]
        is False
    )
    assert payload["comparisons"]["result"]["consistent"] is True
    assert set(payload["executions"]) == {"source", "normalized"}
    assert payload["claims"]["source_to_normalized_outputs_consistent"] is True
    assert payload["claims"]["constituent_lowering_numerically_verified"] is True
    assert payload["claims"]["migrated_program_uses_add_one_interface"] is False
    assert payload["claims"]["add_one_interface_integration_verified"] is False
    assert payload["claims"]["strict_npu_provenance_verified"] is False
    assert payload["claims"]["performance_verified"] is False
    assert payload["claims"]["cannbot_custom_kernel_activated"] is False
    assert destination.stat().st_mode & 0o777 == 0o600


def test_lowering_verification_mismatch_fails_closed_and_is_write_once(tmp_path):
    source, normalized, report = _lowering_fixture(tmp_path)
    changed = normalized.read_text(encoding="utf-8").replace(
        "np.ones_like(x, dtype=np.float32)",
        "np.full_like(x, 2.0, dtype=np.float32)",
    )
    normalized.write_text(changed, encoding="utf-8")
    report_value = json.loads(report.read_text(encoding="utf-8"))
    report_value["normalized_source_sha256"] = hashlib.sha256(
        changed.encode("utf-8")
    ).hexdigest()
    report.write_text(
        json.dumps(report_value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    destination = tmp_path / "failed.json"

    with pytest.raises(AddOneLoweringVerificationError) as caught:
        verify_add_one_lowering(
            source,
            normalized,
            report,
            ["result"],
            destination,
            user_confirmed=True,
        )

    assert caught.value.code == "numerical_mismatch"
    assert caught.value.receipt is not None
    failed = json.loads(destination.read_text(encoding="utf-8"))
    assert failed["status"] == "failed"
    assert failed["failure"]["code"] == "numerical_mismatch"
    assert failed["claims"]["source_to_normalized_outputs_consistent"] is False
    assert failed["claims"]["constituent_lowering_numerically_verified"] is False
    assert failed["claims"]["migrated_program_uses_add_one_interface"] is False
    assert failed["claims"]["add_one_interface_integration_verified"] is False
    original_receipt = destination.read_bytes()

    with pytest.raises(AddOneLoweringVerificationError) as repeated:
        verify_add_one_lowering(
            source,
            normalized,
            report,
            ["result"],
            destination,
            user_confirmed=True,
        )
    assert repeated.value.code == "receipt_exists"
    assert destination.read_bytes() == original_receipt


def test_lowering_verification_rejects_tampered_report_before_execution(tmp_path):
    source, normalized, report = _lowering_fixture(tmp_path)
    report_value = json.loads(report.read_text(encoding="utf-8"))
    report_value["active_route"] = "CUSTOM_KERNEL"
    report.write_text(json.dumps(report_value) + "\n", encoding="utf-8")
    destination = tmp_path / "preflight-failure.json"

    with pytest.raises(AddOneLoweringVerificationError) as caught:
        verify_add_one_lowering(
            source,
            normalized,
            report,
            ["result"],
            destination,
            user_confirmed=True,
        )

    assert caught.value.code == "invalid_lowering_report"
    assert caught.value.receipt is None
    assert not destination.exists()


def test_lowering_verification_rejects_false_interface_usage_claim(tmp_path):
    source, normalized, report = _lowering_fixture(tmp_path)
    report_value = json.loads(report.read_text(encoding="utf-8"))
    report_value["migrated_program_uses_target_api"] = True
    report.write_text(json.dumps(report_value) + "\n", encoding="utf-8")
    destination = tmp_path / "preflight-failure.json"

    with pytest.raises(AddOneLoweringVerificationError) as caught:
        verify_add_one_lowering(
            source,
            normalized,
            report,
            ["result"],
            destination,
            user_confirmed=True,
        )

    assert caught.value.code == "invalid_lowering_report"
    assert caught.value.receipt is None
    assert not destination.exists()


@pytest.mark.parametrize(
    "annotation",
    (
        "marker: exec(\"open({marker!r}, 'w').write('pwn')\")",
        "x: open({marker!r}, 'w') = np.ones(4, dtype=np.float32)",
    ),
)
def test_runtime_annotation_side_effect_is_rejected_before_execution(
    tmp_path, annotation
):
    marker = tmp_path / "pwn"
    source_text = (
        "import numpy as np\n"
        + annotation.format(marker=str(marker))
        + "\nx = np.ones(4, dtype=np.float32)\n"
        + "result = np.add(x, 1.0)\n"
    )
    source = tmp_path / "unsafe-annotation.py"
    source.write_text(source_text, encoding="utf-8")

    with pytest.raises(AddOneGapError, match="static admission"):
        lower_add_one_calls(
            source_path=str(source),
            source=source_text,
            calls=detect_add_one_calls(source_text),
            out_dir=tmp_path / "lowering",
        )

    assert not marker.exists()


def test_postponed_annotation_is_safe_through_numerical_execution(tmp_path):
    marker = tmp_path / "must-not-exist"
    source_text = (
        "from __future__ import annotations\n"
        "import numpy as np\n"
        "x: open(%r, 'w') = np.ones(4, dtype=np.float32)\n"
        "result = np.add(x, 1.0)\n" % str(marker)
    )
    source = tmp_path / "postponed-annotation.py"
    source.write_text(source_text, encoding="utf-8")
    normalized, report = lower_add_one_calls(
        source_path=str(source),
        source=source_text,
        calls=detect_add_one_calls(source_text),
        out_dir=tmp_path / "lowering",
    )

    receipt = verify_add_one_lowering(
        source,
        normalized,
        report,
        ["result"],
        tmp_path / "verification.json",
        user_confirmed=True,
    )

    assert receipt.status == "passed"
    assert not marker.exists()
