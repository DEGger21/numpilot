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

"""Contract tests for the interactive session navigation report."""

from __future__ import annotations

import json

from numpilot.interactive_report import (
    build_session_report,
    render_session_markdown,
    write_session_reports,
)


def test_report_repeats_only_receipt_validated_performance(tmp_path):
    session = tmp_path / "session"
    session.mkdir(mode=0o700)
    workflow = session / "workflow"
    (workflow / "migration").mkdir(parents=True)
    (workflow / "migration" / "migrated.py").write_text(
        "import asnumpy as np\nresult = np.ones(4)\n", encoding="utf-8"
    )
    (workflow / "workflow-report.json").write_text("{}\n", encoding="utf-8")
    (workflow / "script-verification-report.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "comparisons": {
                    "result": {
                        "consistent": True,
                        "n_elements": 4,
                        "max_abs_diff": 0.0,
                        "max_rel_diff": 0.0,
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (workflow / "script-benchmark-report.json").write_text(
        json.dumps(
            {
                "performance": {
                    "performance_valid": True,
                    "cpu_median_ms": 12.0,
                    "npu_median_ms": 4.0,
                    "median_speedup": 3.0,
                    "conservative_cpu_p10_over_npu_p90": 2.5,
                },
                "workers": {
                    "npu": {
                        "outputs": [
                            {
                                "shape": [4],
                                "dtype": "float32",
                                "elements": 4,
                                "data_bytes": 16,
                            }
                        ]
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    workflow_result = {
        "status": "passed",
        "workflow_report": str(workflow / "workflow-report.json"),
        "stages": [{"name": "04-benchmark", "status": "passed", "elapsed_seconds": 1.0}],
    }

    report = build_session_report(
        session_root=session,
        source={"path": "/tmp/input.py", "sha256": "a" * 64},
        status="passed",
        gap_result={
            "status": "qualified_composite_interface",
            "cannbot_draft_activated": False,
            "claim_boundary": {"interface_available": True, "activated": True},
        },
        workflow=workflow_result,
    )

    performance = report["workflow"]["performance"]
    assert performance["performance_valid"] is True
    assert performance["speedup"] == 3.0
    assert performance["measured_scale"]["elements"] == 4
    assert performance["advantage_threshold"] is None
    assert report["claims_boundary"]["operator_draft_is_not_activated"] is True
    assert "3.0 x" in render_session_markdown(report)
    json_path, markdown_path = write_session_reports(session, report)
    assert json_path.is_file()
    assert markdown_path.is_file()


def test_report_drops_unvalidated_speedup_and_untrusted_gap_fields(tmp_path):
    session = tmp_path / "session"
    session.mkdir(mode=0o700)
    workflow = session / "workflow"
    workflow.mkdir()
    (workflow / "workflow-report.json").write_text("{}\n", encoding="utf-8")
    (workflow / "script-benchmark-report.json").write_text(
        '{"performance":{"performance_valid":false,"median_speedup":999}}\n',
        encoding="utf-8",
    )
    report = build_session_report(
        session_root=session,
        source={"path": "/tmp/input.py", "sha256": "b" * 64},
        status="blocked",
        gap_result={
            "status": "generated_unverified",
            "model_body": "must-not-copy",
            "claim_boundary": {"interface_available": False},
        },
        workflow={
            "status": "blocked",
            "workflow_report": str(workflow / "workflow-report.json"),
            "stages": [],
        },
    )

    assert report["workflow"]["performance"]["speedup"] is None
    assert "model_body" not in report["gap_recovery"]
    assert "999" not in render_session_markdown(report)
