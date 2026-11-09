from pathlib import Path

import pytest

from numpilot.gaps.official_candidate_preflight import (
    OfficialCandidatePreflightError,
    _safe_under,
    preflight_official_candidate,
)


def test_safe_under_rejects_absolute_and_escape(tmp_path):
    root = tmp_path / "cann"
    root.mkdir()
    with pytest.raises(OfficialCandidatePreflightError):
        _safe_under(str(root), "/etc/passwd", "header")
    with pytest.raises(OfficialCandidatePreflightError):
        _safe_under(str(root), "../outside", "header")


def test_preflight_rejects_tampered_handoff_before_resolution(tmp_path):
    handoff = tmp_path / "handoff.json"
    handoff.write_text('{"schema":"numpilot.official-candidate-gap.v1"}\n', encoding="utf-8")
    source = tmp_path / "source.py"
    source.write_text("import numpy as np\n", encoding="utf-8")
    resolution = tmp_path / "resolution.json"
    resolution.write_text("{}\n", encoding="utf-8")
    with pytest.raises(OfficialCandidatePreflightError):
        preflight_official_candidate(
            str(handoff), str(source), str(resolution), str(tmp_path / "out.json")
        )


def test_preflight_receipt_is_write_once(tmp_path):
    # The public function reaches hand-off validation first; an existing output
    # must never be replaced even when an input later fails validation.
    handoff = tmp_path / "handoff.json"
    handoff.write_text("{}\n", encoding="utf-8")
    source = tmp_path / "source.py"
    source.write_text("import numpy as np\n", encoding="utf-8")
    resolution = tmp_path / "resolution.json"
    resolution.write_text("{}\n", encoding="utf-8")
    out = tmp_path / "out.json"
    out.write_text("sentinel\n", encoding="utf-8")
    with pytest.raises(OfficialCandidatePreflightError):
        preflight_official_candidate(str(handoff), str(source), str(resolution), str(out))
    assert out.read_text(encoding="utf-8") == "sentinel\n"
from pathlib import Path

import pytest

from numpilot.gaps.official_candidate_preflight import (
    OfficialCandidatePreflightError,
    prepare_official_adapter_input_from_preflight,
)


def test_adapter_bridge_rejects_missing_or_tampered_inputs(tmp_path):
    handoff = tmp_path / "handoff.json"
    preflight = tmp_path / "preflight.json"
    source = tmp_path / "source.py"
    resolution = tmp_path / "resolution.json"
    for path, text in (
        (handoff, "{}\n"),
        (preflight, "{}\n"),
        (source, "import numpy as np\n"),
        (resolution, "{}\n"),
    ):
        path.write_text(text, encoding="utf-8")
    with pytest.raises(OfficialCandidatePreflightError):
        prepare_official_adapter_input_from_preflight(
            str(handoff), str(preflight), str(source), str(resolution)
        )
