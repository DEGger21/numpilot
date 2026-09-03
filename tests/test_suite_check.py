import json
from pathlib import Path

import pytest

from numpilot.__main__ import main
from numpilot.suite_check import (
    SuiteCheckError,
    check_suite,
    public_suite_report,
    write_suite_report,
)


DIRECT = """\
import numpy as np

def build_inputs():
    return (np.ones(8, dtype=np.float32),)

def forward(x):
    return np.square(x)
"""


SINGLE_GAP = """\
import numpy as np

def build_inputs():
    return (np.linspace(-8.0, 8.0, 17, dtype=np.float32),)

def forward(x):
    return np.cbrt(x)
"""


def _write(path: Path, source: str) -> Path:
    path.write_text(source, encoding="utf-8")
    return path


def test_five_plus_five_suite_contract_is_counted_without_execution(tmp_path):
    marker = tmp_path / "must-not-exist"
    direct_with_main = DIRECT + (
        "\nif __name__ == '__main__':\n"
        "    open(%r, 'w').write('executed')\n" % str(marker)
    )
    paths = []
    for index in range(5):
        source = direct_with_main if index == 0 else DIRECT
        paths.append(_write(tmp_path / ("direct_%d.py" % index), source))
        paths.append(_write(tmp_path / ("gap_%d.py" % index), SINGLE_GAP))

    report = check_suite(
        [str(path) for path in paths],
        target_soc="ascend910b",
        cann_version="9.0.0",
        expected_direct=5,
        expected_single_gap=5,
    )

    assert report["status"] == "pass"
    assert report["execution_performed"] is False
    assert report["runtime_capability_checked"] is False
    assert report["counts"] == {
        "direct_migration_candidate": 5,
        "single_gap_candidate": 5,
        "blocked": 0,
    }
    assert not marker.exists()
    gap = next(
        item for item in report["cases"]
        if item["category"] == "single_gap_candidate"
    )
    assert gap["generatable_gaps"][0]["qualified_api"] == "numpy.cbrt"
    assert gap["gap_lane"]["kind"] == "simple_unary_binary_ufunc"


def test_suite_check_reports_nested_namespace_and_multiple_gaps(tmp_path):
    nested = _write(
        tmp_path / "nested.py",
        SINGLE_GAP.replace("np.cbrt(x)", "np.fft.fft(x)"),
    )
    multiple = _write(
        tmp_path / "multiple.py",
        SINGLE_GAP.replace("np.cbrt(x)", "np.cbrt(np.spacing(x))"),
    )

    report = check_suite(
        [str(nested), str(multiple)],
        target_soc="ascend910b",
        cann_version="9.0.0",
    )

    assert report["status"] == "fail"
    assert report["counts"]["blocked"] == 2
    nested_result = next(
        item for item in report["cases"] if item["path"].endswith("nested.py")
    )
    assert any("嵌套 NumPy namespace" in error for error in nested_result["errors"])
    multiple_result = next(
        item for item in report["cases"] if item["path"].endswith("multiple.py")
    )
    assert "case contains more than one generatable missing API" in multiple_result["errors"]


def test_suite_check_rejects_alias_gap_scalar_signature_and_live_top_level(tmp_path):
    alias = _write(
        tmp_path / "alias.py",
        SINGLE_GAP.replace("np.cbrt(x)", "np.abs(x)"),
    )
    scalar = _write(
        tmp_path / "scalar.py",
        DIRECT.replace("np.square(x)", "np.add(x, 1.0)"),
    )
    live = _write(
        tmp_path / "live.py",
        DIRECT.replace("import numpy as np", "import numpy as np\nREADY = True"),
    )

    report = check_suite(
        [str(alias), str(scalar), str(live)],
        target_soc="ascend910b",
        cann_version="9.0.0",
    )

    assert report["counts"]["blocked"] == 3
    errors = {
        Path(item["path"]).name: item["errors"] for item in report["cases"]
    }
    assert any("aliases an already covered" in item for item in errors["alias.py"])
    assert any("标量" in item for item in errors["scalar.py"])
    assert any("top-level Assign" in item for item in errors["live.py"])


def test_suite_check_blocks_numpy_keyword_that_asnumpy_cannot_rewrite(tmp_path):
    incompatible = _write(
        tmp_path / "linspace_num.py",
        DIRECT.replace("np.square(x)", "np.linspace(0.0, 1.0, num=8)"),
    )

    report = check_suite(
        [str(incompatible)],
        target_soc="ascend910b",
        cann_version="9.0.0",
    )

    assert report["status"] == "fail"
    assert report["counts"]["blocked"] == 1
    case = report["cases"][0]
    assert any(
        "num" in item.get("reason", "")
        for item in case["blocked_unsupported"]
    )


def test_suite_check_audits_build_inputs_without_executing_it(tmp_path):
    marker = tmp_path / "must-not-be-written"
    unsafe = _write(
        tmp_path / "unsafe_inputs.py",
        DIRECT.replace(
            "return (np.ones(8, dtype=np.float32),)",
            "open(%r, 'w').write('bad')\n    "
            "return (np.ones(8, dtype=np.float32),)" % str(marker),
        ),
    )
    wrong_arity = _write(
        tmp_path / "wrong_arity.py",
        DIRECT.replace("def forward(x):", "def forward(x, y):"),
    )

    report = check_suite(
        [str(unsafe), str(wrong_arity)],
        target_soc="ascend910b",
        cann_version="9.0.0",
    )

    assert report["counts"]["blocked"] == 2
    assert not marker.exists()
    errors = {
        Path(item["path"]).name: item["errors"] for item in report["cases"]
    }
    assert any("build_inputs statement" in item for item in errors["unsafe_inputs.py"])
    assert any("tuple arity" in item for item in errors["wrong_arity.py"])


def test_suite_check_rejects_scalar_input_for_array_only_asnumpy_binding(tmp_path):
    source = DIRECT.replace(
        "return (np.ones(8, dtype=np.float32),)",
        "return (np.ones(8, dtype=np.float32), np.random.rand())",
    ).replace("def forward(x):", "def forward(x, scale):").replace(
        "np.square(x)", "np.add(x, scale)"
    )
    case = _write(tmp_path / "scalar_operand.py", source)

    report = check_suite(
        [str(case)], target_soc="ascend910b", cann_version="9.0.0"
    )

    assert report["counts"]["blocked"] == 1
    assert any(
        "静态证明为数组" in item.get("reason", "")
        for item in report["cases"][0]["blocked_unsupported"]
    )


def test_suite_check_allows_simple_local_input_construction(tmp_path):
    source = DIRECT.replace(
        "return (np.ones(8, dtype=np.float32),)",
        "inputs = np.arange(8, dtype=np.float32)\n    return (inputs,)",
    )
    case = _write(tmp_path / "local_inputs.py", source)

    report = check_suite(
        [str(case)],
        target_soc="ascend910b",
        cann_version="9.0.0",
        expected_direct=1,
        expected_single_gap=0,
    )

    assert report["status"] == "pass"


@pytest.mark.parametrize("api", ["relu", "gelu", "softmax", "angle"])
def test_suite_check_blocks_non_numpy_or_unaudited_direct_api(tmp_path, api):
    case = _write(
        tmp_path / (api + ".py"),
        DIRECT.replace("np.square(x)", "np.%s(x)" % api),
    )

    report = check_suite(
        [str(case)],
        target_soc="ascend910b",
        cann_version="9.0.0",
    )

    assert report["status"] == "fail"
    assert any(
        "not audited current AsNumpy targets" in error
        for error in report["cases"][0]["errors"]
    )


def test_suite_report_is_private_json_and_symlinks_are_rejected(tmp_path):
    case = _write(tmp_path / "direct.py", DIRECT)
    report = check_suite(
        [str(case)],
        target_soc="ascend910b",
        cann_version="9.0.0",
        expected_direct=1,
        expected_single_gap=0,
    )
    output = tmp_path / "report.json"

    write_suite_report(str(output), report)

    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "pass"
    assert output.stat().st_mode & 0o777 == 0o600
    linked = tmp_path / "linked.py"
    linked.symlink_to(case)
    with pytest.raises(SuiteCheckError, match="symlink"):
        check_suite(
            [str(linked)],
            target_soc="ascend910b",
            cann_version="9.0.0",
        )


def test_public_suite_report_redacts_nested_gap_source_without_mutating_receipt(
    tmp_path,
):
    case = _write(tmp_path / "gap.py", SINGLE_GAP)
    private = check_suite(
        [str(case)], target_soc="ascend910b", cann_version="9.0.0"
    )

    public = public_suite_report(private)

    assert public["path_redacted"] is True
    assert public["cases"][0]["path"] == "gap.py"
    assert (
        public["cases"][0]["generatable_gaps"][0]["source_path"]
        == "gap.py"
    )
    assert private["cases"][0]["path"] == str(case.absolute())
    assert (
        private["cases"][0]["generatable_gaps"][0]["source_path"]
        == str(case.absolute())
    )


def test_suite_expectation_mismatch_fails_without_reclassifying_case(tmp_path):
    case = _write(tmp_path / "direct.py", DIRECT)

    report = check_suite(
        [str(case)],
        target_soc="ascend910b",
        cann_version="9.0.0",
        expected_direct=5,
        expected_single_gap=5,
    )

    assert report["status"] == "fail"
    assert report["cases"][0]["category"] == "direct_migration_candidate"
    assert len(report["expectation_errors"]) == 3


def test_check_suite_cli_writes_report_and_uses_nonzero_for_mismatch(
    tmp_path, capsys
):
    direct = _write(tmp_path / "direct.py", DIRECT)
    gap = _write(tmp_path / "gap.py", SINGLE_GAP)
    output = tmp_path / "suite.json"

    passed = main([
        "check-suite",
        str(direct),
        str(gap),
        "--expect-direct", "1",
        "--expect-single-gap", "1",
        "--out", str(output),
    ])
    failed = main([
        "check-suite",
        str(direct),
        str(gap),
        "--expect-direct", "5",
        "--expect-single-gap", "5",
    ])

    assert passed == 0
    assert failed == 1
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "pass"
    terminal = capsys.readouterr()
    assert "execution_performed=no" in terminal.out
    assert "expected 5 direct candidates" in terminal.err


def test_check_suite_cli_json_is_strict_stdout_and_can_publish_receipt(
    tmp_path, capsys
):
    direct = _write(tmp_path / "direct.py", DIRECT)
    output = tmp_path / "suite.json"

    rc = main([
        "check-suite",
        str(direct),
        "--expect-direct", "1",
        "--expect-single-gap", "0",
        "--out", str(output),
        "--json",
    ])

    assert rc == 0
    captured = capsys.readouterr()
    # A JSON consumer must not have to strip the normal human summary.
    payload = json.loads(captured.out)
    assert payload["status"] == "pass"
    assert payload["path_redacted"] is True
    assert payload["execution_performed"] is False
    assert payload["counts"]["direct_migration_candidate"] == 1
    assert payload["cases"][0]["path"] == "direct.py"
    assert str(tmp_path) not in captured.out
    assert captured.err == ""
    private_payload = json.loads(output.read_text(encoding="utf-8"))
    assert private_payload["status"] == "pass"
    assert private_payload["cases"][0]["path"] == str(direct.absolute())


def test_check_suite_cli_json_failure_preserves_exit_contract(tmp_path, capsys):
    direct = _write(tmp_path / "direct.py", DIRECT)

    rc = main([
        "check-suite",
        str(direct),
        "--expect-direct", "2",
        "--json",
    ])

    assert rc == 1
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["status"] == "fail"
    assert payload["path_redacted"] is True
    assert payload["expectation_errors"]
    assert "[测试集准入]" not in captured.out
    # Diagnostics remain available to a human/CI log on stderr without
    # corrupting the machine-readable stdout stream.
    assert captured.err == ""
