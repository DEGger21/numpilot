from numpilot.orchestrator import Orchestrator


def test_e2e_mlp_migrates_and_verifies():
    res = Orchestrator().run("examples/mlp_forward.py")
    assert res["status"] == "pass"
    assert res["report"].numerically_consistent
    assert "ap.matmul(" in res["migrated_source"]


def test_e2e_unsupported_stops_early():
    res = Orchestrator().run("examples/unsupported_demo.py")
    assert res["status"] == "unsupported"
    assert "fft" in res["unsupported"]


def test_e2e_elementwise_strict_npu_candidate_under_shim():
    res = Orchestrator().run("examples/elementwise_forward.py")
    assert res["status"] == "pass"
    assert res["plan"].strategy == "whole_forward_npu"
    assert res["report"].artifact_inputs_executed
    assert "ap.square(" in res["migrated_source"]


def test_from_numpy_import_migrates_and_verifies_end_to_end(tmp_path):
    source = tmp_path / "from_import.py"
    source.write_text(
        "from numpy import array, sqrt as root\n"
        "def build_inputs(): return (array([1.0, 4.0, 9.0]),)\n"
        "def forward(x): return root(x)\n",
        encoding="utf-8",
    )

    result = Orchestrator().run(str(source))

    assert result["status"] == "pass"
    assert result["expected_rewrites"] == {"sqrt": 1}
    assert result["rewrites"] == {"sqrt": 1}
    assert "ap.sqrt(x)" in result["migrated_source"]
    assert "import numpy as _numpilot_numpy" in result["migrated_source"]
    assert "return root(x)" not in result["migrated_source"]


def test_wrong_operator_with_same_call_count_is_rejected(tmp_path):
    from numpilot.backends.shim import ShimBackend

    source = tmp_path / "sqrt_case.py"
    source.write_text(
        "import numpy as np\n"
        "def build_inputs(): return (np.ones(3, dtype=np.float32),)\n"
        "def forward(x): return np.sqrt(x)\n",
        encoding="utf-8",
    )

    class WrongOperatorEngine:
        name = "wrong-op"

        def rewrite(self, source, plan):
            return (
                "import numpy as np\nimport asnumpy as ap\n"
                "def build_inputs(): return (np.ones(3, dtype=np.float32),)\n"
                "def forward(x): return ap.square(x)\n"
            )

    result = Orchestrator(
        backend=ShimBackend(), engine=WrongOperatorEngine()
    ).run(str(source), out_dir=str(tmp_path / "wrong-op-out"))

    assert result["status"] == "fail"
    assert not result["rewrite_integrity"]
    assert result["report"].exception_phase == "migrated.surface"
    assert "expected={'sqrt': 1}" in result["report"].exception_message


def test_shim_markdown_does_not_claim_strict_npu_gate(tmp_path):
    from numpilot.backends.shim import ShimBackend

    out = tmp_path / "shim-report"
    result = Orchestrator(backend=ShimBackend()).run(
        "examples/elementwise_forward.py", out_dir=str(out)
    )
    assert result["status"] == "pass"
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "strict-NPU 门禁:未执行（shim/非 strict）" in report
    assert "strict-NPU 门禁:True" not in report
