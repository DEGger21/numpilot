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
