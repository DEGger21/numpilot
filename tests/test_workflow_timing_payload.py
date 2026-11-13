from numpilot import workflow


def _stage(name, elapsed):
    return workflow.WorkflowStage(
        name=name, argv=("python",), returncode=0, status="passed",
        elapsed_seconds=elapsed, stdout_path="out", stderr_path="err",
        stdout_sha256="a" * 64, stderr_sha256="b" * 64,
    )


def test_workflow_timing_payload_is_compact_and_stage_bound():
    assert workflow._workflow_timing_payload([_stage("01-analyze", 1.25), _stage("02-migrate", 2.75)]) == {
        "stage_elapsed_seconds": {"01-analyze": 1.25, "02-migrate": 2.75},
        "total_stage_elapsed_seconds": 4.0,
    }


def test_workflow_performance_payload_whitelists_benchmark_summary():
    payload = workflow._workflow_performance_payload({
        "performance": {
            "performance_valid": True,
            "cpu_median_ms": 100.0,
            "npu_median_ms": 40.0,
            "median_speedup": 2.5,
            "issues": [],
            "median_speedup_bootstrap_95ci": [2.0, 3.0],
            "warm_runtime_single_pass_component_sum": {
                "directly_sampled_end_to_end": False,
                "formula": "cpu + npu",
                "cpu_ms": 110.0,
                "npu_ms": 50.0,
                "private_path": "/should/not/leak",
            },
            "private_path": "/should/not/leak",
        },
        "workers": {"npu": {"logs": "/should/not/leak"}},
    })
    assert payload == {
        "performance_valid": True,
        "issues": [],
        "cpu_median_ms": 100.0,
        "npu_median_ms": 40.0,
        "median_speedup": 2.5,
        "median_speedup_bootstrap_95ci": [2.0, 3.0],
        "warm_runtime_single_pass_component_sum": {
            "directly_sampled_end_to_end": False,
            "formula": "cpu + npu",
            "cpu_ms": 110.0,
            "npu_ms": 50.0,
        },
    }
