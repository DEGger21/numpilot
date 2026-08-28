from pathlib import Path

import numpy as np
import pytest

from numpilot import OperatorSpec, operator_spec
from numpilot.agents.analysis import AnalysisAgent
from numpilot.agents.decision import DecisionAgent
from numpilot.agents.migration import MigrationAgent
from numpilot.backends.shim import ShimBackend
from numpilot.gaps.pipeline import GapCompletionError, GapCompletionPipeline
from numpilot.gaps.spec import (
    build_gap_specs,
    extract_semantic_contract,
    render_generation_request,
)
from numpilot.orchestrator import Orchestrator


DOMAIN_SOURCE = '''\
import numpy as np
from numpilot import operator_spec

@operator_spec(
    name="apply_gradient_descent",
    version="1",
    numpy_oracle="gradient_oracle",
    target_api="asnumpy.apply_gradient_descent",
    provenance="ops_nn_reuse_candidate",
    route="cann_existing_reuse_candidate",
    signature_variants=({"positional_args": 3, "keyword_args": ()},),
    dtype_rules=("float32",),
    shape_rules=("rank>=1", "contiguous"),
    source="https://gitcode.com/cann/ops-nn.git@0123456789abcdef0123456789abcdef01234567:experimental/optim/apply_gradient_descent",
    license="CANN Open Software License Agreement Version 2.0",
    target_soc="ascend910b",
    cann_version="9.0.0",
)
def gradient_oracle(var, alpha, delta):
    return np.subtract(var, np.multiply(alpha, delta))

def build_inputs():
    return (
        np.array([1.0, 2.0], dtype=np.float32),
        np.array([0.1, 0.1], dtype=np.float32),
        np.array([2.0, -1.0], dtype=np.float32),
    )

def forward(var, alpha, delta):
    return gradient_oracle(var, alpha, delta)
'''


def _spec_kwargs():
    return {
        "name": "apply_gradient_descent",
        "version": "1",
        "numpy_oracle": "oracle",
        "target_api": "asnumpy.apply_gradient_descent",
        "provenance": "ops_nn_reuse_candidate",
        "route": "cann_existing_reuse_candidate",
        "signature_variants": (
            {"positional_args": 3, "keyword_args": ()},
        ),
        "dtype_rules": ("float32",),
        "shape_rules": ("rank>=1", "contiguous"),
        "source": "https://gitcode.com/cann/ops-nn.git@0123456789abcdef0123456789abcdef01234567:path",
        "license": "CANN Open Software License Agreement Version 2.0",
        "target_soc": "ascend910b",
        "cann_version": "9.0.0",
    }


def test_runtime_decorator_attaches_round_trippable_contract():
    @operator_spec(**_spec_kwargs())
    def oracle(var, alpha, delta):
        return var - alpha * delta

    spec = oracle.__numpilot_operator_spec__
    restored = OperatorSpec.from_dict(spec.to_dict())

    assert restored == spec
    assert restored.qualified_api == "operator.apply_gradient_descent"
    np.testing.assert_allclose(
        oracle(np.array([1.0]), np.array([0.1]), np.array([2.0])),
        np.array([0.8]),
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("license", "Apache-2.0", "CANN OSL"),
        (
            "source",
            "https://example.com/cann/ops-nn.git@"
            + "0" * 40
            + ":optim/apply_gradient_descent",
            "official origin",
        ),
        (
            "source",
            "https://gitcode.com/cann/ops-nn.git@"
            + "0" * 16
            + ":optim/apply_gradient_descent",
            "full commit",
        ),
        (
            "source",
            "https://gitcode.com/cann/ops-nn.git@"
            + "0" * 40
            + ":optim/../apply_gradient_descent",
            "unsafe",
        ),
    ),
)
def test_ops_nn_reuse_requires_official_hash_bound_source_and_license(
    field, value, message
):
    kwargs = _spec_kwargs()
    kwargs[field] = value

    with pytest.raises(ValueError, match=message):
        OperatorSpec(**kwargs)


def test_analysis_and_ast_rewrite_domain_helper_without_numpy_spoofing():
    analysis = AnalysisAgent().analyze(DOMAIN_SOURCE, "case.py")
    domain = [op for op in analysis.op_uses if op.kind == "domain_operator"]

    assert analysis.migratable
    assert len(domain) == 1
    assert domain[0].qualified_name == "operator.apply_gradient_descent"
    assert domain[0].target_api == "asnumpy.apply_gradient_descent"
    assert domain[0].requires_qualification

    migration = MigrationAgent().migrate(
        DOMAIN_SOURCE,
        DecisionAgent().plan(analysis),
    )
    assert "ap.apply_gradient_descent(var, alpha, delta)" in migration.migrated_source
    assert migration.rewrites == {"apply_gradient_descent": 1}
    assert migration.rewrite_integrity


def test_domain_helper_builds_reuse_gap_spec_with_declared_scope():
    analysis = AnalysisAgent().analyze(DOMAIN_SOURCE, "case.py")
    specs = build_gap_specs(
        DOMAIN_SOURCE,
        analysis,
        target_soc="ascend910b",
        cann_version="9.0.0",
    )

    assert len(specs) == 1
    spec = specs[0]
    assert spec.api_name == "apply_gradient_descent"
    assert spec.qualified_api == "operator.apply_gradient_descent"
    assert spec.asnumpy_api == "asnumpy.apply_gradient_descent"
    assert spec.route == "cann_existing_reuse_candidate"
    assert spec.signature_variants == ({
        "positional_args": 3,
        "keyword_args": [],
    },)
    assert spec.dtype_rules == ("float32",)
    assert spec.shape_rules == ("rank>=1", "contiguous")


def test_operator_spec_target_drift_fails_closed(tmp_path):
    source_path = tmp_path / "case.py"
    source_path.write_text(DOMAIN_SOURCE, encoding="utf-8")

    with pytest.raises(GapCompletionError, match="does not match requested"):
        GapCompletionPipeline().run(
            str(source_path),
            str(tmp_path / "out"),
            target_soc="ascend910b",
            cann_version="8.0.0",
        )


def test_non_literal_or_wrong_signature_operator_spec_is_blocked():
    dynamic = DOMAIN_SOURCE.replace(
        'source="https://gitcode.com/cann/ops-nn.git@0123456789abcdef0123456789abcdef01234567:experimental/optim/apply_gradient_descent",',
        'source="prefix" + "suffix",',
    )
    wrong_signature = DOMAIN_SOURCE.replace(
        "return gradient_oracle(var, alpha, delta)",
        "return gradient_oracle(var, alpha)",
    )

    dynamic_result = AnalysisAgent().analyze(dynamic)
    signature_result = AnalysisAgent().analyze(wrong_signature)

    assert not dynamic_result.migratable
    assert dynamic_result.unsupported == ["operator_spec.gradient_oracle"]
    assert "literals" in dynamic_result.op_uses[0].reason
    assert not signature_result.migratable
    assert signature_result.unsupported == ["apply_gradient_descent"]
    assert "variants" in signature_result.op_uses[0].reason


def test_poisson_step_migrates_and_verifies_on_shim(tmp_path):
    source_path = Path(__file__).parents[1] / "examples" / "poisson_ritz_step.py"
    source = source_path.read_text(encoding="utf-8")
    analysis = AnalysisAgent().analyze(source, str(source_path))

    domain_counts = {}
    for op in analysis.op_uses:
        if op.kind == "domain_operator":
            domain_counts[op.name] = domain_counts.get(op.name, 0) + 1
    assert analysis.migratable
    assert domain_counts == {
        "apply_gradient_descent": 4,
        "smooth_l1_loss": 2,
    }

    result = Orchestrator(backend=ShimBackend()).run(
        str(source_path),
        out_dir=str(tmp_path / "migration"),
    )

    assert result["status"] == "pass"
    assert result["rewrite_integrity"]
    assert result["report"].numerically_consistent
    assert result["rewrites"] == {
        "add": 1,
        "apply_gradient_descent": 4,
        "clip": 1,
        "multiply": 1,
        "smooth_l1_loss": 2,
    }


def test_poisson_specs_select_direct_local_generation_without_ops_nn_dependency():
    source_path = Path(__file__).parents[1] / "examples" / "poisson_ritz_step.py"
    source = source_path.read_text(encoding="utf-8")
    specs = build_gap_specs(
        source,
        AnalysisAgent().analyze(source, str(source_path)),
        target_soc="ascend910b",
        cann_version="9.0.0",
    )

    assert {spec.qualified_api for spec in specs} == {
        "operator.apply_gradient_descent",
        "operator.smooth_l1_loss",
    }
    for spec in specs:
        assert spec.route == "ascendc_registry_candidate"
        assert spec.route_decisions[1]["route"] == "direct_generation"
        assert spec.route_decisions[1]["status"] == "custom_generation_candidate"
        assert spec.route_decisions[1]["reason"].startswith(
            "numpilot://local-llm-generation/"
        )
        semantic_contract = extract_semantic_contract(source, spec)
        request = render_generation_request(
            spec,
            semantic_contract=semantic_contract,
        )
        assert "selected direct local generation" in request
        assert "Do not fetch, clone" in request
        assert "ops-nn" not in request
        assert "NumPy semantic oracle (authoritative)" in request
    assert "return np.subtract(var, np.multiply(alpha, delta))" in next(
        render_generation_request(
            spec,
            semantic_contract=extract_semantic_contract(source, spec),
        )
        for spec in specs
        if spec.api_name == "apply_gradient_descent"
    )


def test_poisson_full_iteration_scans_only_two_direct_local_gaps_and_runs_on_shim(
    tmp_path,
):
    source_path = (
        Path(__file__).parents[1] / "examples" / "poisson_ritz_iteration.py"
    )
    source = source_path.read_text(encoding="utf-8")
    analysis = AnalysisAgent().analyze(source, str(source_path))

    assert analysis.migratable
    domain_counts = {}
    for op in analysis.op_uses:
        if op.kind == "domain_operator":
            domain_counts[op.name] = domain_counts.get(op.name, 0) + 1
    assert domain_counts == {
        "apply_gradient_descent": 4,
        "smooth_l1_loss": 2,
    }

    specs = build_gap_specs(
        source,
        analysis,
        target_soc="ascend910b",
        cann_version="9.0.0",
    )
    assert {spec.qualified_api for spec in specs} == {
        "operator.apply_gradient_descent",
        "operator.smooth_l1_loss",
    }
    assert all(spec.route == "ascendc_registry_candidate" for spec in specs)
    assert all(
        spec.route_decisions[1]["route"] == "direct_generation"
        for spec in specs
    )

    result = Orchestrator(backend=ShimBackend()).run(
        str(source_path),
        out_dir=str(tmp_path / "migration"),
    )

    assert result["status"] == "pass"
    assert result["rewrite_integrity"]
    assert result["report"].numerically_consistent
    assert result["rewrites"]["outer"] == 14
    assert result["rewrites"]["matmul"] == 10
    assert result["rewrites"]["smooth_l1_loss"] == 2
    assert result["rewrites"]["apply_gradient_descent"] == 4
