import hashlib
import json
import types
from dataclasses import replace

import numpy as np
import pytest

from numpilot.capabilities import (
    CapabilityManifest,
    ExecutionProvenance,
    OperatorCapability,
    OperatorSignature,
)
from numpilot.gaps.models import GapSpec
from numpilot.gaps.pipeline import (
    GapCompletionError,
    GapCompletionPipeline,
    load_script_gap_request,
)
from numpilot.script_migration import (
    ScriptMigrationError,
    audit_glm_script,
    build_gap_request,
    classify_script,
    detect_source_mode,
    generate_script_with_glm,
    scan_numpy_script,
    write_script_artifacts,
)


SIMPLE_SOURCE = """\
import numpy as np

x = np.full((2,), 1.0, dtype=np.float32)
y = np.exp(x)
print(y)
"""


ARRAY_DIVISION_SOURCE = """\
import numpy as np

x = np.full((2, 3), 1.0, dtype=np.float32)
w = np.full((3, 4), 2.0, dtype=np.float32)
logits = np.matmul(x, w)
exp_logits = np.exp(logits)
probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
row_sums = np.sum(probs, axis=1)
print(f"logits={logits}")
print(f"probs={probs}")
print(f"row_sums={row_sums}")
"""


ARRAY_DIVISION_MIGRATED = """\
import asnumpy as ap

x = ap.full((2, 3), 1.0, dtype=ap.float32)
w = ap.full((3, 4), 2.0, dtype=ap.float32)
logits = ap.matmul(x, w)
exp_logits = ap.exp(logits)
probs = ap.divide(exp_logits, ap.sum(exp_logits, axis=1, keepdims=True))
row_sums = ap.sum(probs, axis=1)
print(f"logits={logits.to_numpy()}")
print(f"probs={probs.to_numpy()}")
print(f"row_sums={row_sums.to_numpy()}")
"""


class _Backend:
    name = "npu"

    def __init__(self, capabilities, module=None):
        self.capabilities = dict(capabilities)
        self.module = module or types.SimpleNamespace(
            full=lambda *args, **kwargs: None,
            exp=lambda *args, **kwargs: None,
            float32=np.float32,
        )

    def get_module(self):
        return self.module

    def capability_manifest(self, required_ops=None):
        required = tuple(required_ops or ())
        return CapabilityManifest(
            backend=self.name,
            backend_version="test-wheel",
            audit_id="test-audit-v1",
            environment={},
            operators={
                name: self.capabilities.get(
                    name,
                    OperatorCapability(
                        name=name,
                        available=False,
                        evidence="missing in fake runtime",
                    ),
                )
                for name in required
            },
        )


def _cap(name, provenance, *, available=True, constraints=()):
    return OperatorCapability(
        name=name,
        available=available,
        provenance=provenance,
        evidence="test evidence for " + name,
        constraints=tuple(constraints),
    )


def _receipt(content, model="glm-5.3"):
    return {
        "schema_version": 1,
        "utc_start": "2026-08-25T12:00:00Z",
        "utc_end": "2026-08-25T12:00:01Z",
        "http_status": 200,
        "requested_model": model,
        "response_model": model,
        "response_id": "response-script-test",
        "response_created": 1787659200,
        "request_body_sha256": "1" * 64,
        "response_body_sha256": "2" * 64,
        "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "request_id": "request-script-test",
        "usage": {"total_tokens": 101},
        "tls_verification": "enabled",
    }


def _strict_backend():
    return _Backend({
        "full": _cap("full", ExecutionProvenance.NATIVE_NPU),
        "exp": _cap("exp", ExecutionProvenance.NATIVE_NPU),
    })


def _array_division_backend(divide_provenance=ExecutionProvenance.NATIVE_NPU):
    return _Backend({
        "full": _cap("full", ExecutionProvenance.NATIVE_NPU),
        "matmul": _cap("matmul", ExecutionProvenance.NATIVE_NPU),
        "exp": _cap("exp", ExecutionProvenance.NATIVE_NPU),
        "sum": _cap("sum", ExecutionProvenance.NATIVE_NPU),
        "divide": _cap("divide", divide_provenance),
    })


def test_plain_script_scan_does_not_require_forward_or_build_inputs():
    scan = scan_numpy_script(SIMPLE_SOURCE, "case.py")
    assert scan.ok
    assert detect_source_mode(SIMPLE_SOURCE) == "script"
    assert [(item.api_path, item.kind) for item in scan.uses] == [
        ("full", "call"),
        ("float32", "reference"),
        ("exp", "call"),
    ]
    assert scan.rewrite_count == 3


def test_array_division_is_a_source_bound_synthetic_divide_callsite():
    scan = scan_numpy_script(ARRAY_DIVISION_SOURCE, "array-division.py")

    assert scan.ok, scan.issues
    assert scan.synthetic_divide_count == 1
    assert scan.output_boundary_count == 3
    synthetic = [item for item in scan.call_uses if item.synthetic]
    assert [
        (
            item.qualified_api,
            item.positional_args,
            item.keyword_args,
        )
        for item in synthetic
    ] == [("numpy.divide", 2, ())]

    analysis = classify_script(scan, _array_division_backend())
    decision = next(
        item for item in analysis.decisions if item.api_path == "divide"
    )
    assert decision.status == "strict_npu"
    assert decision.available is True
    assert [item.synthetic for item in decision.callsites] == [True]
    assert analysis.can_migrate_directly


def test_array_division_uses_the_same_fallback_consent_state_as_explicit_api():
    analysis = classify_script(
        scan_numpy_script(ARRAY_DIVISION_SOURCE),
        _array_division_backend(ExecutionProvenance.CPU_FALLBACK),
    )

    assert [item.api_path for item in analysis.fallbacks] == ["divide"]
    assert not analysis.can_migrate_directly


def test_array_division_requires_a_real_divide_capability_before_glm():
    backend = _array_division_backend()
    backend.capabilities.pop("divide")
    analysis = classify_script(
        scan_numpy_script(ARRAY_DIVISION_SOURCE), backend
    )

    decision = next(
        item for item in analysis.unsupported if item.api_path == "divide"
    )
    assert decision.available is False
    with pytest.raises(ScriptMigrationError, match="unsupported capability"):
        generate_script_with_glm(
            analysis,
            expected_model="glm-5.3",
            expected_provider="glm",
            chat=lambda _s, _u: pytest.fail("GLM must not be called"),
        )


def test_explicit_divide_result_propagates_to_a_proved_print_boundary():
    source = """\
import numpy as np
base = np.full((2, 2), 1.0, dtype=np.float32)
probs = np.divide(base, np.sum(base, axis=1, keepdims=True))
checksum = np.sum(probs, axis=0, keepdims=True)
print(f"checksum={checksum}")
"""
    migrated = """\
import asnumpy as ap
base = ap.full((2, 2), 1.0, dtype=ap.float32)
probs = ap.divide(base, ap.sum(base, axis=1, keepdims=True))
checksum = ap.sum(probs, axis=0, keepdims=True)
print(f"checksum={checksum.to_numpy()}")
"""

    scan = scan_numpy_script(source)
    analysis = classify_script(scan, _array_division_backend())

    assert scan.ok, scan.issues
    assert scan.synthetic_divide_count == 0
    assert scan.output_boundary_count == 1
    assert analysis.can_migrate_directly
    audit_glm_script(source, migrated, analysis)


def test_existing_contract_is_still_detected():
    source = """\
import numpy as np
def build_inputs():
    return (np.ones((1,), dtype=np.float32),)
def forward(x):
    return np.exp(x)
"""
    assert detect_source_mode(source) == "contract"


def test_source_mode_wraps_syntax_errors_in_script_migration_error():
    with pytest.raises(
        ScriptMigrationError, match="invalid Python syntax at line 2"
    ):
        detect_source_mode("import numpy as np\nif\n")


@pytest.mark.parametrize(
    "source, expected",
    [
        (
            "import numpy as np\nimport scipy\nx = np.exp(1)\n",
            "only one top-level `import numpy as np` is allowed",
        ),
        (
            "from numpy import exp\nx = exp(1)\n",
            "non-NumPy/from import",
        ),
        (
            "import numpy as np\nnp = object()\n",
            "NumPy alias `np` is rebound",
        ),
        (
            "import numpy as np\nap = object()\nx = np.exp(1)\n",
            "reserved AsNumpy alias `ap`",
        ),
        (
            "import numpy as np\ndef f(module=np): return module\n",
            "used dynamically",
        ),
    ],
)
def test_scan_blocks_inputs_outside_the_simple_contract(source, expected):
    scan = scan_numpy_script(source)
    assert not scan.ok
    assert any(expected in issue for issue in scan.issues)


def test_classification_uses_three_honest_states():
    scan = scan_numpy_script(SIMPLE_SOURCE)
    backend = _Backend({
        "full": _cap("full", ExecutionProvenance.NATIVE_NPU),
        "exp": _cap("exp", ExecutionProvenance.CPU_FALLBACK),
    })
    analysis = classify_script(scan, backend)
    assert [item.api_path for item in analysis.strict] == ["float32", "full"]
    assert [item.api_path for item in analysis.fallbacks] == ["exp"]
    assert not analysis.unsupported
    assert not analysis.can_migrate_directly


def test_unknown_callable_requires_capability_audit_not_silent_fallback():
    scan = scan_numpy_script(SIMPLE_SOURCE)
    backend = _Backend({
        "full": _cap("full", ExecutionProvenance.NATIVE_NPU),
        "exp": _cap("exp", ExecutionProvenance.UNKNOWN),
    })
    analysis = classify_script(scan, backend)
    decision = next(item for item in analysis.decisions if item.api_path == "exp")
    assert decision.status == "capability_audit_required"
    assert decision.available is True
    assert decision.generatable is False
    assert analysis.audit_required == (decision,)


def test_shim_fallback_is_not_eligible_for_npu_replacement_generation():
    analysis = classify_script(
        scan_numpy_script(SIMPLE_SOURCE),
        _Backend({
            "full": _cap("full", ExecutionProvenance.SHIM_CPU),
            "exp": _cap("exp", ExecutionProvenance.SHIM_CPU),
        }),
    )

    assert analysis.fallbacks
    assert all(not item.generatable for item in analysis.fallbacks)
    request = build_gap_request(
        analysis,
        analysis.fallbacks,
        target_soc="ascend910b",
        cann_version="9.0.0",
    )
    assert request["status"] == "gap_generation_unavailable"
    assert request["gap_specs"] == []


def test_scoped_capability_requires_more_evidence_in_script_mode():
    scan = scan_numpy_script(SIMPLE_SOURCE)
    backend = _Backend({
        "full": _cap("full", ExecutionProvenance.NATIVE_NPU),
        "exp": _cap(
            "exp", ExecutionProvenance.NATIVE_NPU, constraints=("dtype=float32",)
        ),
    })
    analysis = classify_script(scan, backend)
    decision = next(item for item in analysis.decisions if item.api_path == "exp")
    assert decision.status == "capability_audit_required"


def test_qualified_scoped_npu_signature_can_reenter_direct_migration_lane():
    scan = scan_numpy_script(SIMPLE_SOURCE)
    backend = _Backend({
        "full": _cap("full", ExecutionProvenance.NATIVE_NPU),
        "exp": OperatorCapability(
            name="exp",
            available=True,
            provenance=ExecutionProvenance.CUSTOM_NPU,
            evidence="six gates passed",
            constraints=(
                "dtype=float32",
                "rank>=1",
                "rank<=4",
                "numel>=1",
                "contiguous=true",
            ),
            signature_variants=(OperatorSignature(positional_args=1),),
        ),
    })

    analysis = classify_script(scan, backend)

    scoped = next(item for item in analysis.scoped if item.api_path == "exp")
    assert scoped.provenance == "custom_npu"
    assert not analysis.unsupported
    assert analysis.can_migrate_directly


def test_nested_api_is_not_promoted_by_hasattr_alone():
    source = "import numpy as np\nx = np.linalg.det(np.eye(2, dtype=np.float32))\n"
    scan = scan_numpy_script(source)
    backend = _Backend(
        {"eye": _cap("eye", ExecutionProvenance.NATIVE_NPU)},
        module=types.SimpleNamespace(
            linalg=types.SimpleNamespace(det=lambda value: value),
            eye=lambda *args, **kwargs: None,
            float32=np.float32,
        ),
    )
    analysis = classify_script(scan, backend)
    nested = next(item for item in analysis.decisions if item.api_path == "linalg.det")
    assert nested.available is True
    assert nested.status == "capability_audit_required"
    assert nested.generatable is False


def test_glm_generates_exact_namespace_rewrite_and_receipt_is_bound():
    analysis = classify_script(scan_numpy_script(SIMPLE_SOURCE), _strict_backend())
    migrated = """\
import asnumpy as ap

x = ap.full((2,), 1.0, dtype=ap.float32)
y = ap.exp(x)
print(y)
"""
    raw_content = "```python\n" + migrated + "```"

    def chat(system, user):
        assert "exactly these AST-level changes" in system
        assert analysis.scan.source_sha256 in user
        return raw_content, _receipt(raw_content)

    artifact = generate_script_with_glm(
        analysis,
        expected_model="glm-5.3",
        expected_provider="glm",
        chat=chat,
    )
    assert artifact.migrated_source == migrated
    assert artifact.model == "glm-5.3"
    assert artifact.response_model == "glm-5.3"
    assert artifact.migrated_sha256 == hashlib.sha256(migrated.encode()).hexdigest()


def test_glm_alone_emits_proved_divide_and_print_output_boundary_rewrites():
    analysis = classify_script(
        scan_numpy_script(ARRAY_DIVISION_SOURCE),
        _array_division_backend(),
    )
    raw_content = "```python\n" + ARRAY_DIVISION_MIGRATED + "```"

    def chat(system, user):
        assert "ap.divide(left, right)" in system
        assert "value.to_numpy()" in system
        assert "Do not rewrite any other Python operator" in system
        assert "Authorized synthetic numpy.divide callsites" in user
        assert "Authorized print output-boundary values" in user
        return raw_content, _receipt(raw_content)

    artifact = generate_script_with_glm(
        analysis,
        expected_model="glm-5.3",
        expected_provider="glm",
        chat=chat,
    )

    assert artifact.migrated_source == ARRAY_DIVISION_MIGRATED
    assert " / " not in artifact.migrated_source
    assert artifact.migrated_source.count(".to_numpy()") == 3


@pytest.mark.parametrize(
    "migrated",
    [
        ARRAY_DIVISION_MIGRATED.replace(
            "ap.divide(exp_logits, ap.sum(exp_logits, axis=1, keepdims=True))",
            "exp_logits / ap.sum(exp_logits, axis=1, keepdims=True)",
        ),
        ARRAY_DIVISION_MIGRATED.replace("logits.to_numpy()", "logits"),
        ARRAY_DIVISION_MIGRATED.replace(
            "exp_logits = ap.exp(logits)",
            "exp_logits = ap.exp(logits).to_numpy()",
        ),
    ],
)
def test_divide_and_output_boundary_audit_rejects_missing_or_extra_rewrites(
    migrated,
):
    analysis = classify_script(
        scan_numpy_script(ARRAY_DIVISION_SOURCE),
        _array_division_backend(),
    )

    with pytest.raises(ScriptMigrationError, match="outside the exact"):
        audit_glm_script(ARRAY_DIVISION_SOURCE, migrated, analysis)


def test_synthetic_expression_audit_cannot_bypass_capability_analysis():
    with pytest.raises(ScriptMigrationError, match="source-bound analysis"):
        audit_glm_script(
            ARRAY_DIVISION_SOURCE,
            ARRAY_DIVISION_MIGRATED,
        )


@pytest.mark.parametrize(
    "tampered",
    [
        "import asnumpy as ap\nx = ap.full((3,), 1.0, dtype=ap.float32)\ny = ap.exp(x)\nprint(y)\n",
        "import asnumpy as ap\nx = ap.full((2,), 1.0, dtype=ap.float32)\ny = ap.log(x)\nprint(y)\n",
        "import asnumpy as ap\nx = ap.full((2,), 1.0, dtype=ap.float32)\ny = ap.exp(x).to_numpy()\nprint(y)\n",
        "import asnumpy as ap\nimport numpy as np\nx = ap.full((2,), 1.0, dtype=ap.float32)\ny = ap.exp(x)\nprint(y)\n",
    ],
)
def test_glm_output_fails_closed_on_any_semantic_change(tampered):
    with pytest.raises(ScriptMigrationError, match="outside the exact"):
        audit_glm_script(SIMPLE_SOURCE, tampered)


def test_receipt_content_hash_must_match_model_response():
    analysis = classify_script(scan_numpy_script(SIMPLE_SOURCE), _strict_backend())
    migrated = SIMPLE_SOURCE.replace("import numpy as np", "import asnumpy as ap").replace(
        "np.", "ap."
    )

    raw = "```python\n" + migrated + "```"

    def chat(_system, _user):
        receipt = _receipt(raw)
        receipt["content_sha256"] = "0" * 64
        return raw, receipt

    with pytest.raises(ScriptMigrationError, match="content hash mismatch"):
        generate_script_with_glm(
            analysis,
            expected_model="glm-5.3",
            expected_provider="glm",
            chat=chat,
        )


@pytest.mark.parametrize(
    "content",
    [
        "import asnumpy as ap\n",
        "explanation\n```python\nimport asnumpy as ap\n```",
        "```python\nimport asnumpy as ap\n```\n```python\npass\n```",
    ],
)
def test_glm_response_requires_one_bare_python_envelope(content):
    analysis = classify_script(scan_numpy_script(SIMPLE_SOURCE), _strict_backend())

    with pytest.raises(ScriptMigrationError, match="exactly one python code block"):
        generate_script_with_glm(
            analysis,
            expected_model="glm-5.3",
            expected_provider="glm",
            chat=lambda _s, _u: (content, _receipt(content)),
        )


def test_incomplete_glm_receipt_is_rejected():
    analysis = classify_script(scan_numpy_script(SIMPLE_SOURCE), _strict_backend())
    migrated = SIMPLE_SOURCE.replace(
        "import numpy as np", "import asnumpy as ap"
    ).replace("np.", "ap.")
    raw = "```python\n" + migrated + "```"

    with pytest.raises(ScriptMigrationError, match="incomplete receipt"):
        generate_script_with_glm(
            analysis,
            expected_model="glm-5.3",
            expected_provider="glm",
            chat=lambda _s, _u: (raw, {"content_sha256": "0" * 64}),
        )


@pytest.mark.parametrize(
    "response_model, message",
    [
        (None, "no valid GLM response_model"),
        ("other-model", "no valid GLM response_model"),
        ("glm-5.2", "does not match requested/configured model"),
    ],
)
def test_glm_receipt_requires_matching_actual_response_model(
    response_model, message
):
    analysis = classify_script(scan_numpy_script(SIMPLE_SOURCE), _strict_backend())
    migrated = SIMPLE_SOURCE.replace(
        "import numpy as np", "import asnumpy as ap"
    ).replace("np.", "ap.")
    raw = "```python\n" + migrated + "```"
    receipt = _receipt(raw)
    receipt["response_model"] = response_model

    with pytest.raises(ScriptMigrationError, match=message):
        generate_script_with_glm(
            analysis,
            expected_model="glm-5.3",
            expected_provider="glm",
            chat=lambda _s, _u: (raw, receipt),
        )


def test_comment_tampering_is_rejected_even_when_ast_matches():
    source = "import numpy as np\n# keep this license\nx = np.exp(1)\n"
    migrated = "import asnumpy as ap\n# changed license\nx = ap.exp(1)\n"

    with pytest.raises(ScriptMigrationError, match="changed comments"):
        audit_glm_script(source, migrated)


@pytest.mark.parametrize(
    "tail",
    [
        "args = (1,)\nx = np.exp(*args)\n",
        "kwargs = {}\nx = np.exp(**kwargs)\n",
        "match value:\n    case np:\n        pass\n",
    ],
)
def test_dynamic_signatures_and_pattern_shadowing_are_blocked(tail):
    scan = scan_numpy_script("import numpy as np\n" + tail)
    assert not scan.ok


@pytest.mark.parametrize(
    "expression, issue",
    [
        ("y = x / 2", "Python binary operator Div"),
        ("y = x > 0", "Python comparison Gt"),
        ("y = -x", "Python unary operator USub"),
        ("x += x", "Python augmented operator Add"),
    ],
)
def test_python_array_operators_fail_closed_before_glm(expression, issue):
    source = (
        "import numpy as np\n"
        "x = np.ones((2,), dtype=np.float32)\n"
        + expression
        + "\n"
    )

    scan = scan_numpy_script(source)

    assert not scan.ok
    assert any(issue in item for item in scan.issues)


@pytest.mark.parametrize(
    "division",
    [
        "left / 2",
        "left / np.sum(left, axis=1)",
        "left / np.sum(other, axis=1, keepdims=True)",
        "unknown / np.sum(unknown, axis=1, keepdims=True)",
        "np.exp(left) / np.sum(left, axis=1, keepdims=True)",
    ],
)
def test_only_the_narrow_proved_array_division_shape_is_authorized(division):
    source = (
        "import numpy as np\n"
        "base = np.full((2, 2), 1.0, dtype=np.float32)\n"
        "other = np.exp(base)\n"
        "left = np.exp(base)\n"
        "result = %s\n" % division
    )

    scan = scan_numpy_script(source)

    assert not scan.ok
    assert scan.synthetic_divide_count == 0
    assert any("Python binary operator Div" in item for item in scan.issues)


def test_proved_division_inside_an_unfamiliar_scope_remains_blocked():
    source = """\
import numpy as np
def calculate():
    base = np.full((2, 2), 1.0, dtype=np.float32)
    left = np.exp(base)
    return left / np.sum(left, axis=1, keepdims=True)
"""

    scan = scan_numpy_script(source)

    assert not scan.ok
    assert scan.synthetic_divide_count == 0
    assert any("Python binary operator Div" in item for item in scan.issues)


@pytest.mark.parametrize(
    "formatted",
    ["{value!r}", "{value!s}", "{value:.3f}"],
)
def test_print_output_conversion_does_not_expand_to_formatted_values(formatted):
    source = (
        "import numpy as np\n"
        "value = np.full((2,), 1.0, dtype=np.float32)\n"
        "print(f\"%s\")\n" % formatted
    )
    migrated = source.replace(
        "import numpy as np", "import asnumpy as ap"
    ).replace("np.", "ap.")
    analysis = classify_script(
        scan_numpy_script(source),
        _Backend({"full": _cap("full", ExecutionProvenance.NATIVE_NPU)}),
    )

    assert analysis.scan.ok
    assert analysis.scan.output_boundary_count == 0
    audit_glm_script(source, migrated, analysis)
    with pytest.raises(ScriptMigrationError, match="outside the exact"):
        audit_glm_script(
            source,
            migrated.replace("{value", "{value.to_numpy()", 1),
            analysis,
        )


@pytest.mark.parametrize("method", ["astype(np.float32)", "item()"])
def test_ndarray_method_calls_fail_closed_before_glm(method):
    source = (
        "import numpy as np\n"
        "x = np.ones((2,), dtype=np.float32)\n"
        "y = x.%s\n" % method
    )

    scan = scan_numpy_script(source)

    assert not scan.ok
    assert any("attribute/method call" in item for item in scan.issues)


@pytest.mark.parametrize(
    "call, reason",
    [
        (
            "np.eye(3, 4, dtype=np.float32)",
            "位置参数语义不兼容",
        ),
        ("np.amax(x)", "静态整数 axis"),
        ("np.add(x, 1)", "不接受标量"),
        ("np.maximum(x, 0)", "maximum-to-relu"),
        (
            "np.full(shape=(2,), fill_value=1.0, dtype=np.float32)",
            "未验证 NumPy 关键字",
        ),
    ],
)
def test_existing_asnumpy_call_contract_fails_closed_before_glm(call, reason):
    source = (
        "import numpy as np\n"
        "x = np.ones((2,), dtype=np.float32)\n"
        "y = %s\n" % call
    )

    scan = scan_numpy_script(source)

    assert not scan.ok
    assert any(
        "AsNumpy call contract rejected" in item and reason in item
        for item in scan.issues
    )


def test_explicit_numpy_apis_with_shared_call_contract_remain_admissible():
    source = """\
import numpy as np
x = np.ones((2,), dtype=np.float32)
y = np.add(x, x)
z = np.amax(y, axis=0)
"""

    scan = scan_numpy_script(source)

    assert scan.ok, scan.issues


def test_gap_request_and_reports_do_not_claim_generation_or_execution(tmp_path):
    scan = scan_numpy_script(SIMPLE_SOURCE, "case.py")
    analysis = classify_script(scan, _Backend({
        "full": _cap("full", ExecutionProvenance.NATIVE_NPU),
        "exp": _cap("exp", ExecutionProvenance.CPU_FALLBACK),
    }))
    request = build_gap_request(
        analysis,
        analysis.fallbacks,
        target_soc="ascend910b",
        cann_version="9.0.0",
    )
    paths = write_script_artifacts(
        str(tmp_path),
        analysis,
        status="gap_generation_required",
        fallback_accepted=False,
        gap_request=request,
    )
    assert "migrated" not in paths
    report = json.loads((tmp_path / "script-migration-report.json").read_text())
    gap = json.loads((tmp_path / "gap-request.json").read_text())
    assert report["execution_performed"] is False
    assert report["performance_valid"] is False
    assert gap["generated_code"] is False
    assert gap["gaps"][0]["qualified_api"] == "numpy.exp"
    assert len(gap["gap_specs"]) == 1
    spec = GapSpec.from_dict(gap["gap_specs"][0])
    assert spec.qualified_api == "numpy.exp"
    assert any(
        item["status"] == "user_rejected"
        for item in spec.route_decisions
    )


def test_rejected_fallback_request_enters_existing_gap_pipeline(tmp_path):
    source_path = tmp_path / "case.py"
    source_path.write_text(SIMPLE_SOURCE)
    analysis = classify_script(
        scan_numpy_script(SIMPLE_SOURCE, str(source_path.absolute())),
        _Backend({
            "full": _cap("full", ExecutionProvenance.NATIVE_NPU),
            "exp": _cap("exp", ExecutionProvenance.CPU_FALLBACK),
        }),
    )
    request = build_gap_request(
        analysis,
        analysis.fallbacks,
        target_soc="ascend910b",
        cann_version="9.0.0",
    )
    request_dir = tmp_path / "migration"
    write_script_artifacts(
        str(request_dir),
        analysis,
        status="gap_generation_required",
        fallback_accepted=False,
        gap_request=request,
    )

    selection = load_script_gap_request(
        str(request_dir / "gap-request.json"),
        str(source_path),
        target_soc="ascend910b",
        cann_version="9.0.0",
    )
    result = GapCompletionPipeline().run(
        str(source_path),
        str(tmp_path / "completion"),
        target_soc="ascend910b",
        cann_version="9.0.0",
        requested_specs=selection.specs,
        requested_blocked=selection.blocked,
    )

    assert result.status == "prepared"
    assert [item.qualified_api for item in result.specs] == ["numpy.exp"]
    assert not result.blocked
    workspace = tmp_path / "completion" / "gaps" / result.specs[0].gap_id
    assert (workspace / "gap-spec.json").is_file()
    assert (workspace / "GENERATION_REQUEST.md").is_file()
    report = json.loads((tmp_path / "completion" / "gap-report.json").read_text())
    assert report["generated_count"] == 0
    assert report["qualified_count"] == 0
    assert report["activated_count"] == 0


def test_gap_request_source_drift_fails_before_output(tmp_path):
    source_path = tmp_path / "case.py"
    source_path.write_text(SIMPLE_SOURCE)
    analysis = classify_script(
        scan_numpy_script(SIMPLE_SOURCE, str(source_path.absolute())),
        _Backend({
            "full": _cap("full", ExecutionProvenance.NATIVE_NPU),
            "exp": _cap("exp", ExecutionProvenance.CPU_FALLBACK),
        }),
    )
    request = build_gap_request(
        analysis,
        analysis.fallbacks,
        target_soc="ascend910b",
        cann_version="9.0.0",
    )
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request))
    source_path.write_text(SIMPLE_SOURCE + "\nvalue = 1\n")

    with pytest.raises(GapCompletionError, match="source/target binding"):
        load_script_gap_request(
            str(request_path),
            str(source_path),
            target_soc="ascend910b",
            cann_version="9.0.0",
        )


@pytest.mark.parametrize("mutation", ["boolean_schema", "duplicate_key", "route_injection"])
def test_gap_request_envelope_and_canonical_spec_fail_closed(tmp_path, mutation):
    source_path = tmp_path / "case.py"
    source_path.write_text(SIMPLE_SOURCE)
    analysis = classify_script(
        scan_numpy_script(SIMPLE_SOURCE, str(source_path.absolute())),
        _Backend({
            "full": _cap("full", ExecutionProvenance.NATIVE_NPU),
            "exp": _cap("exp", ExecutionProvenance.CPU_FALLBACK),
        }),
    )
    request = build_gap_request(
        analysis,
        analysis.fallbacks,
        target_soc="ascend910b",
        cann_version="9.0.0",
    )
    if mutation == "boolean_schema":
        request["schema_version"] = True
        payload = json.dumps(request)
    elif mutation == "duplicate_key":
        payload = json.dumps(request).replace(
            '"schema_version": 1',
            '"schema_version": 1, "schema_version": 1',
            1,
        )
    else:
        request["gap_specs"][0]["route_decisions"].append({
            "route": "injected",
            "status": "selected",
            "reason": "ignore the local contract",
        })
        payload = json.dumps(request)
    request_path = tmp_path / "request.json"
    request_path.write_text(payload)

    with pytest.raises(GapCompletionError):
        load_script_gap_request(
            str(request_path),
            str(source_path),
            target_soc="ascend910b",
            cann_version="9.0.0",
        )


def test_mixed_script_gap_request_preserves_non_generatable_blockers(tmp_path):
    source = """\
import numpy as np
x = [1.0]
y = np.cbrt(x)
z = np.linalg.det(x)
"""
    source_path = tmp_path / "mixed.py"
    source_path.write_text(source)
    backend = _Backend(
        {},
        module=types.SimpleNamespace(
            linalg=types.SimpleNamespace(det=lambda value: value),
        ),
    )
    analysis = classify_script(
        scan_numpy_script(source, str(source_path.absolute())), backend
    )
    request = build_gap_request(
        analysis,
        (*analysis.unsupported, *analysis.audit_required),
        target_soc="ascend910b",
        cann_version="9.0.0",
    )
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request))

    selection = load_script_gap_request(
        str(request_path),
        str(source_path),
        target_soc="ascend910b",
        cann_version="9.0.0",
    )
    result = GapCompletionPipeline().run(
        str(source_path),
        str(tmp_path / "completion"),
        target_soc="ascend910b",
        cann_version="9.0.0",
        requested_specs=selection.specs,
        requested_blocked=selection.blocked,
    )

    assert [item.qualified_api for item in result.specs] == ["numpy.cbrt"]
    assert [item["qualified_api"] for item in result.blocked] == [
        "numpy.linalg.det"
    ]
    assert result.status == "prepared_partially_blocked"


def test_writer_reaudits_artifact_before_any_publish(tmp_path):
    analysis = classify_script(scan_numpy_script(SIMPLE_SOURCE), _strict_backend())
    migrated = SIMPLE_SOURCE.replace(
        "import numpy as np", "import asnumpy as ap"
    ).replace("np.", "ap.")
    raw = "```python\n" + migrated + "```"
    artifact = generate_script_with_glm(
        analysis,
        expected_model="glm-5.3",
        expected_provider="glm",
        chat=lambda _s, _u: (raw, _receipt(raw)),
    )

    with pytest.raises(ScriptMigrationError, match="hash is inconsistent"):
        write_script_artifacts(
            str(tmp_path),
            analysis,
            status="migrated_strict_npu",
            fallback_accepted=False,
            artifact=replace(artifact, migrated_sha256="0" * 64),
        )
    assert list(tmp_path.iterdir()) == []


def test_writer_reaudits_divide_and_output_boundary_plan_before_publish(tmp_path):
    analysis = classify_script(
        scan_numpy_script(ARRAY_DIVISION_SOURCE),
        _array_division_backend(),
    )
    raw = "```python\n" + ARRAY_DIVISION_MIGRATED + "```"
    artifact = generate_script_with_glm(
        analysis,
        expected_model="glm-5.3",
        expected_provider="glm",
        chat=lambda _s, _u: (raw, _receipt(raw)),
    )
    tampered_source = ARRAY_DIVISION_MIGRATED.replace(
        "row_sums.to_numpy()", "row_sums"
    )
    tampered_content = "```python\n" + tampered_source + "```"
    tampered_artifact = replace(
        artifact,
        migrated_source=tampered_source,
        migrated_sha256=hashlib.sha256(tampered_source.encode()).hexdigest(),
        response_content=tampered_content,
        receipt=_receipt(tampered_content),
    )

    with pytest.raises(ScriptMigrationError, match="outside the exact"):
        write_script_artifacts(
            str(tmp_path),
            analysis,
            status="migrated_strict_npu",
            fallback_accepted=False,
            artifact=tampered_artifact,
        )
    assert list(tmp_path.iterdir()) == []


def test_writer_rejects_tampered_response_model_evidence_before_publish(tmp_path):
    analysis = classify_script(scan_numpy_script(SIMPLE_SOURCE), _strict_backend())
    migrated = SIMPLE_SOURCE.replace(
        "import numpy as np", "import asnumpy as ap"
    ).replace("np.", "ap.")
    raw = "```python\n" + migrated + "```"
    artifact = generate_script_with_glm(
        analysis,
        expected_model="glm-5.3",
        expected_provider="glm",
        chat=lambda _s, _u: (raw, _receipt(raw)),
    )

    with pytest.raises(
        ScriptMigrationError, match="response_model evidence is inconsistent"
    ):
        write_script_artifacts(
            str(tmp_path),
            analysis,
            status="migrated_strict_npu",
            fallback_accepted=False,
            artifact=replace(artifact, response_model="glm-5.2"),
        )
    assert list(tmp_path.iterdir()) == []


def test_shim_exposes_dtype_metadata_for_whole_file_rewrite():
    from numpilot.backends.shim import ShimBackend

    module = ShimBackend().get_module()
    assert module.float32 is np.float32
    assert module.int64 is np.int64


def _strict_script_artifact(analysis):
    migrated = SIMPLE_SOURCE.replace(
        "import numpy as np", "import asnumpy as ap"
    ).replace("np.", "ap.")
    raw = "```python\n" + migrated + "```"
    return generate_script_with_glm(
        analysis,
        expected_model="glm-5.3",
        expected_provider="glm",
        chat=lambda _s, _u: (raw, _receipt(raw)),
    )


def _directory_bytes(path):
    return {
        item.name: item.read_bytes()
        for item in sorted(path.iterdir(), key=lambda entry: entry.name)
    }


def test_writer_rejects_old_migrated_output_before_report_only_publish(tmp_path):
    out = tmp_path / "one-shot-output"
    analysis = classify_script(scan_numpy_script(SIMPLE_SOURCE), _strict_backend())
    write_script_artifacts(
        str(out),
        analysis,
        status="migrated_strict_npu",
        fallback_accepted=False,
        artifact=_strict_script_artifact(analysis),
    )
    before = _directory_bytes(out)

    with pytest.raises(ScriptMigrationError, match="use a fresh --out"):
        write_script_artifacts(
            str(out),
            analysis,
            status="blocked",
            fallback_accepted=False,
        )

    assert _directory_bytes(out) == before


def test_writer_rejects_old_gap_output_before_migrated_publish(tmp_path):
    out = tmp_path / "one-shot-output"
    fallback_analysis = classify_script(
        scan_numpy_script(SIMPLE_SOURCE),
        _Backend({
            "full": _cap("full", ExecutionProvenance.NATIVE_NPU),
            "exp": _cap("exp", ExecutionProvenance.CPU_FALLBACK),
        }),
    )
    request = build_gap_request(
        fallback_analysis,
        fallback_analysis.fallbacks,
        target_soc="ascend910b",
        cann_version="9.0.0",
    )
    write_script_artifacts(
        str(out),
        fallback_analysis,
        status="gap_generation_required",
        fallback_accepted=False,
        gap_request=request,
    )
    before = _directory_bytes(out)
    strict_analysis = classify_script(
        scan_numpy_script(SIMPLE_SOURCE), _strict_backend()
    )

    with pytest.raises(ScriptMigrationError, match="use a fresh --out"):
        write_script_artifacts(
            str(out),
            strict_analysis,
            status="migrated_strict_npu",
            fallback_accepted=False,
            artifact=_strict_script_artifact(strict_analysis),
        )

    assert _directory_bytes(out) == before


@pytest.mark.parametrize(
    "managed_name",
    ["migrated.py", "gap-request.json", "script-migration-report.json"],
)
def test_writer_rejects_source_conflict_with_every_managed_target(
    tmp_path, managed_name
):
    out = tmp_path / "source-output-conflict"
    out.mkdir()
    source_path = out / managed_name
    source_path.write_text(SIMPLE_SOURCE)
    analysis = classify_script(
        scan_numpy_script(SIMPLE_SOURCE, str(source_path.absolute())),
        _strict_backend(),
    )
    before = _directory_bytes(out)

    with pytest.raises(
        ScriptMigrationError, match="source conflicts with managed output"
    ):
        write_script_artifacts(
            str(out),
            analysis,
            status="blocked",
            fallback_accepted=False,
        )

    assert _directory_bytes(out) == before
