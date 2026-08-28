"""Fail-closed official-binary candidate to AsNumPy adapter contracts."""

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path

import pytest

from numpilot.gaps.models import GapSpec
from numpilot.gaps.official_adapter import (
    OFFICIAL_ACLNN_EXISTING_MODULE_STRATEGY,
    OFFICIAL_CANDIDATE_MAPPING_REF,
    OFFICIAL_REQUIRED_SYMBOLS_REF,
    OFFICIAL_SELECTED_HEADER_REF,
    OFFICIAL_SELECTED_LIBRARY_REF,
    OFFICIAL_SEMANTIC_REVIEW_REF,
    OfficialAdapterInputError,
    bind_official_adapter_context,
    load_official_candidate_adapter_input,
    prepare_official_candidate_adapter_input,
)
from numpilot.asnumpy_adapter_context import (
    AsnumpyAdapterContextPlan,
    EXISTING_MODULE_MODE,
    OFFICIAL_ACLNN_ARCHITECTURE,
)
from types import MappingProxyType
from numpilot.gaps.official_resolution import (
    OfficialGapResolution,
    gap_spec_sha256,
)
from numpilot.gaps.recovery import (
    EvidenceRef,
    LicenseBinding,
    OperatorCandidateOrigin,
    create_operator_candidate,
)


REPO = Path(__file__).resolve().parents[1]
EVIDENCE = REPO / "tests" / "fixtures" / "ops-nn-recovery"
GAP_REQUEST = EVIDENCE / "pre-integration-gap-v2" / "gap-request.json"
OFFICIAL_REPORT = EVIDENCE / "official-resolution-linkage-v2.json"


def _canonical_sha256(value) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _fixture_values():
    request = json.loads(GAP_REQUEST.read_text(encoding="utf-8"))
    report = json.loads(OFFICIAL_REPORT.read_text(encoding="utf-8"))
    spec = GapSpec.from_dict(request["gap_specs"][0])
    resolution = OfficialGapResolution.from_dict(report["decisions"][0])
    return spec, resolution


def _ref(name: str, digest: str) -> EvidenceRef:
    return EvidenceRef(name, "/evidence/%s" % name, digest)


def _candidate(
    spec: GapSpec,
    resolution: OfficialGapResolution,
    *,
    origin: OperatorCandidateOrigin = OperatorCandidateOrigin.OFFICIAL_BINARY,
    ref_overrides=None,
    omit=(),
):
    installed = resolution.installed_cann
    mapping = installed["candidate"]
    digests = {
        OFFICIAL_CANDIDATE_MAPPING_REF: installed[
            "candidate_mapping_sha256"
        ],
        OFFICIAL_SELECTED_HEADER_REF: installed["selected_header"]["sha256"],
        OFFICIAL_SELECTED_LIBRARY_REF: installed["selected_library"]["sha256"],
        OFFICIAL_REQUIRED_SYMBOLS_REF: _canonical_sha256({
            "required_symbols": installed["required_symbols"],
        }),
        OFFICIAL_SEMANTIC_REVIEW_REF: _canonical_sha256({
            "semantic_review": mapping["semantic_review"],
        }),
    }
    digests.update(ref_overrides or {})
    for name in omit:
        digests.pop(name, None)
    provenance_names = {
        OFFICIAL_CANDIDATE_MAPPING_REF,
        OFFICIAL_SEMANTIC_REVIEW_REF,
    }
    provenance = tuple(
        _ref(name, digest)
        for name, digest in digests.items()
        if name in provenance_names
    )
    content = tuple(
        _ref(name, digest)
        for name, digest in digests.items()
        if name not in provenance_names
    )
    return create_operator_candidate(
        origin_kind=origin,
        gap_id=spec.gap_id,
        qualified_api=spec.qualified_api,
        gap_spec_ref=EvidenceRef(
            "gap_spec", "/evidence/gap-spec.json", gap_spec_sha256(spec)
        ),
        resolution_decision_ref=EvidenceRef(
            "resolution_decision",
            "/evidence/official-resolution.json#%s" % spec.gap_id,
            _canonical_sha256(resolution.to_dict()),
        ),
        target_soc=spec.target_soc,
        cann_version=spec.cann_version,
        license=LicenseBinding(
            identifier="LicenseRef-CANN-Toolkit",
            accepted=False,
            ref=_ref("license", hashlib.sha256(b"license").hexdigest()),
        ),
        provenance_refs=provenance,
        candidate_refs=content,
    )


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_official_binary_contract_is_deterministic_and_explicitly_not_model_ready():
    spec, resolution = _fixture_values()
    candidate = _candidate(spec, resolution)

    first = prepare_official_candidate_adapter_input(
        candidate, spec, resolution
    )
    second = prepare_official_candidate_adapter_input(
        candidate, spec, resolution
    )

    assert first.to_dict() == second.to_dict()
    manifest = first.operator_manifest
    contract = json.loads(first.adapter_contract)
    installed = resolution.installed_cann
    assert manifest["artifact_type"] == "official_binary_adapter_input"
    assert manifest["artifact_id"] == candidate.candidate_id
    assert manifest["artifact_id_semantics"] == (
        "operator_candidate_id_compatibility_alias"
    )
    assert manifest["origin_kind"] == "official_binary"
    assert manifest["strategy"] == OFFICIAL_ACLNN_EXISTING_MODULE_STRATEGY
    assert manifest["model_invocation_allowed"] is False
    assert manifest["new_kernel_generation_allowed"] is False
    assert manifest["custom_opp_generation_allowed"] is False
    assert manifest["activation_allowed"] is False
    assert "files" not in manifest
    assert "generation" not in manifest
    assert contract["strategy"] == OFFICIAL_ACLNN_EXISTING_MODULE_STRATEGY
    assert contract["source_candidate"]["origin_kind"] == "official_binary"
    assert contract["official_binary"]["selected_header"] == (
        installed["selected_header"]
    )
    assert contract["official_binary"]["selected_library"] == (
        installed["selected_library"]
    )
    assert contract["official_binary"]["required_symbols"] == (
        installed["required_symbols"]
    )
    assert contract["official_binary"]["semantic_review"] == (
        installed["candidate"]["semantic_review"]
    )
    assert contract["permissions"] == {
        "activation_allowed": False,
        "generate_custom_opp": False,
        "generate_new_kernel": False,
        "reuse_installed_aclnn": True,
    }
    assert manifest["adapter_contract_sha256"] == (
        first.adapter_contract_sha256
    )
    with pytest.raises(OfficialAdapterInputError, match="custom-op planners"):
        first.require_model_ready()


def test_reviewed_existing_module_context_opens_only_the_model_gate(tmp_path):
    spec, resolution = _fixture_values()
    candidate = _candidate(spec, resolution)
    prepared = prepare_official_candidate_adapter_input(candidate, spec, resolution)
    cmake_path = "bindings/python/CMakeLists.txt"
    cmake = (
        "target_link_libraries(_core\n"
        "    PUBLIC\n"
        "    ascendcl\n"
        "    runtime\n"
        "    nnopbase\n"
        "    opapi\n"
        "    Python::NumPy\n"
        ")\n"
        "target_link_libraries(_core\n"
        "    PUBLIC\n"
        "    array\n"
        "    linalg\n"
        ")\n"
    )
    context = MappingProxyType({
        "csrc/linalg/product.cpp": "fixture\n",
        cmake_path: cmake,
    })
    hashes = MappingProxyType({
        path: hashlib.sha256(content.encode("utf-8")).hexdigest()
        for path, content in context.items()
    })
    dependencies = MappingProxyType({
        "csrc/linalg/product.cpp": (("reference", cmake_path),),
    })
    selected_library = prepared.selected_library
    library_binding = MappingProxyType({
        "schema_version": 1,
        "proof_kind": "reviewed_exact_cmake_target_link_item",
        "cmake_path": cmake_path,
        "cmake_sha256": hashes[cmake_path],
        "target": "_core",
        "link_item": "opapi",
        "selected_library_relative_path": selected_library["relative_path"],
        "selected_library_basename": "libopapi.so",
        "selected_library_sha256": selected_library["sha256"],
    })
    identity = {
        "schema_version": 3,
        "architecture": OFFICIAL_ACLNN_ARCHITECTURE,
        "mode": EXISTING_MODULE_MODE,
        "context_sha256": dict(hashes),
        "expected_overlay_paths": ["csrc/linalg/product.cpp"],
        "prompt_dependencies": {
            "csrc/linalg/product.cpp": [
                {"source": "reference", "path": cmake_path},
            ],
        },
        "official_library_binding": dict(library_binding),
    }
    plan = AsnumpyAdapterContextPlan(
        architecture=OFFICIAL_ACLNN_ARCHITECTURE,
        mode=EXISTING_MODULE_MODE,
        reference_root=tmp_path / "asnumpy-reference",
        context=context,
        context_sha256=hashes,
        expected_overlay_paths=("csrc/linalg/product.cpp",),
        total_bytes=sum(
            len(content.encode("utf-8")) for content in context.values()
        ),
        plan_sha256=_canonical_sha256(identity),
        schema_version=3,
        prompt_dependencies=dependencies,
        official_library_binding=library_binding,
    )

    ready = bind_official_adapter_context(prepared, plan, api_name="matmul")

    ready.require_model_ready()
    assert ready.model_invocation_allowed is True
    assert ready.operator_manifest["context_plan_status"] == "prepared"
    assert ready.operator_manifest["activation_allowed"] is False
    assert ready.operator_manifest["asnumpy_adapter_context_plan"] == plan.manifest()


def test_custom_opp_context_cannot_open_official_model_gate():
    spec, resolution = _fixture_values()
    prepared = prepare_official_candidate_adapter_input(
        _candidate(spec, resolution), spec, resolution
    )
    plan = AsnumpyAdapterContextPlan(
        architecture="legacy_manual",
        mode="manual",
        reference_root=Path("/tmp/asnumpy-reference"),
        context=MappingProxyType({"src/asnumpy/custom_ops.py": "fixture\n"}),
        context_sha256=MappingProxyType({
            "src/asnumpy/custom_ops.py": hashlib.sha256(b"fixture\n").hexdigest()
        }),
        expected_overlay_paths=("src/asnumpy/custom_ops.py",),
        total_bytes=len(b"fixture\n"),
        plan_sha256="f" * 64,
    )

    with pytest.raises(OfficialAdapterInputError, match="existing-module"):
        bind_official_adapter_context(prepared, plan, api_name="matmul")


@pytest.mark.parametrize(
    "origin",
    (
        OperatorCandidateOrigin.OFFICIAL_SOURCE,
        OperatorCandidateOrigin.REMOTE_LLM,
    ),
)
def test_source_and_remote_candidates_cannot_use_official_binary_adapter(origin):
    spec, resolution = _fixture_values()
    candidate = _candidate(spec, resolution, origin=origin)

    with pytest.raises(OfficialAdapterInputError, match="official_binary"):
        prepare_official_candidate_adapter_input(candidate, spec, resolution)


@pytest.mark.parametrize(
    "name",
    (
        OFFICIAL_CANDIDATE_MAPPING_REF,
        OFFICIAL_SELECTED_HEADER_REF,
        OFFICIAL_SELECTED_LIBRARY_REF,
        OFFICIAL_REQUIRED_SYMBOLS_REF,
        OFFICIAL_SEMANTIC_REVIEW_REF,
    ),
)
def test_every_official_evidence_binding_is_required(name):
    spec, resolution = _fixture_values()
    candidate = _candidate(spec, resolution, omit=(name,))

    with pytest.raises(OfficialAdapterInputError, match=name):
        prepare_official_candidate_adapter_input(candidate, spec, resolution)


def test_official_evidence_hash_drift_is_rejected():
    spec, resolution = _fixture_values()
    candidate = _candidate(
        spec,
        resolution,
        ref_overrides={OFFICIAL_SELECTED_LIBRARY_REF: "f" * 64},
    )

    with pytest.raises(
        OfficialAdapterInputError, match=OFFICIAL_SELECTED_LIBRARY_REF
    ):
        prepare_official_candidate_adapter_input(candidate, spec, resolution)


def test_gap_and_resolution_binding_drift_is_rejected():
    spec, resolution = _fixture_values()
    candidate = _candidate(spec, resolution)
    drifted_spec = replace(spec, dtype_rules=("float16",))

    with pytest.raises(OfficialAdapterInputError, match="binding.*drift"):
        prepare_official_candidate_adapter_input(
            candidate, drifted_spec, resolution
        )

    drifted_value = resolution.to_dict()
    drifted_value["reason"] = "another bounded but unbound resolution reason"
    drifted_resolution = OfficialGapResolution.from_dict(drifted_value)
    with pytest.raises(OfficialAdapterInputError, match="binding.*drift"):
        prepare_official_candidate_adapter_input(
            candidate, spec, drifted_resolution
        )


def test_catalog_mapping_drift_is_rejected_even_when_candidate_binds_decision():
    spec, resolution = _fixture_values()
    value = resolution.to_dict()
    value["installed_cann"]["candidate"]["semantic_review"][0] = (
        "tampered semantic review"
    )
    drifted = OfficialGapResolution.from_dict(value)
    candidate = _candidate(spec, drifted)

    with pytest.raises(OfficialAdapterInputError, match="mapping drift"):
        prepare_official_candidate_adapter_input(candidate, spec, drifted)


def test_bounded_loader_accepts_direct_decision_and_complete_report(tmp_path):
    spec, resolution = _fixture_values()
    candidate = _candidate(spec, resolution)
    candidate_path = tmp_path / "candidate.json"
    spec_path = tmp_path / "gap-spec.json"
    decision_path = tmp_path / "decision.json"
    _write_json(candidate_path, candidate.to_dict())
    _write_json(spec_path, spec.to_dict())
    _write_json(decision_path, resolution.to_dict())

    direct = load_official_candidate_adapter_input(
        candidate_path, spec_path, decision_path
    )
    report = load_official_candidate_adapter_input(
        candidate_path, spec_path, OFFICIAL_REPORT
    )

    assert direct.to_dict() == report.to_dict()


def test_bounded_loader_rejects_symlink_fifo_duplicate_and_unknown_fields(tmp_path):
    spec, resolution = _fixture_values()
    candidate = _candidate(spec, resolution)
    candidate_path = tmp_path / "candidate.json"
    spec_path = tmp_path / "gap-spec.json"
    decision_path = tmp_path / "decision.json"
    _write_json(candidate_path, candidate.to_dict())
    _write_json(spec_path, spec.to_dict())
    _write_json(decision_path, resolution.to_dict())

    symlink = tmp_path / "candidate-link.json"
    symlink.symlink_to(candidate_path)
    with pytest.raises(OfficialAdapterInputError, match="safely read"):
        load_official_candidate_adapter_input(symlink, spec_path, decision_path)

    fifo = tmp_path / "candidate.fifo"
    os.mkfifo(fifo)
    with pytest.raises(OfficialAdapterInputError, match="regular file"):
        load_official_candidate_adapter_input(fifo, spec_path, decision_path)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"schema_version":1,"schema_version":1}\n', encoding="utf-8"
    )
    with pytest.raises(OfficialAdapterInputError, match="duplicate JSON key"):
        load_official_candidate_adapter_input(
            duplicate, spec_path, decision_path
        )

    unknown = candidate.to_dict()
    unknown["remote_model_artifact"] = True
    _write_json(candidate_path, unknown)
    with pytest.raises(OfficialAdapterInputError, match="invalid operator candidate"):
        load_official_candidate_adapter_input(
            candidate_path, spec_path, decision_path
        )
