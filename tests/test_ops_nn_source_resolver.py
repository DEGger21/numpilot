import hashlib
from pathlib import Path
import subprocess

import pytest

from numpilot.gaps.official_source import (
    OfficialSourceSemanticAudit,
    OpsNnSourcePin,
    OpsNnSourceResolver,
)


OFFICIAL_ORIGIN = "https://gitcode.com/cann/ops-nn.git"


def _run(*args: str, cwd: Path) -> str:
    return subprocess.run(
        args, cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def _operator(
    root: Path,
    op_path: str,
    *,
    api_name: str,
    soc: str = "ascend910b",
    with_op_api: bool = True,
) -> None:
    base = root / op_path
    (base / "op_host" / "config" / soc).mkdir(parents=True)
    (base / "op_kernel").mkdir(parents=True)
    (base / "docs").mkdir(parents=True)
    (base / "CMakeLists.txt").write_text(
        'set(SUPPORT_COMPUTE_UNIT "%s")\n' % soc, encoding="utf-8"
    )
    (base / "op_host" / "definition.cpp").write_text(
        "// host definition\n", encoding="utf-8"
    )
    (base / "op_host" / "config" / soc / "binary.json").write_text(
        "{}\n", encoding="utf-8"
    )
    (base / "op_kernel" / "kernel.cpp").write_text(
        "// kernel\n", encoding="utf-8"
    )
    (base / "docs" / ("aclnn%s.md" % api_name)).write_text(
        "reviewable API semantics\n", encoding="utf-8"
    )
    if with_op_api:
        (base / "op_api").mkdir()
        snake = "".join(
            ("_" + char.lower()) if char.isupper() else char
            for char in api_name
        ).lstrip("_")
        (base / "op_api" / ("aclnn_%s.h" % snake)).write_text(
            "// public ACLNN declaration\n", encoding="utf-8"
        )


def _repo(
    tmp_path: Path,
    *,
    version: str = "9.0.0",
    origin: str = OFFICIAL_ORIGIN,
    operators=(
        ("loss/smooth_l1_loss_v2", "SmoothL1Loss", "ascend910b", True),
    ),
    license_text: str = (
        "CANN Open Software License Agreement Version 2.0\n"
        "Licensed solely for Huawei AI Processors.\n"
    ),
):
    root = tmp_path / "ops-nn"
    root.mkdir(parents=True)
    (root / "version.cmake").write_text(
        'set_package(ops-nn VERSION "%s")\n' % version, encoding="utf-8"
    )
    (root / "LICENSE").write_text(license_text, encoding="utf-8")
    for op_path, api_name, soc, with_op_api in operators:
        _operator(
            root,
            op_path,
            api_name=api_name,
            soc=soc,
            with_op_api=with_op_api,
        )
    _run("git", "init", "-q", cwd=root)
    _run("git", "remote", "add", "origin", origin, cwd=root)
    _run("git", "add", ".", cwd=root)
    _run(
        "git", "-c", "user.name=NumPilot Test", "-c",
        "user.email=numpilot@example.invalid", "commit", "-qm", "fixture",
        cwd=root,
    )
    _run("git", "tag", "v%s" % version, cwd=root)
    commit = _run("git", "rev-parse", "HEAD^{commit}", cwd=root)
    tree = _run("git", "rev-parse", "HEAD^{tree}", cwd=root)
    pin = OpsNnSourcePin(
        ref="refs/tags/v%s" % version,
        expected_commit=commit,
        expected_tree=tree,
    )
    return root, pin


def _resolve(root: Path, pin: OpsNnSourcePin, **overrides):
    request = {
        "cann_version": "9.0.0",
        "target_soc": "Ascend910B4",
        "op_name": "smooth_l1_loss",
        "qualified_api": "operator.smooth_l1_loss",
    }
    request.update(overrides)
    return OpsNnSourceResolver(root, pins={"9.0.0": pin}).resolve(**request)


def test_pinned_object_is_inspected_without_reading_or_mutating_worktree(
    tmp_path, monkeypatch
):
    root, pin = _repo(tmp_path)
    cmake = root / "loss" / "smooth_l1_loss_v2" / "CMakeLists.txt"
    cmake.write_text(
        'set(SUPPORT_COMPUTE_UNIT "ascend950")\nDIRTY_SENTINEL\n',
        encoding="utf-8",
    )
    status_before = _run("git", "status", "--porcelain=v1", cwd=root)
    head_before = _run("git", "rev-parse", "HEAD", cwd=root)
    real_popen = subprocess.Popen
    observed_commands = []

    def observe(command, **kwargs):
        observed_commands.append(tuple(command))
        return real_popen(command, **kwargs)

    monkeypatch.setattr(
        "numpilot.gaps.official_source.subprocess.Popen", observe
    )

    result = _resolve(root, pin)
    resolver_commands = tuple(observed_commands)

    assert result.status == "ambiguous"
    assert result.resolution_kind == "candidate"
    assert result.requires_audit is True
    assert result.activation_allowed is False
    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.op_path == "loss/smooth_l1_loss_v2"
    assert candidate.stability == "stable"
    assert candidate.integration_surface_complete is True
    assert candidate.soc_cmake_clues == ("ascend910b",)
    assert candidate.soc_config_clues == ("ascend910b",)
    assert candidate.target_soc_evidence == "family"
    assert candidate.source_commit == pin.expected_commit
    assert candidate.source_tree == pin.expected_tree
    assert candidate.to_dict()["license"][
        "acceptance_required_before_use_or_build"
    ] is True
    assert result.to_dict()["activation_allowed"] is False
    assert resolver_commands
    assert not any(
        forbidden in command
        for command in resolver_commands
        for forbidden in ("fetch", "checkout", "switch", "worktree", "build")
    )
    assert _run("git", "status", "--porcelain=v1", cwd=root) == status_before
    assert _run("git", "rev-parse", "HEAD", cwd=root) == head_before
    assert "DIRTY_SENTINEL" not in str(candidate.to_dict())


def test_resolver_never_executes_a_path_shadowed_git(tmp_path, monkeypatch):
    root, pin = _repo(tmp_path / "repo")
    shadow_root = tmp_path / "shadow-bin"
    shadow_root.mkdir()
    marker = tmp_path / "shadow-was-executed"
    shadow = shadow_root / "git"
    shadow.write_text(
        "#!/bin/sh\nprintf shadow > '%s'\nexit 1\n" % marker,
        encoding="utf-8",
    )
    shadow.chmod(0o755)
    monkeypatch.setenv("PATH", str(shadow_root))

    result = _resolve(root, pin)

    assert result.status == "ambiguous"
    assert not marker.exists()


def test_oversized_git_config_output_fails_closed(tmp_path):
    oversized_origin = OFFICIAL_ORIGIN + "?" + ("x" * 8_192)
    root, pin = _repo(tmp_path, origin=oversized_origin)

    result = _resolve(root, pin)

    assert result.status == "error"
    assert result.candidates == ()
    assert result.activation_allowed is False
    assert "exceeded" in result.reason
    assert "4096-byte" in result.reason


def test_oversized_git_blob_fails_closed_while_streaming(tmp_path):
    license_text = (
        "CANN Open Software License Agreement Version 2.0\n"
        "Licensed solely for Huawei AI Processors.\n"
        + ("x" * (4 * 1024 * 1024))
    )
    root, pin = _repo(tmp_path, license_text=license_text)

    result = _resolve(root, pin)

    assert result.status == "error"
    assert result.candidates == ()
    assert result.activation_allowed is False
    assert "exceeded" in result.reason
    assert "4194304-byte" in result.reason


def test_resolver_ignores_local_git_replace_refs(tmp_path):
    root, pin = _repo(tmp_path)
    original_blob = _run("git", "rev-parse", "HEAD:LICENSE", cwd=root)
    replacement = root / "replacement-license"
    replacement.write_text("unreviewed replacement content\n", encoding="utf-8")
    replacement_blob = _run(
        "git", "hash-object", "-w", str(replacement), cwd=root
    )
    _run("git", "replace", original_blob, replacement_blob, cwd=root)

    # A normal local object read now sees the attacker-controlled replacement.
    assert "unreviewed replacement" in _run(
        "git", "cat-file", "blob", original_blob, cwd=root
    )

    result = _resolve(root, pin)

    assert result.status == "ambiguous"
    assert len(result.candidates) == 1
    assert result.candidates[0].license_sha256 == hashlib.sha256(
        b"CANN Open Software License Agreement Version 2.0\n"
        b"Licensed solely for Huawei AI Processors.\n"
    ).hexdigest()


def test_untrusted_semantic_audit_cannot_promote_candidate(tmp_path):
    root, pin = _repo(tmp_path)
    discovered = _resolve(root, pin)
    candidate = discovered.candidates[0]
    audit = OfficialSourceSemanticAudit(
        candidate_id=candidate.candidate_id,
        qualified_api="operator.smooth_l1_loss",
        cann_version="9.0.0",
        target_soc="Ascend910B4",
        contract_sha256=hashlib.sha256(b"gap contract").hexdigest(),
        evidence_sha256=hashlib.sha256(b"semantic audit evidence").hexdigest(),
        reviewer="offline-semantic-audit/v1",
    )

    result = _resolve(root, pin, semantic_audit=audit)

    assert result.status == "error"
    assert result.selected_candidate_id == ""
    assert result.requires_audit is True
    assert result.activation_allowed is False
    assert "promotion is disabled" in result.reason


def test_forged_contract_hash_and_reviewer_fail_closed(tmp_path):
    root, pin = _repo(tmp_path)
    candidate = _resolve(root, pin).candidates[0]
    audit = OfficialSourceSemanticAudit(
        candidate_id=candidate.candidate_id,
        qualified_api="operator.smooth_l1_loss",
        cann_version="9.0.0",
        target_soc="Ascend910B4",
        contract_sha256="1" * 64,
        evidence_sha256="2" * 64,
        reviewer="forged-reviewer",
    )

    result = _resolve(root, pin, semantic_audit=audit)

    assert result.status == "error"
    assert result.requires_audit is True
    assert result.activation_allowed is False
    assert "promotion is disabled" in result.reason


def test_name_match_without_complete_source_surface_stays_ambiguous(tmp_path):
    root, pin = _repo(
        tmp_path,
        operators=(
            ("optim/apply_gradient_descent", "ApplyGradientDescent", "ascend910b", False),
        ),
    )
    resolver = OpsNnSourceResolver(root, pins={"9.0.0": pin})
    found = resolver.resolve(
        cann_version="9.0.0",
        target_soc="Ascend910B4",
        op_name="apply_gradient_descent",
        qualified_api="operator.apply_gradient_descent",
    )
    candidate = found.candidates[0]
    audit = OfficialSourceSemanticAudit(
        candidate_id=candidate.candidate_id,
        qualified_api="operator.apply_gradient_descent",
        cann_version="9.0.0",
        target_soc="Ascend910B4",
        contract_sha256="3" * 64,
        evidence_sha256="4" * 64,
        reviewer="manual review",
    )

    result = resolver.resolve(
        cann_version="9.0.0",
        target_soc="Ascend910B4",
        op_name="apply_gradient_descent",
        qualified_api="operator.apply_gradient_descent",
        semantic_audit=audit,
    )

    assert found.status == "ambiguous"
    assert candidate.has_op_api is False
    assert result.status == "error"
    assert result.requires_audit is True
    assert "promotion is disabled" in result.reason


def test_multiple_stable_and_experimental_matches_are_reported(tmp_path):
    root, pin = _repo(
        tmp_path,
        operators=(
            ("activation/foo", "Foo", "ascend910b", True),
            ("experimental/activation/foo_v2", "Foo", "ascend950", True),
        ),
    )
    result = OpsNnSourceResolver(root, pins={"9.0.0": pin}).resolve(
        cann_version="9.0.0",
        target_soc="Ascend910B4",
        op_name="foo",
        qualified_api="operator.foo",
    )

    assert result.status == "ambiguous"
    assert result.requires_audit is True
    assert {item.op_path for item in result.candidates} == {
        "activation/foo", "experimental/activation/foo_v2"
    }
    assert {item.stability for item in result.candidates} == {
        "stable", "experimental"
    }


def test_no_match_does_not_claim_audit_or_activation(tmp_path):
    root, pin = _repo(tmp_path)

    result = _resolve(
        root,
        pin,
        op_name="arange",
        qualified_api="numpy.arange",
    )

    assert result.status == "no_match"
    assert result.candidates == ()
    assert result.requires_audit is False
    assert result.activation_allowed is False
    assert result.negative_search_complete is True
    assert result.search_contract_revision
    assert set(result.searched_aliases) == {"arange", "range"}


def test_reviewed_matmul_resolves_unique_v3_candidate(tmp_path):
    root, pin = _repo(
        tmp_path,
        operators=(("matmul/mat_mul_v3", "Matmul", "ascend910b", True),),
    )

    result = _resolve(
        root,
        pin,
        op_name="matmul",
        qualified_api="numpy.matmul",
    )

    assert result.status == "ambiguous"
    assert result.resolution_kind == "candidate"
    assert result.requires_audit is True
    assert result.activation_allowed is False
    assert result.negative_search_complete is False
    assert set(result.searched_aliases) == {"matmul", "mat_mul", "mat_mul_v3"}
    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.op_path == "matmul/mat_mul_v3"
    assert candidate.integration_surface_complete is True
    assert candidate.target_soc_evidence == "family"
    assert candidate.source_commit == pin.expected_commit
    assert candidate.source_tree == pin.expected_tree


def test_unreviewed_name_miss_is_ambiguous_not_clean_no_match(tmp_path):
    root, pin = _repo(tmp_path)

    result = _resolve(
        root,
        pin,
        op_name="unreviewed_gap",
        qualified_api="numpy.unreviewed_gap",
    )

    assert result.status == "ambiguous"
    assert result.resolution_kind == "unmapped"
    assert result.requires_audit is True
    assert result.negative_search_complete is False
    assert "not a clean official miss" in result.reason


def test_nested_numpy_api_is_syntactically_valid_but_unreviewed_and_ambiguous(
    tmp_path,
):
    root, pin = _repo(tmp_path)

    result = _resolve(
        root,
        pin,
        op_name="normal",
        qualified_api="numpy.random.normal",
    )

    assert result.status == "ambiguous"
    assert result.resolution_kind == "unmapped"
    assert result.negative_search_complete is False


def test_origin_version_license_and_pin_mismatches_are_incompatible(tmp_path):
    wrong_origin, origin_pin = _repo(
        tmp_path / "origin", origin="https://example.invalid/cann/ops-nn.git"
    )
    origin_result = _resolve(wrong_origin, origin_pin)

    wrong_version, version_pin = _repo(tmp_path / "version", version="9.1.0")
    version_result = OpsNnSourceResolver(
        wrong_version, pins={"9.0.0": version_pin}
    ).resolve(
        cann_version="9.0.0",
        target_soc="Ascend910B4",
        op_name="smooth_l1_loss",
        qualified_api="operator.smooth_l1_loss",
    )

    wrong_license, license_pin = _repo(
        tmp_path / "license", license_text="Apache License 2.0\n"
    )
    license_result = _resolve(wrong_license, license_pin)

    real_root, real_pin = _repo(tmp_path / "pin")
    moved_pin = OpsNnSourcePin(
        ref=real_pin.ref,
        expected_commit="0" * 40,
        expected_tree=real_pin.expected_tree,
    )
    pin_result = _resolve(real_root, moved_pin)

    assert origin_result.status == "incompatible"
    assert version_result.status == "incompatible"
    assert license_result.status == "incompatible"
    assert pin_result.status == "incompatible"
    assert all(
        result.activation_allowed is False
        for result in (origin_result, version_result, license_result, pin_result)
    )


def test_unreviewed_cann_version_and_invalid_request_fail_closed(tmp_path):
    root, pin = _repo(tmp_path)
    resolver = OpsNnSourceResolver(root, pins={"9.0.0": pin})

    unsupported = resolver.resolve(
        cann_version="9.1.0",
        target_soc="Ascend910B4",
        op_name="smooth_l1_loss",
        qualified_api="operator.smooth_l1_loss",
    )
    invalid = resolver.resolve(
        cann_version="9.0.0",
        target_soc="Ascend910B4",
        op_name="smooth_l1_loss;checkout",
        qualified_api="operator.smooth_l1_loss",
    )

    assert unsupported.status == "incompatible"
    assert "no reviewed" in unsupported.reason
    assert invalid.status == "error"


def test_explicit_empty_pin_catalog_does_not_fall_back_to_defaults(tmp_path):
    root, _pin = _repo(tmp_path)

    result = OpsNnSourceResolver(root, pins={}).resolve(
        cann_version="9.0.0",
        target_soc="Ascend910B4",
        op_name="smooth_l1_loss",
        qualified_api="operator.smooth_l1_loss",
    )

    assert result.status == "incompatible"
    assert "no reviewed" in result.reason


def test_checkout_symlink_loop_is_rejected_as_a_local_input_error(tmp_path):
    loop = tmp_path / "ops-nn-loop"
    loop.symlink_to(loop, target_is_directory=True)

    with pytest.raises(ValueError, match="real canonical directory"):
        OpsNnSourceResolver(loop)
