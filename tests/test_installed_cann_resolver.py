import hashlib
import json
from dataclasses import replace
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

import numpilot.gaps.installed_cann as installed_cann
from numpilot.gaps.installed_cann import (
    ACLNN_ADAPTER_CANDIDATES,
    AMBIGUOUS,
    CannInventoryError,
    ERROR,
    EXACT_CANDIDATE,
    INCOMPATIBLE,
    NO_MATCH,
    SymbolProbeOutput,
    inventory_installed_cann,
    probe_dynamic_symbols,
    resolve_installed_aclnn as _raw_resolve_installed_aclnn,
    resolve_installed_aclnn_from_inventory as _raw_resolve_from_inventory,
)


TARGET_SOC = "ascend910b4"


def resolve_installed_aclnn(*args, **kwargs):
    kwargs.setdefault("expected_target_soc", TARGET_SOC)
    return _raw_resolve_installed_aclnn(*args, **kwargs)


def resolve_installed_aclnn_from_inventory(*args, **kwargs):
    kwargs.setdefault("expected_target_soc", TARGET_SOC)
    return _raw_resolve_from_inventory(*args, **kwargs)


def _candidate(api):
    return next(
        item for item in ACLNN_ADAPTER_CANDIDATES
        if item.qualified_api == api
    )


def _symbols_by_library(candidates=ACLNN_ADAPTER_CANDIDATES):
    result = {}
    for candidate in candidates:
        for library in candidate.library_priority:
            result.setdefault(library, set()).update(candidate.required_symbols)
    return result


def _write_toolkit(
    tmp_path,
    *,
    version="9.0.0",
    candidates=ACLNN_ADAPTER_CANDIDATES,
    headers=True,
    libraries=True,
    platform="aarch64-linux",
):
    root = tmp_path / ("cann-" + version)
    platform_root = root / platform
    header_root = platform_root / "include" / "aclnnop"
    library_root = platform_root / "lib64"
    header_root.mkdir(parents=True)
    library_root.mkdir(parents=True)
    arch, operating_system = platform.rsplit("-", 1)
    (platform_root / "ascend_toolkit_install.info").write_text(
        "package_name=Ascend-cann-toolkit\n"
        "version=%s\n"
        "arch=%s\n"
        "os=%s\n" % (version, arch, operating_system),
        encoding="utf-8",
    )
    opp = root / "opp"
    opp.mkdir()
    (opp / "version.info").write_text(
        "Version=%s\nversion_dir=cann\n" % version,
        encoding="utf-8",
    )
    if headers:
        by_header = {}
        for candidate in candidates:
            by_header.setdefault(candidate.header_name, set()).update(
                candidate.required_symbols
            )
        for name, symbols in by_header.items():
            (header_root / name).write_text(
                "#pragma once\n" + "\n".join(
                    "int %s(void);" % symbol for symbol in sorted(symbols)
                ) + "\n",
                encoding="utf-8",
            )
    if libraries:
        for name in _symbols_by_library(candidates):
            (library_root / name).write_bytes(
                ("ELF-test-fixture:" + name).encode("ascii")
            )
    return root


def _probe(symbols=None, calls=None):
    symbol_map = symbols or _symbols_by_library()

    def probe(path):
        if calls is not None:
            calls.append(path)
        return SymbolProbeOutput(
            tool="injected-test-probe",
            symbols=tuple(symbol_map.get(path.name, ())),
        )

    return probe


@pytest.mark.parametrize(
    ("api", "expected_header", "expected_library", "expected_symbols"),
    (
        (
            "numpy.cbrt",
            "aclnn_pow.h",
            "libopapi_math.so",
            ("aclnnPowTensorScalarGetWorkspaceSize", "aclnnPowTensorScalar"),
        ),
        (
            "numpy.arange",
            "aclnn_arange.h",
            "libopapi_math.so",
            ("aclnnArangeGetWorkspaceSize", "aclnnArange"),
        ),
        (
            "numpy.matmul",
            "aclnn_matmul.h",
            "libopapi.so",
            ("aclnnMatmulGetWorkspaceSize", "aclnnMatmul"),
        ),
        (
            "numpy.cumsum",
            "aclnn_cumsum.h",
            "libopapi_math.so",
            ("aclnnCumsumGetWorkspaceSize", "aclnnCumsum"),
        ),
        (
            "numpy.cumprod",
            "aclnn_cumprod.h",
            "libopapi_math.so",
            ("aclnnCumprodGetWorkspaceSize", "aclnnCumprod"),
        ),
        (
            "operator.erf",
            "aclnn_erf.h",
            "libopapi_math.so",
            ("aclnnErfGetWorkspaceSize", "aclnnErf"),
        ),
        (
            "numpy.random.normal",
            "aclnn_normal.h",
            "libopapi_math.so",
            ("aclnnInplaceNormalGetWorkspaceSize", "aclnnInplaceNormal"),
        ),
        (
            "operator.smooth_l1_loss",
            "aclnn_smooth_l1_loss.h",
            "libopapi_nn.so",
            ("aclnnSmoothL1LossGetWorkspaceSize", "aclnnSmoothL1Loss"),
        ),
    ),
)
def test_audited_installed_mapping_is_only_an_adapter_candidate(
    tmp_path,
    api,
    expected_header,
    expected_library,
    expected_symbols,
):
    root = _write_toolkit(tmp_path)

    result = resolve_installed_aclnn(
        api,
        cann_root=root,
        expected_cann_version="9.0.0",
        symbol_probe=_probe(),
    )

    assert result.status == EXACT_CANDIDATE
    assert result.adapter_candidate is True
    assert result.header.file.relative_path.endswith("/" + expected_header)
    assert result.library.file.relative_path.endswith("/" + expected_library)
    assert result.candidate.required_symbols == expected_symbols
    assert result.inventory.version == "9.0.0"
    assert result.inventory.platform == "aarch64-linux"
    payload = result.to_dict()
    assert payload["expected_target_soc"] == TARGET_SOC
    assert payload["candidate"]["audited_target_socs"] == ["ascend910b4"]
    assert payload["semantic_equivalence"] == "unverified"
    assert payload["required_next_gate"] == (
        "generate_and_qualify_asnumpy_adapter"
    )
    assert payload["provenance"] == (
        "installed_cann_toolkit_read_only_inventory"
    )
    assert len(payload["candidate_mapping_sha256"]) == 64
    assert len(payload["selected_header"]["sha256"]) == 64
    assert len(payload["selected_library"]["sha256"]) == 64
    assert len(payload["selected_library"]["aclnn_symbols_sha256"]) == 64
    assert all(
        len(item["sha256"]) == 64
        for item in payload["inventory"]["version_evidence"]
    )
    json.dumps(payload, sort_keys=True)


def test_concatenate_mapping_matches_only_the_recorded_runtime(tmp_path):
    candidate = _candidate("numpy.concatenate")
    assert candidate.candidate_id == "numpy.concatenate-aclnn-cat"
    assert candidate.header_name == "aclnn_cat.h"
    assert candidate.required_symbols == (
        "aclnnCatGetWorkspaceSize",
        "aclnnCat",
    )
    assert candidate.audited_cann_versions == ("9.2.0-beta.2",)
    assert candidate.audited_target_socs == ("ascend910b3",)
    assert candidate.library_priority == (
        "libopapi.so",
        "libopapi_math.so",
    )

    root = _write_toolkit(tmp_path, version="9.2.0-beta.2")
    result = _raw_resolve_installed_aclnn(
        "numpy.concatenate",
        cann_root=root,
        expected_cann_version="9.2.0-beta.2",
        expected_target_soc="Ascend910B3",
        symbol_probe=_probe(),
    )

    assert result.status == EXACT_CANDIDATE
    assert result.adapter_candidate is True
    assert result.reason_code == "header_and_dynamic_symbols_present"
    assert result.header.file.relative_path.endswith("/aclnn_cat.h")
    assert result.library.file.relative_path.endswith("/libopapi.so")
    payload = result.to_dict()
    assert payload["semantic_equivalence"] == "unverified"
    assert payload["required_next_gate"] == (
        "generate_and_qualify_asnumpy_adapter"
    )


def test_concatenate_mapping_does_not_inherit_older_audit_profiles(tmp_path):
    root = _write_toolkit(tmp_path)

    result = resolve_installed_aclnn(
        "numpy.concatenate",
        cann_root=root,
        expected_cann_version="9.0.0",
        symbol_probe=lambda path: pytest.fail("probe must not run"),
    )

    assert result.status == INCOMPATIBLE
    assert result.reason_code == "candidate_target_soc_not_audited"
    assert result.candidate.candidate_id == "numpy.concatenate-aclnn-cat"


def test_cbrt_910b3_mapping_matches_only_the_recorded_runtime(tmp_path):
    candidate = next(
        item for item in ACLNN_ADAPTER_CANDIDATES
        if item.candidate_id == "numpy.cbrt-aclnn-pow-tensor-scalar-910b3"
    )
    assert candidate.qualified_api == "numpy.cbrt"
    assert candidate.header_name == "aclnn_pow.h"
    assert candidate.required_symbols == (
        "aclnnPowTensorScalarGetWorkspaceSize",
        "aclnnPowTensorScalar",
    )
    assert candidate.audited_cann_versions == ("9.2.0-beta.2",)
    assert candidate.audited_target_socs == ("ascend910b3",)
    assert candidate.library_priority == (
        "libopapi.so",
        "libopapi_math.so",
    )

    root = _write_toolkit(tmp_path, version="9.2.0-beta.2")
    result = _raw_resolve_installed_aclnn(
        "numpy.cbrt",
        cann_root=root,
        expected_cann_version="9.2.0-beta.2",
        expected_target_soc="Ascend910B3",
        symbol_probe=_probe(),
    )

    assert result.status == EXACT_CANDIDATE
    assert result.adapter_candidate is True
    assert result.reason_code == "header_and_dynamic_symbols_present"
    assert result.candidate.candidate_id == (
        "numpy.cbrt-aclnn-pow-tensor-scalar-910b3"
    )
    assert result.header.file.relative_path.endswith("/aclnn_pow.h")
    assert result.library.file.relative_path.endswith("/libopapi.so")
    payload = result.to_dict()
    assert payload["semantic_equivalence"] == "unverified"
    assert payload["required_next_gate"] == (
        "generate_and_qualify_asnumpy_adapter"
    )


def test_cbrt_910b4_mapping_still_resolves_only_its_own_profile(tmp_path):
    root = _write_toolkit(tmp_path)

    result = resolve_installed_aclnn(
        "numpy.cbrt",
        cann_root=root,
        expected_cann_version="9.0.0",
        symbol_probe=_probe(),
    )

    assert result.status == EXACT_CANDIDATE
    assert result.candidate.candidate_id == (
        "numpy.cbrt-aclnn-pow-tensor-scalar"
    )
    assert result.candidate.audited_target_socs == ("ascend910b4",)


def test_cbrt_profile_does_not_cross_the_two_audit_claims(tmp_path):
    root = _write_toolkit(tmp_path)

    result = _raw_resolve_installed_aclnn(
        "numpy.cbrt",
        cann_root=root,
        expected_cann_version="9.0.0",
        expected_target_soc="Ascend910B2",
        symbol_probe=lambda path: pytest.fail("probe must not run"),
    )

    assert result.status == INCOMPATIBLE
    assert result.reason_code == "candidate_target_soc_not_audited"
    assert result.candidate.candidate_id == (
        "numpy.cbrt-aclnn-pow-tensor-scalar"
    )


def test_inventory_hashes_exact_files_and_records_no_full_symbol_listing(tmp_path):
    root = _write_toolkit(tmp_path)

    inventory = inventory_installed_cann(root, symbol_probe=_probe())

    header = inventory.header("aclnn_arange.h")
    library = inventory.library("libopapi_math.so")
    assert header.file.sha256 == hashlib.sha256(
        (root / header.file.relative_path).read_bytes()
    ).hexdigest()
    assert library.file.sha256 == hashlib.sha256(
        (root / library.file.relative_path).read_bytes()
    ).hexdigest()
    serialized = inventory.to_dict()
    assert "symbols" not in serialized["libraries"][0]
    assert serialized["libraries"][0]["aclnn_symbol_count"] > 0


def test_one_inventory_resolves_multiple_gaps_without_reprobing_libraries(tmp_path):
    root = _write_toolkit(tmp_path)
    calls = []
    inventory = inventory_installed_cann(
        root,
        symbol_probe=_probe(calls=calls),
    )
    probe_count = len(calls)

    fixture_candidates = tuple(
        candidate
        for candidate in ACLNN_ADAPTER_CANDIDATES
        if "9.0.0" in candidate.audited_cann_versions
        and TARGET_SOC in candidate.audited_target_socs
    )
    results = tuple(
        resolve_installed_aclnn_from_inventory(
            candidate.qualified_api,
            inventory=inventory,
            expected_cann_version="9.0.0",
        )
        for candidate in fixture_candidates
    )

    assert probe_count == 3
    assert len(calls) == probe_count
    assert len(fixture_candidates) == 8
    assert all(result.status == EXACT_CANDIDATE for result in results)
    assert all(result.inventory is inventory for result in results)


@pytest.mark.parametrize(
    "api",
    (
        "numpy.definitely_missing",
        "numpy.linalg.future_operator",
        "operator.apply_gradient_descent",
    ),
)
def test_unreviewed_api_is_ambiguous_without_touching_cann_or_running_probe(
    tmp_path,
    api,
):
    calls = []

    result = resolve_installed_aclnn(
        api,
        cann_root=tmp_path / "does-not-exist",
        expected_cann_version="9.0.0",
        symbol_probe=_probe(calls=calls),
    )

    assert result.status == AMBIGUOUS
    assert result.status != NO_MATCH
    assert result.reason_code == "catalog_unreviewed_api"
    assert result.inventory is None
    assert calls == []


def test_unreviewed_api_from_inventory_is_not_no_match(tmp_path):
    root = _write_toolkit(tmp_path)
    inventory = inventory_installed_cann(root, symbol_probe=_probe())

    result = resolve_installed_aclnn_from_inventory(
        "numpy.unknown_after_inventory",
        inventory=inventory,
        expected_cann_version="9.0.0",
    )

    assert result.status == AMBIGUOUS
    assert result.status != NO_MATCH
    assert result.reason_code == "catalog_unreviewed_api"


def test_invalid_api_fails_closed_before_filesystem_access(tmp_path):
    result = resolve_installed_aclnn(
        "numpy.arange;touch /tmp/oops",
        cann_root=tmp_path / "does-not-exist",
        expected_cann_version="9.0.0",
        symbol_probe=lambda path: pytest.fail("probe must not run"),
    )

    assert result.status == ERROR
    assert result.reason_code == "invalid_request"


def test_missing_target_soc_is_ambiguous_before_filesystem_access(tmp_path):
    calls = []

    result = _raw_resolve_installed_aclnn(
        "numpy.arange",
        cann_root=tmp_path / "does-not-exist",
        expected_cann_version="9.0.0",
        symbol_probe=_probe(calls=calls),
    )

    assert result.status == AMBIGUOUS
    assert result.status not in {EXACT_CANDIDATE, NO_MATCH}
    assert result.reason_code == "target_soc_required"
    assert result.expected_target_soc == ""
    assert calls == []


@pytest.mark.parametrize(
    "target_soc",
    (
        "garbage",
        "Ascend910B4/../../tmp",
    ),
)
def test_unrecognized_target_soc_is_ambiguous_without_inventory(
    tmp_path,
    target_soc,
):
    result = _raw_resolve_installed_aclnn(
        "numpy.arange",
        cann_root=tmp_path / "does-not-exist",
        expected_cann_version="9.0.0",
        expected_target_soc=target_soc,
        symbol_probe=lambda path: pytest.fail("probe must not run"),
    )

    assert result.status == AMBIGUOUS
    assert result.status not in {EXACT_CANDIDATE, NO_MATCH}
    assert result.reason_code == "target_soc_unrecognized"


@pytest.mark.parametrize("target_soc", ("Ascend310P3", "Ascend910B1"))
def test_recognized_but_unaudited_target_soc_is_incompatible_without_inventory(
    tmp_path,
    target_soc,
):
    result = _raw_resolve_installed_aclnn(
        "numpy.arange",
        cann_root=tmp_path / "does-not-exist",
        expected_cann_version="9.0.0",
        expected_target_soc=target_soc,
        symbol_probe=lambda path: pytest.fail("probe must not run"),
    )

    assert result.status == INCOMPATIBLE
    assert result.status not in {EXACT_CANDIDATE, NO_MATCH}
    assert result.reason_code == "candidate_target_soc_not_audited"
    assert result.expected_target_soc == target_soc
    assert result.inventory is None


def test_generic_910b_family_fails_closed_before_inventory(tmp_path):
    result = _raw_resolve_installed_aclnn(
        "numpy.arange",
        cann_root=tmp_path / "does-not-exist",
        expected_cann_version="9.0.0",
        expected_target_soc="ascend910b",
        candidates=(object(),),
        symbol_probe=lambda path: pytest.fail("probe must not run"),
    )

    assert result.status == AMBIGUOUS
    assert result.reason_code == "target_soc_family_not_exact"
    assert result.expected_target_soc == "ascend910b"
    assert result.inventory is None


def test_generic_910b_family_from_inventory_is_ambiguous(tmp_path):
    root = _write_toolkit(tmp_path)
    inventory = inventory_installed_cann(root, symbol_probe=_probe())

    result = _raw_resolve_from_inventory(
        "numpy.arange",
        inventory=inventory,
        expected_cann_version="9.0.0",
        expected_target_soc="Ascend-910B",
    )

    assert result.status == AMBIGUOUS
    assert result.reason_code == "target_soc_family_not_exact"
    assert result.inventory is inventory


def test_cross_soc_from_existing_inventory_is_incompatible(tmp_path):
    root = _write_toolkit(tmp_path)
    inventory = inventory_installed_cann(root, symbol_probe=_probe())

    result = _raw_resolve_from_inventory(
        "numpy.arange",
        inventory=inventory,
        expected_cann_version="9.0.0",
        expected_target_soc="Ascend310P3",
    )

    assert result.status == INCOMPATIBLE
    assert result.reason_code == "candidate_target_soc_not_audited"
    assert result.inventory is inventory


def test_missing_target_soc_from_existing_inventory_is_ambiguous(tmp_path):
    root = _write_toolkit(tmp_path)
    inventory = inventory_installed_cann(root, symbol_probe=_probe())

    result = _raw_resolve_from_inventory(
        "numpy.arange",
        inventory=inventory,
        expected_cann_version="9.0.0",
    )

    assert result.status == AMBIGUOUS
    assert result.status not in {EXACT_CANDIDATE, NO_MATCH}
    assert result.reason_code == "target_soc_required"
    assert result.inventory is inventory


def test_invalid_candidate_catalog_returns_error_instead_of_throwing(tmp_path):
    result = resolve_installed_aclnn(
        "numpy.arange",
        cann_root=tmp_path / "does-not-exist",
        expected_cann_version="9.0.0",
        candidates=(object(),),
        symbol_probe=lambda path: pytest.fail("probe must not run"),
    )

    assert result.status == ERROR
    assert result.reason_code == "invalid_candidate_catalog"


def test_installed_version_mismatch_is_incompatible_with_version_hashes(tmp_path):
    root = _write_toolkit(tmp_path)

    result = resolve_installed_aclnn(
        "numpy.arange",
        cann_root=root,
        expected_cann_version="9.1.0",
        symbol_probe=_probe(),
    )

    assert result.status == INCOMPATIBLE
    assert result.reason_code == "cann_version_mismatch"
    assert result.inventory.version == "9.0.0"
    assert len(result.inventory.version_evidence) == 2


def test_mapping_must_be_audited_for_the_exact_installed_version(tmp_path):
    root = _write_toolkit(tmp_path, version="9.1.0")

    result = resolve_installed_aclnn(
        "numpy.arange",
        cann_root=root,
        expected_cann_version="9.1.0",
        symbol_probe=_probe(),
    )

    assert result.status == INCOMPATIBLE
    assert result.reason_code == "candidate_version_not_audited"


def test_absent_header_and_symbols_is_no_match(tmp_path):
    root = _write_toolkit(tmp_path, headers=False, libraries=False)

    result = resolve_installed_aclnn(
        "numpy.arange",
        cann_root=root,
        expected_cann_version="9.0.0",
        symbol_probe=_probe(),
    )

    assert result.status == NO_MATCH
    assert result.reason_code == "candidate_artifacts_absent"
    assert set(result.missing_symbols) == set(
        _candidate("numpy.arange").required_symbols
    )


@pytest.mark.parametrize("missing_part", ("header", "workspace_symbol", "run_symbol"))
def test_partial_candidate_is_incompatible_not_a_match(tmp_path, missing_part):
    root = _write_toolkit(tmp_path)
    candidate = _candidate("numpy.arange")
    symbols = _symbols_by_library()
    if missing_part == "header":
        (root / "aarch64-linux/include/aclnnop/aclnn_arange.h").unlink()
    else:
        missing = candidate.required_symbols[
            0 if missing_part == "workspace_symbol" else 1
        ]
        for available in symbols.values():
            available.discard(missing)

    result = resolve_installed_aclnn(
        "numpy.arange",
        cann_root=root,
        expected_cann_version="9.0.0",
        symbol_probe=_probe(symbols),
    )

    assert result.status == INCOMPATIBLE
    assert result.reason_code == "candidate_artifacts_incomplete"
    assert result.adapter_candidate is False


def test_symbols_split_across_libraries_do_not_form_a_candidate(tmp_path):
    root = _write_toolkit(tmp_path)
    first, second = _candidate("numpy.arange").required_symbols
    symbols = _symbols_by_library()
    symbols["libopapi_math.so"] = {first}
    symbols["libopapi.so"] = {second}

    result = resolve_installed_aclnn(
        "numpy.arange",
        cann_root=root,
        expected_cann_version="9.0.0",
        symbol_probe=_probe(symbols),
    )

    assert result.status == INCOMPATIBLE
    assert result.library is None
    assert result.missing_symbols == ()


def test_multiple_catalog_mappings_are_ambiguous_without_inventory(tmp_path):
    original = _candidate("numpy.arange")
    duplicate = replace(original, candidate_id="numpy.arange-second-audit")

    result = resolve_installed_aclnn(
        "numpy.arange",
        cann_root=tmp_path / "does-not-exist",
        expected_cann_version="9.0.0",
        candidates=(original, duplicate),
        symbol_probe=lambda path: pytest.fail("probe must not run"),
    )

    assert result.status == AMBIGUOUS
    assert result.reason_code == "catalog_multiple_candidates"
    assert result.alternatives == (
        original.candidate_id,
        duplicate.candidate_id,
    )


def test_multiple_platform_manifests_are_ambiguous(tmp_path):
    root = _write_toolkit(tmp_path)
    other = root / "x86_64-linux"
    other.mkdir()
    (other / "ascend_toolkit_install.info").write_text(
        "package_name=Ascend-cann-toolkit\n"
        "version=9.0.0\narch=x86_64\nos=linux\n",
        encoding="utf-8",
    )

    result = resolve_installed_aclnn(
        "numpy.arange",
        cann_root=root,
        expected_cann_version="9.0.0",
        symbol_probe=_probe(),
    )

    assert result.status == AMBIGUOUS
    assert result.reason_code == "ambiguous_cann_platform"


def test_relative_and_symlinked_roots_fail_closed(tmp_path):
    root = _write_toolkit(tmp_path)
    alias = tmp_path / "latest"
    alias.symlink_to(root, target_is_directory=True)

    relative = resolve_installed_aclnn(
        "numpy.arange",
        cann_root=Path("relative/cann"),
        expected_cann_version="9.0.0",
        symbol_probe=_probe(),
    )
    symlinked = resolve_installed_aclnn(
        "numpy.arange",
        cann_root=alias,
        expected_cann_version="9.0.0",
        symbol_probe=_probe(),
    )

    assert relative.status == ERROR
    assert relative.reason_code == "invalid_cann_root"
    assert symlinked.status == ERROR
    assert symlinked.reason_code == "noncanonical_cann_root"


def test_root_replacement_before_openat_cannot_escape_anchor(
    tmp_path,
    monkeypatch,
):
    root = _write_toolkit(tmp_path)
    original_root = tmp_path / "original-cann-root"
    attacker_root = _write_toolkit(tmp_path / "attacker-root")
    real_open = os.open
    raced = []
    probe_calls = []

    def replacing_open(path, flags, *args, **kwargs):
        if path == root.name and not raced:
            assert flags & os.O_NOFOLLOW
            assert flags & os.O_DIRECTORY
            assert kwargs.get("dir_fd") is not None
            root.replace(original_root)
            root.symlink_to(attacker_root, target_is_directory=True)
            raced.append(True)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(installed_cann.os, "open", replacing_open)

    result = resolve_installed_aclnn(
        "numpy.arange",
        cann_root=root,
        expected_cann_version="9.0.0",
        symbol_probe=_probe(calls=probe_calls),
    )

    assert raced == [True]
    assert probe_calls == []
    assert result.status == ERROR
    assert result.reason_code in {
        "noncanonical_cann_root",
        "directory_identity_changed",
    }


def test_intermediate_replacement_before_openat_cannot_escape_root(
    tmp_path,
    monkeypatch,
):
    root = _write_toolkit(tmp_path)
    platform = root / "aarch64-linux"
    original_platform = root / "original-aarch64-linux"
    attacker_root = _write_toolkit(tmp_path / "attacker-platform")
    attacker_platform = attacker_root / "aarch64-linux"
    real_open = os.open
    raced = []
    probe_calls = []

    def replacing_open(path, flags, *args, **kwargs):
        if path == "aarch64-linux" and not raced:
            assert flags & os.O_NOFOLLOW
            assert flags & os.O_DIRECTORY
            assert kwargs.get("dir_fd") is not None
            platform.replace(original_platform)
            platform.symlink_to(attacker_platform, target_is_directory=True)
            raced.append(True)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(installed_cann.os, "open", replacing_open)

    result = resolve_installed_aclnn(
        "numpy.arange",
        cann_root=root,
        expected_cann_version="9.0.0",
        symbol_probe=_probe(calls=probe_calls),
    )

    assert raced == [True]
    assert probe_calls == []
    assert result.status == ERROR
    assert result.reason_code in {
        "symlink_evidence_rejected",
        "directory_identity_changed",
    }


@pytest.mark.parametrize("kind", ("header", "library"))
def test_symlinked_evidence_is_rejected(tmp_path, kind):
    root = _write_toolkit(tmp_path)
    if kind == "header":
        path = root / "aarch64-linux/include/aclnnop/aclnn_arange.h"
    else:
        path = root / "aarch64-linux/lib64/libopapi_math.so"
    backup = tmp_path / ("outside-" + path.name)
    path.replace(backup)
    path.symlink_to(backup)

    result = resolve_installed_aclnn(
        "numpy.arange",
        cann_root=root,
        expected_cann_version="9.0.0",
        symbol_probe=_probe(),
    )

    assert result.status == ERROR
    assert result.reason_code == "symlink_evidence_rejected"


def test_conflicting_version_manifests_are_error(tmp_path):
    root = _write_toolkit(tmp_path)
    (root / "opp/version.info").write_text(
        "Version=9.1.0\n",
        encoding="utf-8",
    )

    result = resolve_installed_aclnn(
        "numpy.arange",
        cann_root=root,
        expected_cann_version="9.0.0",
        symbol_probe=_probe(),
    )

    assert result.status == ERROR
    assert result.reason_code == "conflicting_cann_versions"


@pytest.mark.parametrize(
    "bad_probe",
    (
        lambda path: (_ for _ in ()).throw(RuntimeError("secret output")),
        lambda path: ("aclnnArange",),
    ),
)
def test_bad_injected_symbol_probe_returns_error_without_leaking_exception(tmp_path, bad_probe):
    root = _write_toolkit(tmp_path)

    result = resolve_installed_aclnn(
        "numpy.arange",
        cann_root=root,
        expected_cann_version="9.0.0",
        symbol_probe=bad_probe,
    )

    assert result.status == ERROR
    assert "secret output" not in result.reason


def test_injected_probe_path_replacement_is_detected(tmp_path):
    root = _write_toolkit(tmp_path)
    symbols = _symbols_by_library()
    replaced = []

    def replacing_probe(path):
        if not replaced:
            original = path.with_suffix(".original")
            path.replace(original)
            path.write_bytes(b"replacement ELF")
            replaced.append((path, original))
        return SymbolProbeOutput(
            tool="trusted-race-test-probe",
            symbols=tuple(symbols.get(path.name, ())),
        )

    result = resolve_installed_aclnn(
        "numpy.arange",
        cann_root=root,
        expected_cann_version="9.0.0",
        symbol_probe=replacing_probe,
    )

    assert result.status == ERROR
    assert result.reason_code in {
        "evidence_identity_changed",
        "directory_identity_changed",
    }
    assert replaced


def test_bounded_version_read_detects_path_replacement(tmp_path, monkeypatch):
    root = _write_toolkit(tmp_path)
    version_path = root / "aarch64-linux/ascend_toolkit_install.info"
    original_read = os.read
    replaced = []

    def racing_read(descriptor, count):
        payload = original_read(descriptor, count)
        if not replaced:
            backup = version_path.with_suffix(".original")
            version_path.replace(backup)
            version_path.write_bytes(payload or b"replacement")
            replaced.append(backup)
        return payload

    monkeypatch.setattr(installed_cann.os, "read", racing_read)

    with pytest.raises(CannInventoryError, match="changed"):
        installed_cann._read_bounded(
            root,
            version_path,
            installed_cann._MAX_VERSION_BYTES,
        )


@pytest.mark.parametrize("kind", ("directory", "fifo"))
def test_non_regular_library_is_rejected_without_blocking(tmp_path, kind):
    root = _write_toolkit(tmp_path)
    library = root / "aarch64-linux/lib64/libopapi_math.so"
    library.unlink()
    if kind == "directory":
        library.mkdir()
    else:
        os.mkfifo(library)

    result = resolve_installed_aclnn(
        "numpy.arange",
        cann_root=root,
        expected_cann_version="9.0.0",
        symbol_probe=_probe(),
    )

    assert result.status == ERROR
    assert result.reason_code == "invalid_evidence_type"


def test_default_inventory_hash_and_probe_share_the_same_descriptor(
    tmp_path,
    monkeypatch,
):
    root = _write_toolkit(tmp_path)
    symbols = _symbols_by_library()
    probed_hashes = {}

    def descriptor_probe(path, *, inherited_fd=None, **kwargs):
        assert inherited_fd is not None
        metadata = os.fstat(inherited_fd)
        assert metadata.st_ino == path.stat().st_ino
        offset = os.lseek(inherited_fd, 0, os.SEEK_CUR)
        os.lseek(inherited_fd, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        while True:
            chunk = os.read(inherited_fd, 1024)
            if not chunk:
                break
            digest.update(chunk)
        os.lseek(inherited_fd, offset, os.SEEK_SET)
        probed_hashes[path.name] = digest.hexdigest()
        return SymbolProbeOutput(
            tool="descriptor-binding-test-probe",
            symbols=tuple(symbols.get(path.name, ())),
        )

    monkeypatch.setattr(
        installed_cann,
        "probe_dynamic_symbols",
        descriptor_probe,
    )

    inventory = inventory_installed_cann(root)

    assert probed_hashes
    assert {
        item.file.relative_path.rsplit("/", 1)[-1]: item.file.sha256
        for item in inventory.libraries
    } == probed_hashes


def _executable(path):
    path.write_text("test tool placeholder\n", encoding="utf-8")
    path.chmod(0o755)
    return str(path)


def test_default_symbol_probe_prefers_nm_and_uses_no_shell_or_inherited_env(tmp_path):
    library = tmp_path / "libopapi.so"
    library.write_bytes(b"ELF")
    nm = _executable(tmp_path / "nm")
    calls = []

    def runner(command, **kwargs):
        inherited_fd = kwargs["pass_fds"][0]
        assert os.fstat(inherited_fd).st_ino == library.stat().st_ino
        calls.append((command, kwargs))
        return SimpleNamespace(
            returncode=0,
            stdout=(
                "aclnnArange T 0 1\n"
                "aclnnArangeGetWorkspaceSize T 1 1\n"
                "notAclnn T 2 1\n"
            ),
            stderr="",
        )

    result = probe_dynamic_symbols(
        library,
        finder=lambda name: nm if name == "nm" else None,
        runner=runner,
    )

    assert result.symbols == (
        "aclnnArange",
        "aclnnArangeGetWorkspaceSize",
    )
    assert result.tool.startswith("nm:")
    command, options = calls[0]
    assert command[0] == str(Path(nm).resolve())
    assert command[-1] == "/proc/self/fd/%d" % options["pass_fds"][0]
    assert options["pass_fds"]
    assert options["close_fds"] is True
    assert "shell" not in options
    assert options["stdin"] is not None
    assert options["stderr"] is not subprocess.PIPE
    assert options["text"] is False
    assert options["env"] == {"LANG": "C", "LC_ALL": "C"}


def test_default_symbol_probe_falls_back_to_readelf(tmp_path):
    library = tmp_path / "libopapi.so"
    library.write_bytes(b"ELF")
    nm = _executable(tmp_path / "nm")
    readelf = _executable(tmp_path / "readelf")
    calls = []

    def finder(name):
        return {"nm": nm, "readelf": readelf}.get(name)

    def runner(command, **kwargs):
        calls.append(Path(command[0]).name)
        if Path(command[0]).name == "nm":
            return SimpleNamespace(returncode=1, stdout="", stderr="bad elf")
        return SimpleNamespace(
            returncode=0,
            stdout="  1: 0000 1 FUNC GLOBAL DEFAULT 1 aclnnCumsum@@CANN_1\n",
            stderr="",
        )

    result = probe_dynamic_symbols(
        library,
        finder=finder,
        runner=runner,
    )

    assert result.tool.startswith("readelf:")
    assert result.symbols == ("aclnnCumsum",)
    assert calls == ["nm", "readelf"]


def test_default_symbol_probe_kills_output_over_the_hard_limit(
    tmp_path,
    monkeypatch,
):
    library = tmp_path / "libopapi.so"
    library.write_bytes(b"ELF")
    noisy_nm = tmp_path / "noisy-nm"
    noisy_nm.write_text(
        "#!/bin/sh\nprintf '%01024d\\n' 0\n",
        encoding="utf-8",
    )
    noisy_nm.chmod(0o755)
    monkeypatch.setattr(installed_cann, "_MAX_TOOL_OUTPUT", 64)

    with pytest.raises(CannInventoryError, match="exceeded the safety limit"):
        probe_dynamic_symbols(
            library,
            finder=lambda name: str(noisy_nm) if name == "nm" else None,
        )


def test_default_symbol_probe_requires_nm_or_readelf(tmp_path):
    library = tmp_path / "libopapi.so"
    library.write_bytes(b"ELF")

    with pytest.raises(CannInventoryError, match="neither nm nor readelf"):
        probe_dynamic_symbols(library, finder=lambda name: None)


def test_default_symbol_probe_never_executes_a_path_shadow(tmp_path, monkeypatch):
    library = tmp_path / "libopapi.so"
    library.write_bytes(b"ELF")
    _executable(tmp_path / "nm")
    monkeypatch.setenv("PATH", str(tmp_path))

    with pytest.raises(CannInventoryError, match="neither nm nor readelf"):
        probe_dynamic_symbols(
            library,
            runner=lambda *args, **kwargs: pytest.fail(
                "a user-controlled PATH tool must never execute"
            ),
        )
