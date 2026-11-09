import copy
import hashlib
import json
from pathlib import Path

import pytest

from numpilot.gaps.workspace_materializer import (
    WorkspaceMaterializationError,
    materialize_source_compatible_workspace,
)
from numpilot.gaps.generated_adapter import (
    GeneratedAdapterContractError,
    validate_generated_adapter_manifest,
    validate_operator_manifest_for_overlay,
    prepare_overlay_materializer_preflight,
    prepare_source_compatible_workspace_contract,
)


def _manifest():
    symbol = "softplus"
    paths = [
        "include/asnumpy/custom_ops/softplus.hpp",
        "csrc/custom_ops/softplus.cpp",
        "bindings/python/bind_custom_ops.cpp",
        "bindings/python/module.cpp",
        "src/asnumpy/custom_ops.py",
        "src/asnumpy/__init__.py",
        "bindings/python/CMakeLists.txt",
        "tests/asnumpy_tests/custom_ops_tests/test_softplus.py",
    ]
    return {
        "schema": "asnumpy.adapter.manifest.v1",
        "operator": {
            "source_api": "operator.softplus",
            "public_api": "asnumpy.softplus",
            "target_soc": "Ascend910B3",
            "cann_version": "9.2.0-beta.2",
        },
        "name_mapping": {
            "python": {
                "module": "asnumpy",
                "public_symbol": symbol,
                "public_qualified_name": "asnumpy.softplus",
                "private_extension_module": "asnumpy._core",
                "private_qualified_name": "asnumpy._core.softplus",
                "silent_numpy_fallback": False,
            },
            "aclnn": {
                "get_workspace_size": "aclnnSoftplusGetWorkspaceSize",
                "execute": "aclnnSoftplus",
                "call_style": "two-stage",
                "symbol_resolution": {
                    "strategy": "explicit-handle-dlsym",
                    "library_soname": "libcust_opapi.so",
                    "global_symbol_call": False,
                },
            },
        },
        "integration": {
            "architecture": "asnumpy-core-incremental-v1",
            "extension_module": "asnumpy._core",
            "cmake_target": "_core",
            "cmake_mode": "object-library",
            "binding_submodule": "custom_ops",
            "binding_symbol": "bind_custom_ops",
            "overlay_paths": paths,
            "module_edit": {
                "mode": "marker-anchored",
                "declaration": "void bind_custom_ops(pybind11::module_& custom_ops);",
                "submodule_creation": 'auto custom_ops = module.def_submodule("custom_ops");',
                "binding_call": "asnumpy::bind_custom_ops(custom_ops);",
            },
            "custom_opp": {
                "version": "vendors/softplus_custom/version.info",
                "header": "vendors/softplus_custom/op_api/include/aclnn_softplus.h",
                "library": "vendors/softplus_custom/op_api/lib/libcust_opapi.so",
                "sha256": {
                    "vendors/softplus_custom/version.info": "a" * 64,
                    "vendors/softplus_custom/op_api/include/aclnn_softplus.h": "b" * 64,
                    "vendors/softplus_custom/op_api/lib/libcust_opapi.so": "c" * 64,
                },
            },
        },
        "conformance_tests": [
            {"id": "mapping", "target": "asnumpy.softplus"},
            {"id": "numerical", "target": "asnumpy.softplus"},
        ],
    }


def test_source_compatible_generated_adapter_is_accepted():
    contract = validate_generated_adapter_manifest(
        _manifest(),
        expected_source_api="operator.softplus",
        expected_public_api="asnumpy.softplus",
    )
    assert contract.symbol == "softplus"
    assert contract.aclnn_execute == "aclnnSoftplus"
    assert contract.manifest_sha256
    hashes = {path: "d" * 64 for path in contract.overlay_paths}
    contract.validate_overlay(hashes)


@pytest.mark.parametrize(
    "mutator, message",
    [
        (
            lambda data: data["name_mapping"]["python"].update(
                private_extension_module="asnumpy._asnumpy",
                private_qualified_name="asnumpy._asnumpy.softplus",
            ),
            "Python _core mapping",
        ),
        (
            lambda data: data["integration"].update(
                architecture="legacy-manual",
            ),
            "integration binding",
        ),
        (
            lambda data: data["name_mapping"]["aclnn"].update(
                execute="aclnnSoftplusGetWorkspaceSize",
            ),
            "two-stage",
        ),
        (
            lambda data: data["name_mapping"]["aclnn"]["symbol_resolution"].update(
                strategy="global-linker",
            ),
            "explicit-handle-dlsym",
        ),
        (
            lambda data: data["integration"]["custom_opp"]["sha256"].pop(
                "vendors/softplus_custom/op_api/lib/libcust_opapi.so"
            ),
            "custom OPP hash missing",
        ),
    ],
)
def test_generated_adapter_contract_rejects_unsafe_variants(mutator, message):
    data = _manifest()
    mutator(data)
    with pytest.raises(GeneratedAdapterContractError, match=message):
        validate_generated_adapter_manifest(data)


def test_generated_adapter_contract_rejects_operator_identity_drift():
    data = _manifest()
    data["operator"]["source_api"] = "numpy.softplus"
    with pytest.raises(GeneratedAdapterContractError):
        validate_generated_adapter_manifest(
            data,
            expected_source_api="operator.softplus",
            expected_public_api="asnumpy.softplus",
        )


def test_generated_adapter_contract_rejects_overlay_file_drift():
    contract = validate_generated_adapter_manifest(_manifest())
    hashes = {path: "d" * 64 for path in contract.overlay_paths}
    hashes.pop("bindings/python/module.cpp")
    with pytest.raises(GeneratedAdapterContractError, match="overlay files differ"):
        contract.validate_overlay(hashes)


def test_real_softplus_legacy_manifest_is_rejected():
    data = _manifest()
    data["name_mapping"]["python"].update(
        private_extension_module="asnumpy._asnumpy",
        private_qualified_name="asnumpy._asnumpy.softplus",
    )
    with pytest.raises(GeneratedAdapterContractError, match="Python _core mapping"):
        validate_generated_adapter_manifest(data)


def _operator_manifest():
    return {
        "schema": "asnumpy.adapter.manifest.v1",
        "operator": {
            "source_api": "numpy.cbrt",
            "public_api": "asnumpy.cbrt",
            "target_soc": "Ascend910B3",
            "cann_version": "9.2.0-beta.2",
        },
        "name_mapping": {
            "python": {
                "module": "asnumpy",
                "public_symbol": "cbrt",
                "public_qualified_name": "asnumpy.cbrt",
            },
            "aclnn": {
                "call_style": "two-stage",
                "get_workspace_size": "aclnnCbrtGetWorkspaceSize",
                "execute": "aclnnCbrt",
                "symbol_resolution": {
                    "strategy": "explicit-handle-dlsym",
                    "library_soname": "libcust_opapi.so",
                    "global_symbol_call": False,
                },
            },
            "cpp_api": {
                "l2_get_workspace_size": "aclnnCbrtGetWorkspaceSize",
                "l2_execute": "aclnnCbrt",
            },
        },
    }


def test_operator_manifest_contract_bridges_injected_asc_candidate():
    contract = validate_operator_manifest_for_overlay(
        _operator_manifest(),
        expected_source_api="numpy.cbrt",
        expected_public_api="asnumpy.cbrt",
        expected_target_soc="Ascend910B3",
        expected_cann_version="9.2.0-beta.2",
    )
    assert contract.public_api == "asnumpy.cbrt"
    assert contract.aclnn_execute == "aclnnCbrt"
    assert contract.to_dict()["symbol_resolution"]["global_symbol_call"] is False


def _manifest_generator_context_plan():
    paths = [
        "csrc/custom_ops/registry/cbrt/op.yaml",
        "tests/asnumpy_tests/custom_ops_tests/test_cbrt.py",
    ]
    context_sha256 = {"csrc/asnumpy/manifest.py": "b" * 64}
    identity = {
        "schema_version": 1,
        "architecture": "manifest_generator",
        "mode": "manifest",
        "context_sha256": context_sha256,
        "expected_overlay_paths": paths,
    }
    identity_sha256 = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        **identity,
        "reference_root": "/tmp/asnumpy-reference",
        "total_bytes": 1,
        "plan_sha256": identity_sha256,
    }


def test_overlay_materializer_preflight_accepts_manifest_generator_layout():
    plan = _manifest_generator_context_plan()
    receipt = prepare_overlay_materializer_preflight(
        _operator_manifest(),
        adapter_context_plan=plan,
        expected_overlay_paths=plan["expected_overlay_paths"],
        expected_source_api="numpy.cbrt",
        expected_public_api="asnumpy.cbrt",
        expected_target_soc="Ascend910B3",
        expected_cann_version="9.2.0-beta.2",
    )
    assert receipt["status"] == "ready"
    assert receipt["expected_overlay_paths"] == plan["expected_overlay_paths"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda data: data["operator"].update(target_soc="Ascend910B4"),
            "target SoC",
        ),
        (
            lambda data: data["name_mapping"]["aclnn"]["symbol_resolution"].update(
                strategy="global-linker"
            ),
            "explicit-handle-dlsym",
        ),
        (
            lambda data: data["name_mapping"]["cpp_api"].update(
                l2_execute="aclnnOther"
            ),
            "mapping differs",
        ),
    ),
)
def test_operator_manifest_contract_rejects_overlay_boundary_drift(mutation, message):
    data = _operator_manifest()
    mutation(data)
    with pytest.raises(GeneratedAdapterContractError, match=message):
        validate_operator_manifest_for_overlay(
            data,
            expected_source_api="numpy.cbrt",
            expected_public_api="asnumpy.cbrt",
            expected_target_soc="Ascend910B3",
            expected_cann_version="9.2.0-beta.2",
        )


def _manifest_context_plan():
    paths = [
        "include/asnumpy/custom_ops/cbrt.hpp",
        "csrc/custom_ops/cbrt.cpp",
        "bindings/python/bind_custom_ops.cpp",
        "bindings/python/module.cpp",
        "src/asnumpy/custom_ops.py",
        "src/asnumpy/__init__.py",
        "bindings/python/CMakeLists.txt",
        "tests/asnumpy_tests/custom_ops_tests/test_cbrt.py",
    ]
    context_sha256 = {"CMakeLists.txt": "a" * 64}
    identity = {
        "schema_version": 1,
        "architecture": "legacy_manual",
        "mode": "manual",
        "context_sha256": context_sha256,
        "expected_overlay_paths": paths,
    }
    identity_sha256 = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        **identity,
        "reference_root": "/tmp/asnumpy-reference",
        "total_bytes": 1,
        "plan_sha256": identity_sha256,
    }


def test_overlay_materializer_preflight_is_read_only_and_hash_bound():
    receipt = prepare_overlay_materializer_preflight(
        _operator_manifest(),
        adapter_context_plan=_manifest_context_plan(),
        expected_overlay_paths=_manifest_context_plan()["expected_overlay_paths"],
        expected_source_api="numpy.cbrt",
        expected_public_api="asnumpy.cbrt",
        expected_target_soc="Ascend910B3",
        expected_cann_version="9.2.0-beta.2",
    )
    assert receipt["schema"] == "numpilot.asnumpy.overlay-preflight.v1"
    assert receipt["status"] == "ready"
    assert receipt["build_performed"] is False
    assert receipt["activation_performed"] is False
    assert len(receipt["receipt_sha256"]) == 64


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda plan: plan.update(plan_sha256="0" * 64),
            "plan digest mismatch",
        ),
        (
            lambda plan: plan["expected_overlay_paths"].pop(),
            "overlay paths do not match",
        ),
    ),
)
def test_overlay_materializer_preflight_rejects_context_drift(mutation, message):
    plan = _manifest_context_plan()
    mutation(plan)
    with pytest.raises(GeneratedAdapterContractError, match=message):
        prepare_overlay_materializer_preflight(
            _operator_manifest(),
            adapter_context_plan=plan,
            expected_overlay_paths=plan["expected_overlay_paths"],
            expected_source_api="numpy.cbrt",
            expected_public_api="asnumpy.cbrt",
            expected_target_soc="Ascend910B3",
            expected_cann_version="9.2.0-beta.2",
        )


def _preflight_receipt():
    plan = _manifest_context_plan()
    return prepare_overlay_materializer_preflight(
        _operator_manifest(),
        adapter_context_plan=plan,
        expected_overlay_paths=plan["expected_overlay_paths"],
        expected_source_api="numpy.cbrt",
        expected_public_api="asnumpy.cbrt",
        expected_target_soc="Ascend910B3",
        expected_cann_version="9.2.0-beta.2",
    )


def test_source_compatible_workspace_contract_is_read_only():
    receipt = _preflight_receipt()
    workspace = "/tmp/numpilot-isolated-cbrt-workspace"
    contract = prepare_source_compatible_workspace_contract(
        receipt, workspace_root=workspace
    )
    assert contract["status"] == "ready"
    assert contract["workspace_root"] == workspace
    assert contract["target_extension_module"] == "asnumpy._core"
    assert contract["object_library"] == "custom_ops"
    assert contract["materialization_performed"] is False
    assert contract["formal_component_changed"] is False
    assert not Path(workspace).exists()


@pytest.mark.parametrize(
    "workspace",
    (
        "/tmp/asnumpy-reference",
        "/tmp/asnumpy-reference/isolated",
    ),
)
def test_source_compatible_workspace_contract_rejects_non_isolated_root(workspace):
    receipt = _preflight_receipt()
    receipt["reference_root"] = "/tmp/asnumpy-reference"
    unsigned = dict(receipt)
    unsigned.pop("receipt_sha256")
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    with pytest.raises(GeneratedAdapterContractError, match="source root"):
        prepare_source_compatible_workspace_contract(
            receipt, workspace_root=workspace
        )


def test_source_compatible_workspace_contract_rejects_receipt_tampering():
    receipt = _preflight_receipt()
    receipt["expected_overlay_paths"] = ["csrc/other.cpp"]
    with pytest.raises(GeneratedAdapterContractError, match="hash mismatch"):
        prepare_source_compatible_workspace_contract(
            receipt, workspace_root="/tmp/isolated-cbrt"
        )


def _materializer_contract(tmp_path):
    source = tmp_path / "asnumpy-source"
    source.mkdir()
    (source / "README.md").write_text("pinned source\n", encoding="utf-8")
    plan = _manifest_context_plan()
    plan["reference_root"] = str(source)
    preflight = prepare_overlay_materializer_preflight(
        _operator_manifest(),
        adapter_context_plan=plan,
        expected_overlay_paths=plan["expected_overlay_paths"],
        expected_source_api="numpy.cbrt",
        expected_public_api="asnumpy.cbrt",
        expected_target_soc="Ascend910B3",
        expected_cann_version="9.2.0-beta.2",
    )
    workspace = tmp_path / "isolated-workspace"
    return (
        prepare_source_compatible_workspace_contract(
            preflight, workspace_root=str(workspace)
        ),
        source,
        workspace,
        plan["expected_overlay_paths"],
    )


def _write_overlay(root, paths):
    for index, relative in enumerate(paths):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("overlay-%d\n" % index, encoding="utf-8")


def test_source_compatible_workspace_materializer_copies_only_isolated_tree(tmp_path):
    contract, source, workspace, paths = _materializer_contract(tmp_path)
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    _write_overlay(overlay, paths)
    receipt = materialize_source_compatible_workspace(
        contract, overlay_root=str(overlay)
    )
    assert receipt["status"] == "materialized_unverified"
    assert receipt["materialization_performed"] is True
    assert receipt["build_performed"] is False
    assert receipt["formal_component_changed"] is False
    assert receipt["activation_performed"] is False
    assert (source / "README.md").read_text(encoding="utf-8") == "pinned source\n"
    assert (workspace / "README.md").read_text(encoding="utf-8") == "pinned source\n"
    assert (workspace / paths[0]).read_text(encoding="utf-8") == "overlay-0\n"
    assert (workspace / "numpilot-workspace-materialization.json").is_file()


def test_source_compatible_workspace_materializer_preserves_internal_symlink(tmp_path):
    contract, source, workspace, paths = _materializer_contract(tmp_path)
    (source / "CLAUDE.md").write_text("instructions\n", encoding="utf-8")
    (source / "AGENTS.md").symlink_to("CLAUDE.md")
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    _write_overlay(overlay, paths)
    receipt = materialize_source_compatible_workspace(
        contract, overlay_root=str(overlay)
    )
    assert receipt["workspace_tree_symlinks"] == {"AGENTS.md": "CLAUDE.md"}
    assert (workspace / "AGENTS.md").is_symlink()
    assert (workspace / "AGENTS.md").read_text(encoding="utf-8") == "instructions\n"
    assert (source / "AGENTS.md").is_symlink()


def test_source_compatible_workspace_materializer_rejects_overlay_drift(tmp_path):
    contract, _source, workspace, paths = _materializer_contract(tmp_path)
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    _write_overlay(overlay, paths)
    (overlay / "unexpected.txt").write_text("drift\n", encoding="utf-8")
    with pytest.raises(WorkspaceMaterializationError, match="overlay files differ"):
        materialize_source_compatible_workspace(
            contract, overlay_root=str(overlay)
        )
    assert not workspace.exists()


def test_source_compatible_workspace_materializer_rejects_source_symlink(tmp_path):
    contract, source, workspace, paths = _materializer_contract(tmp_path)
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    _write_overlay(overlay, paths)
    (source / "escape").symlink_to("/tmp")
    with pytest.raises(WorkspaceMaterializationError, match="symlink"):
        materialize_source_compatible_workspace(
            contract, overlay_root=str(overlay)
        )
    assert not workspace.exists()
