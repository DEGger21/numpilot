import copy

import pytest

from numpilot.gaps.generated_adapter import (
    GeneratedAdapterContractError,
    validate_generated_adapter_manifest,
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
