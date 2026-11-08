"""Fail-closed contract for generated AsNumpy custom-op adapters."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Mapping

SCHEMA = "asnumpy.adapter.manifest.v1"
INTEGRATION_ARCHITECTURE = "asnumpy-core-incremental-v1"
SYMBOL_RESOLUTION_STRATEGY = "explicit-handle-dlsym"
CUSTOM_OPAPI_SONAME = "libcust_opapi.so"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SYMBOL = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_SOURCE_API = re.compile(r"^operator\.[a-z][a-z0-9_]{0,63}$")
_PUBLIC_API = re.compile(r"^asnumpy\.[a-z][a-z0-9_]{0,63}$")
_ACLNN_SYMBOL = re.compile(r"^aclnn[A-Za-z0-9_]{1,127}$")
_RELATIVE = re.compile(r"^[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*$")


class GeneratedAdapterContractError(ValueError):
    """A generated adapter contract is incomplete or unsafe."""


def _obj(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GeneratedAdapterContractError("%s must be an object" % label)
    return value


def _text(value: Any, label: str, pattern: re.Pattern[str] | None = None) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(c) < 32 for c in value)
        or (pattern is not None and pattern.fullmatch(value) is None)
    ):
        raise GeneratedAdapterContractError("%s is invalid" % label)
    return value


def _sha(value: Any, label: str) -> str:
    value = _text(value, label)
    if _SHA256.fullmatch(value) is None:
        raise GeneratedAdapterContractError("%s is not SHA-256" % label)
    return value


def _path(value: Any, label: str) -> str:
    value = _text(value, label)
    if (
        _RELATIVE.fullmatch(value) is None
        or value.startswith("/")
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise GeneratedAdapterContractError("%s is unsafe" % label)
    return value


def _sha_map(value: Any, label: str) -> dict[str, str]:
    value = _obj(value, label)
    if not value:
        raise GeneratedAdapterContractError("%s is empty" % label)
    result = {_path(k, label + " path"): _sha(v, label + " hash") for k, v in value.items()}
    return dict(sorted(result.items()))


@dataclass(frozen=True)
class GeneratedAdapterContract:
    source_api: str
    public_api: str
    symbol: str
    target_soc: str
    cann_version: str
    aclnn_get_workspace_size: str
    aclnn_execute: str
    symbol_resolution_strategy: str
    custom_opp_library_soname: str
    global_symbol_call: bool
    overlay_paths: tuple[str, ...]
    custom_opp_files: Mapping[str, str]
    conformance_test_ids: tuple[str, ...]
    manifest_sha256: str

    def validate_overlay(self, file_hashes: Mapping[str, str]) -> None:
        actual = _sha_map(file_hashes, "overlay file hashes")
        if set(actual) != set(self.overlay_paths):
            raise GeneratedAdapterContractError(
                "overlay files differ from generated adapter contract"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "integration_architecture": INTEGRATION_ARCHITECTURE,
            "source_api": self.source_api,
            "public_api": self.public_api,
            "symbol": self.symbol,
            "target_soc": self.target_soc,
            "cann_version": self.cann_version,
            "aclnn_get_workspace_size": self.aclnn_get_workspace_size,
            "aclnn_execute": self.aclnn_execute,
            "symbol_resolution": {
                "strategy": self.symbol_resolution_strategy,
                "library_soname": self.custom_opp_library_soname,
                "global_symbol_call": self.global_symbol_call,
            },
            "overlay_paths": list(self.overlay_paths),
            "custom_opp_files": dict(self.custom_opp_files),
            "conformance_test_ids": list(self.conformance_test_ids),
            "manifest_sha256": self.manifest_sha256,
        }


def validate_generated_adapter_manifest(
    payload: Mapping[str, Any] | str | bytes,
    *,
    expected_source_api: str | None = None,
    expected_public_api: str | None = None,
) -> GeneratedAdapterContract:
    """Validate a source-compatible incremental _core manifest."""
    if isinstance(payload, bytes):
        try:
            payload = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise GeneratedAdapterContractError("manifest is not UTF-8") from exc
    if isinstance(payload, str):
        try:
            data = json.loads(payload)
        except (json.JSONDecodeError, RecursionError) as exc:
            raise GeneratedAdapterContractError("manifest is not valid JSON") from exc
    else:
        data = payload
    root = _obj(data, "manifest")
    if root.get("schema") != SCHEMA:
        raise GeneratedAdapterContractError("manifest schema is unsupported")

    operator = _obj(root.get("operator"), "operator")
    source_api = _text(operator.get("source_api"), "operator.source_api", _SOURCE_API)
    public_api = _text(operator.get("public_api"), "operator.public_api", _PUBLIC_API)
    if expected_source_api is not None and source_api != expected_source_api:
        raise GeneratedAdapterContractError("source API does not match GapSpec")
    if expected_public_api is not None and public_api != expected_public_api:
        raise GeneratedAdapterContractError("public API does not match GapSpec")
    symbol = source_api.split(".", 1)[1]
    if public_api != "asnumpy." + symbol:
        raise GeneratedAdapterContractError("operator/public API identity is inconsistent")
    target_soc = _text(operator.get("target_soc"), "operator.target_soc")
    cann_version = _text(operator.get("cann_version"), "operator.cann_version")

    mapping = _obj(root.get("name_mapping"), "name_mapping")
    python = _obj(mapping.get("python"), "name_mapping.python")
    if (
        python.get("module") != "asnumpy"
        or python.get("public_symbol") != symbol
        or python.get("public_qualified_name") != public_api
        or python.get("private_extension_module") != "asnumpy._core"
        or python.get("private_qualified_name") != "asnumpy._core." + symbol
        or python.get("silent_numpy_fallback") is not False
    ):
        raise GeneratedAdapterContractError("Python _core mapping is invalid")

    aclnn = _obj(mapping.get("aclnn"), "name_mapping.aclnn")
    if aclnn.get("call_style") != "two-stage":
        raise GeneratedAdapterContractError("ACLNN call style must be two-stage")
    get_workspace = _text(
        aclnn.get("get_workspace_size"),
        "aclnn.get_workspace_size",
        _ACLNN_SYMBOL,
    )
    execute = _text(aclnn.get("execute"), "aclnn.execute", _ACLNN_SYMBOL)
    if get_workspace == execute or not get_workspace.endswith("GetWorkspaceSize"):
        raise GeneratedAdapterContractError("ACLNN two-stage symbols are invalid")
    symbol_resolution = _obj(aclnn.get("symbol_resolution"), "aclnn.symbol_resolution")
    if (
        symbol_resolution.get("strategy") != SYMBOL_RESOLUTION_STRATEGY
        or symbol_resolution.get("library_soname") != CUSTOM_OPAPI_SONAME
        or symbol_resolution.get("global_symbol_call") is not False
    ):
        raise GeneratedAdapterContractError(
            "ACLNN symbol resolution must use explicit-handle-dlsym"
        )

    integration = _obj(root.get("integration"), "integration")
    if (
        integration.get("architecture") != INTEGRATION_ARCHITECTURE
        or integration.get("extension_module") != "asnumpy._core"
        or integration.get("cmake_target") != "_core"
        or integration.get("cmake_mode") != "object-library"
        or integration.get("binding_submodule") != "custom_ops"
        or integration.get("binding_symbol") != "bind_custom_ops"
    ):
        raise GeneratedAdapterContractError("incremental _core integration binding is invalid")

    required = {
        "include/asnumpy/custom_ops/" + symbol + ".hpp",
        "csrc/custom_ops/" + symbol + ".cpp",
        "bindings/python/bind_custom_ops.cpp",
        "bindings/python/module.cpp",
        "src/asnumpy/custom_ops.py",
        "src/asnumpy/__init__.py",
        "bindings/python/CMakeLists.txt",
        "tests/asnumpy_tests/custom_ops_tests/test_" + symbol + ".py",
    }
    raw_paths = integration.get("overlay_paths")
    if not isinstance(raw_paths, list) or set(raw_paths) != required:
        raise GeneratedAdapterContractError("incremental overlay path set is invalid")
    overlay_paths = tuple(_path(item, "overlay path") for item in raw_paths)
    if len(overlay_paths) != len(set(overlay_paths)):
        raise GeneratedAdapterContractError("overlay paths are duplicated")

    edit = _obj(integration.get("module_edit"), "integration.module_edit")
    if edit.get("mode") != "marker-anchored":
        raise GeneratedAdapterContractError("module edit must be marker-anchored")
    for key in ("declaration", "submodule_creation", "binding_call"):
        _text(edit.get(key), "module edit " + key)

    opp = _obj(integration.get("custom_opp"), "integration.custom_opp")
    opp_files = {
        "version": _path(opp.get("version"), "custom OPP version"),
        "header": _path(opp.get("header"), "custom OPP header"),
        "library": _path(opp.get("library"), "custom OPP library"),
    }
    opp_hashes = _sha_map(opp.get("sha256"), "custom OPP hashes")
    for key, path in opp_files.items():
        if path not in opp_hashes:
            raise GeneratedAdapterContractError("custom OPP hash missing " + key)

    tests = root.get("conformance_tests")
    if not isinstance(tests, list) or not tests:
        raise GeneratedAdapterContractError("conformance_tests must be non-empty")
    ids = []
    for item in tests:
        test = _obj(item, "conformance test")
        test_id = _text(test.get("id"), "conformance test id")
        if test_id in ids or test.get("target") != public_api:
            raise GeneratedAdapterContractError("conformance test identity is invalid")
        ids.append(test_id)

    canonical = json.dumps(
        data, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return GeneratedAdapterContract(
        source_api=source_api,
        public_api=public_api,
        symbol=symbol,
        target_soc=target_soc,
        cann_version=cann_version,
        aclnn_get_workspace_size=get_workspace,
        aclnn_execute=execute,
        symbol_resolution_strategy=SYMBOL_RESOLUTION_STRATEGY,
        custom_opp_library_soname=CUSTOM_OPAPI_SONAME,
        global_symbol_call=False,
        overlay_paths=overlay_paths,
        custom_opp_files=opp_files,
        conformance_test_ids=tuple(ids),
        manifest_sha256=hashlib.sha256(canonical).hexdigest(),
    )


__all__ = [
    "GeneratedAdapterContract",
    "GeneratedAdapterContractError",
    "INTEGRATION_ARCHITECTURE",
    "SYMBOL_RESOLUTION_STRATEGY",
    "CUSTOM_OPAPI_SONAME",
    "SCHEMA",
    "validate_generated_adapter_manifest",
]
