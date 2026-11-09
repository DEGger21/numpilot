"""Fail-closed contract for generated AsNumpy custom-op adapters."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

SCHEMA = "asnumpy.adapter.manifest.v1"
OVERLAY_PREFLIGHT_SCHEMA = "numpilot.asnumpy.overlay-preflight.v1"
INTEGRATION_ARCHITECTURE = "asnumpy-core-incremental-v1"
SYMBOL_RESOLUTION_STRATEGY = "explicit-handle-dlsym"
CUSTOM_OPAPI_SONAME = "libcust_opapi.so"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SYMBOL = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_SOURCE_API = re.compile(r"^operator\.[a-z][a-z0-9_]{0,63}$")
_OPERATOR_SOURCE_API = re.compile(
    r"^(?:numpy|operator)\.[a-z][a-z0-9_]{0,63}$"
)
_PUBLIC_API = re.compile(r"^asnumpy\.[a-z][a-z0-9_]{0,63}$")
_ACLNN_SYMBOL = re.compile(r"^aclnn[A-Za-z0-9_]{1,127}$")
_RELATIVE = re.compile(r"^[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*$")


class GeneratedAdapterContractError(ValueError):
    """A generated adapter contract is incomplete or unsafe."""


@dataclass(frozen=True)
class OperatorManifestContract:
    """The operator-side contract handed to the AsNumpy overlay generator.

    CANNBot output describes an operator before the source-compatible AsNumpy
    overlay exists. Keep this smaller contract separate from the full
    GeneratedAdapterContract, which is only published after overlay paths,
    custom OPP provenance and module edits have been produced.
    """

    source_api: str
    public_api: str
    target_soc: str
    cann_version: str
    aclnn_get_workspace_size: str
    aclnn_execute: str
    symbol_resolution_strategy: str
    custom_opp_library_soname: str
    global_symbol_call: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_api": self.source_api,
            "public_api": self.public_api,
            "target_soc": self.target_soc,
            "cann_version": self.cann_version,
            "aclnn_get_workspace_size": self.aclnn_get_workspace_size,
            "aclnn_execute": self.aclnn_execute,
            "symbol_resolution": {
                "strategy": self.symbol_resolution_strategy,
                "library_soname": self.custom_opp_library_soname,
                "global_symbol_call": self.global_symbol_call,
            },
        }


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


def validate_operator_manifest_for_overlay(
    payload: Mapping[str, Any] | str | bytes,
    *,
    expected_source_api: str | None = None,
    expected_public_api: str | None = None,
    expected_target_soc: str | None = None,
    expected_cann_version: str | None = None,
) -> OperatorManifestContract:
    """Validate the operator manifest before AsNumpy overlay generation.

    This is intentionally weaker than the final adapter contract: overlay
    paths, module edits and custom OPP file hashes do not exist yet. It still
    binds the operator/public identity, target toolchain and ACLNN two-stage
    symbol-resolution boundary so an injected CANNBot candidate cannot enter
    overlay generation with a stale or global-linker ABI.
    """
    if isinstance(payload, bytes):
        try:
            payload = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise GeneratedAdapterContractError("operator manifest is not UTF-8") from exc
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except (json.JSONDecodeError, RecursionError) as exc:
            raise GeneratedAdapterContractError("operator manifest is not valid JSON") from exc
    root = _obj(payload, "operator manifest")
    if root.get("schema") != SCHEMA:
        raise GeneratedAdapterContractError("operator manifest schema is unsupported")
    operator = _obj(root.get("operator"), "operator")
    source_api = _text(
        operator.get("source_api"), "operator.source_api", _OPERATOR_SOURCE_API
    )
    public_api = _text(operator.get("public_api"), "operator.public_api", _PUBLIC_API)
    symbol = source_api.split(".", 1)[1]
    if public_api != "asnumpy." + symbol:
        raise GeneratedAdapterContractError("operator/public API identity is inconsistent")
    if expected_source_api is not None and source_api != expected_source_api:
        raise GeneratedAdapterContractError("operator source API does not match GapSpec")
    if expected_public_api is not None and public_api != expected_public_api:
        raise GeneratedAdapterContractError("operator public API does not match GapSpec")
    target_soc = _text(operator.get("target_soc"), "operator.target_soc")
    cann_version = _text(operator.get("cann_version"), "operator.cann_version")
    if expected_target_soc is not None and target_soc != expected_target_soc:
        raise GeneratedAdapterContractError("operator target SoC does not match GapSpec")
    if expected_cann_version is not None and cann_version != expected_cann_version:
        raise GeneratedAdapterContractError("operator CANN version does not match GapSpec")

    mapping = _obj(root.get("name_mapping"), "name_mapping")
    python = _obj(mapping.get("python"), "name_mapping.python")
    if (
        python.get("module") != "asnumpy"
        or python.get("public_symbol") != symbol
        or python.get("public_qualified_name") != public_api
    ):
        raise GeneratedAdapterContractError("operator Python mapping is invalid")
    aclnn = _obj(mapping.get("aclnn"), "name_mapping.aclnn")
    if aclnn.get("call_style") != "two-stage":
        raise GeneratedAdapterContractError("operator ACLNN call style must be two-stage")
    get_workspace = _text(
        aclnn.get("get_workspace_size"),
        "operator.aclnn.get_workspace_size",
        _ACLNN_SYMBOL,
    )
    execute = _text(
        aclnn.get("execute"), "operator.aclnn.execute", _ACLNN_SYMBOL
    )
    if get_workspace == execute or not get_workspace.endswith("GetWorkspaceSize"):
        raise GeneratedAdapterContractError("operator ACLNN two-stage symbols are invalid")
    symbol_resolution = _obj(
        aclnn.get("symbol_resolution"), "operator.aclnn.symbol_resolution"
    )
    if (
        symbol_resolution.get("strategy") != SYMBOL_RESOLUTION_STRATEGY
        or symbol_resolution.get("library_soname") != CUSTOM_OPAPI_SONAME
        or symbol_resolution.get("global_symbol_call") is not False
    ):
        raise GeneratedAdapterContractError(
            "operator ACLNN symbols must use explicit-handle-dlsym"
        )
    cpp = mapping.get("cpp_api")
    if cpp is not None:
        cpp = _obj(cpp, "name_mapping.cpp_api")
        if (
            cpp.get("l2_get_workspace_size") != get_workspace
            or cpp.get("l2_execute") != execute
        ):
            raise GeneratedAdapterContractError(
                "operator C++/ACLNN mapping differs"
            )
    return OperatorManifestContract(
        source_api=source_api,
        public_api=public_api,
        target_soc=target_soc,
        cann_version=cann_version,
        aclnn_get_workspace_size=get_workspace,
        aclnn_execute=execute,
        symbol_resolution_strategy=SYMBOL_RESOLUTION_STRATEGY,
        custom_opp_library_soname=CUSTOM_OPAPI_SONAME,
        global_symbol_call=False,
    )


def prepare_overlay_materializer_preflight(
    operator_manifest: Mapping[str, Any] | str | bytes,
    *,
    adapter_context_plan: Mapping[str, Any],
    expected_overlay_paths: Sequence[str],
    expected_source_api: str | None = None,
    expected_public_api: str | None = None,
    expected_target_soc: str | None = None,
    expected_cann_version: str | None = None,
) -> dict[str, Any]:
    """Create a read-only receipt for the next AsNumpy overlay stage.

    The preflight proves that an operator manifest and a previously inspected
    AsNumpy context plan agree on the public API and exact overlay paths. It
    only hashes metadata and never reads, builds, or modifies the AsNumpy tree.
    """
    contract = validate_operator_manifest_for_overlay(
        operator_manifest,
        expected_source_api=expected_source_api,
        expected_public_api=expected_public_api,
        expected_target_soc=expected_target_soc,
        expected_cann_version=expected_cann_version,
    )
    if (
        not isinstance(adapter_context_plan, Mapping)
        or not isinstance(expected_overlay_paths, Sequence)
        or isinstance(expected_overlay_paths, (str, bytes))
    ):
        raise GeneratedAdapterContractError("overlay preflight inputs are invalid")
    plan = dict(adapter_context_plan)
    required_plan_fields = {
        "schema_version", "architecture", "mode", "reference_root",
        "context_sha256", "expected_overlay_paths", "total_bytes",
        "plan_sha256",
    }
    if set(plan) != required_plan_fields:
        raise GeneratedAdapterContractError("overlay context plan schema is invalid")
    if (
        plan.get("schema_version") != 1
        or (plan.get("architecture"), plan.get("mode")) not in {
            ("manifest_generator", "manifest"),
            ("legacy_manual", "manual"),
        }
        or not isinstance(plan.get("reference_root"), str)
        or not plan["reference_root"].startswith("/")
    ):
        raise GeneratedAdapterContractError("overlay context plan identity is invalid")
    raw_context = plan.get("context_sha256")
    if (
        not isinstance(raw_context, Mapping)
        or not raw_context
        or any(
            not isinstance(path, str)
            or not path
            or not isinstance(digest, str)
            or _SHA256.fullmatch(digest) is None
            for path, digest in raw_context.items()
        )
    ):
        raise GeneratedAdapterContractError("overlay context hashes are invalid")
    raw_paths = plan.get("expected_overlay_paths")
    if not isinstance(raw_paths, list) or not raw_paths:
        raise GeneratedAdapterContractError("overlay context paths are invalid")
    normalized_plan_paths = []
    for path in raw_paths:
        if (
            not isinstance(path, str)
            or str(PurePosixPath(path)) != path
            or PurePosixPath(path).is_absolute()
            or ".." in PurePosixPath(path).parts
        ):
            raise GeneratedAdapterContractError("overlay context paths are invalid")
        normalized_plan_paths.append(path)
    requested_paths = []
    for path in expected_overlay_paths:
        if (
            not isinstance(path, str)
            or str(PurePosixPath(path)) != path
            or PurePosixPath(path).is_absolute()
            or ".." in PurePosixPath(path).parts
        ):
            raise GeneratedAdapterContractError("overlay paths are invalid")
        requested_paths.append(path)
    symbol = contract.source_api.split(".", 1)[1]
    if plan["architecture"] == "manifest_generator":
        expected_path_set = {
            "csrc/custom_ops/registry/%s/op.yaml" % symbol,
            "tests/asnumpy_tests/custom_ops_tests/test_%s.py" % symbol,
        }
    else:
        expected_path_set = {
            "include/asnumpy/custom_ops/%s.hpp" % symbol,
            "csrc/custom_ops/%s.cpp" % symbol,
            "bindings/python/bind_custom_ops.cpp",
            "bindings/python/module.cpp",
            "src/asnumpy/custom_ops.py",
            "src/asnumpy/__init__.py",
            "bindings/python/CMakeLists.txt",
            "tests/asnumpy_tests/custom_ops_tests/test_%s.py" % symbol,
        }
    if (
        len(set(requested_paths)) != len(requested_paths)
        or requested_paths != normalized_plan_paths
        or set(requested_paths) != expected_path_set
    ):
        raise GeneratedAdapterContractError("overlay paths do not match source-compatible contract")
    if (
        not isinstance(plan.get("total_bytes"), int)
        or isinstance(plan.get("total_bytes"), bool)
        or plan["total_bytes"] <= 0
        or not isinstance(plan.get("plan_sha256"), str)
        or _SHA256.fullmatch(plan["plan_sha256"]) is None
    ):
        raise GeneratedAdapterContractError("overlay context plan digest is invalid")
    plan_identity = {
        "schema_version": plan["schema_version"],
        "architecture": plan["architecture"],
        "mode": plan["mode"],
        "context_sha256": dict(plan["context_sha256"]),
        "expected_overlay_paths": list(plan["expected_overlay_paths"]),
    }
    canonical_plan = json.dumps(
        plan_identity, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    plan_sha256 = hashlib.sha256(canonical_plan).hexdigest()
    if plan_sha256 != plan["plan_sha256"]:
        raise GeneratedAdapterContractError("overlay context plan digest mismatch")
    operator_payload = (
        operator_manifest.encode("utf-8")
        if isinstance(operator_manifest, str)
        else operator_manifest
        if isinstance(operator_manifest, bytes)
        else json.dumps(
            operator_manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    body = {
        "schema": OVERLAY_PREFLIGHT_SCHEMA,
        "status": "ready",
        "operator_manifest_sha256": hashlib.sha256(operator_payload).hexdigest(),
        "operator": contract.to_dict(),
        "adapter_context_plan_sha256": plan_sha256,
        "reference_root": plan["reference_root"],
        "expected_overlay_paths": requested_paths,
        "formal_component_changed": False,
        "build_performed": False,
        "activation_performed": False,
    }
    body["receipt_sha256"] = hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    return body


def prepare_source_compatible_workspace_contract(
    preflight_receipt: Mapping[str, Any] | str | bytes,
    *,
    workspace_root: str,
) -> dict[str, Any]:
    """Bind a preflight receipt to a future isolated _core workspace.

    This is a read-only materializer contract. It validates receipt integrity
    and workspace isolation, then returns the exact copy/build boundary for a
    later materializer. No directory, source file, wheel, or component is
    created or modified here.
    """
    if isinstance(preflight_receipt, bytes):
        try:
            preflight_receipt = preflight_receipt.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise GeneratedAdapterContractError(
                "overlay preflight receipt is not UTF-8"
            ) from exc
    if isinstance(preflight_receipt, str):
        try:
            preflight_receipt = json.loads(preflight_receipt)
        except (json.JSONDecodeError, RecursionError) as exc:
            raise GeneratedAdapterContractError(
                "overlay preflight receipt is not valid JSON"
            ) from exc
    if not isinstance(preflight_receipt, Mapping):
        raise GeneratedAdapterContractError("overlay preflight receipt is invalid")
    receipt = dict(preflight_receipt)
    if (
        receipt.get("schema") != OVERLAY_PREFLIGHT_SCHEMA
        or receipt.get("status") != "ready"
        or receipt.get("formal_component_changed") is not False
        or receipt.get("build_performed") is not False
        or receipt.get("activation_performed") is not False
    ):
        raise GeneratedAdapterContractError("overlay preflight receipt is not ready")
    receipt_sha = receipt.get("receipt_sha256")
    if not isinstance(receipt_sha, str) or _SHA256.fullmatch(receipt_sha) is None:
        raise GeneratedAdapterContractError("overlay preflight receipt hash is invalid")
    unsigned = dict(receipt)
    unsigned.pop("receipt_sha256", None)
    expected_sha = hashlib.sha256(
        json.dumps(
            unsigned, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    if expected_sha != receipt_sha:
        raise GeneratedAdapterContractError("overlay preflight receipt hash mismatch")
    if (
        not isinstance(workspace_root, str)
        or not workspace_root.startswith("/")
        or any(ord(c) < 32 for c in workspace_root)
    ):
        raise GeneratedAdapterContractError("isolated workspace root is invalid")
    source_root = receipt.get("reference_root")
    if (
        not isinstance(source_root, str)
        or not source_root.startswith("/")
        or source_root == workspace_root
    ):
        raise GeneratedAdapterContractError("isolated workspace/source root is invalid")
    source_path = Path(source_root).resolve(strict=False)
    workspace_path = Path(workspace_root).resolve(strict=False)
    try:
        workspace_path.relative_to(source_path)
    except ValueError:
        pass
    else:
        raise GeneratedAdapterContractError(
            "isolated workspace must not be inside the AsNumpy source root"
        )
    expected_paths = receipt.get("expected_overlay_paths")
    if (
        not isinstance(expected_paths, list)
        or not expected_paths
        or any(
            not isinstance(path, str)
            or str(PurePosixPath(path)) != path
            or PurePosixPath(path).is_absolute()
            or ".." in PurePosixPath(path).parts
            for path in expected_paths
        )
    ):
        raise GeneratedAdapterContractError(
            "isolated workspace overlay paths are invalid"
        )
    body = {
        "schema": "numpilot.asnumpy.source-compatible-workspace.v1",
        "status": "ready",
        "preflight_receipt_sha256": receipt_sha,
        "operator_manifest_sha256": receipt.get("operator_manifest_sha256"),
        "adapter_context_plan_sha256": receipt.get(
            "adapter_context_plan_sha256"
        ),
        "reference_root": source_root,
        "workspace_root": workspace_root,
        "expected_overlay_paths": list(expected_paths),
        "target_extension_module": "asnumpy._core",
        "cmake_target": "_core",
        "object_library": "custom_ops",
        "materialization_performed": False,
        "build_performed": False,
        "formal_component_changed": False,
        "activation_performed": False,
    }
    body["receipt_sha256"] = hashlib.sha256(
        json.dumps(
            body, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    return body


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
    "OperatorManifestContract",
    "GeneratedAdapterContractError",
    "INTEGRATION_ARCHITECTURE",
    "OVERLAY_PREFLIGHT_SCHEMA",
    "SYMBOL_RESOLUTION_STRATEGY",
    "CUSTOM_OPAPI_SONAME",
    "SCHEMA",
    "validate_generated_adapter_manifest",
    "validate_operator_manifest_for_overlay",
    "prepare_overlay_materializer_preflight",
    "prepare_source_compatible_workspace_contract",
]
