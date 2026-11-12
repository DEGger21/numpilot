"""Run-scoped admission of verified custom AsNumpy operators.

A custom operator produced by the gap lane is not part of the pinned AsNumpy
capability catalog.  It may only be used by a migration run when the caller
presents an explicit admission file that binds four facts to hashes:

* the operator's public identity (``numpy.<name>`` -> ``asnumpy.<name>``);
* the exact call contract admitted for this run;
* the generated adapter manifest, and
* a passing adapter build receipt (wheel hash + native test counts).

The admission is deliberately run-scoped: nothing here mutates the checked-in
catalog, and no environment, wheel or overlay is discovered implicitly.  The
file path is supplied by the caller (``--custom-operator-admission``), the file
digest becomes the admission id recorded in the migration output, and every
referenced artifact must already exist on disk with a matching SHA-256.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
from types import MappingProxyType
from typing import Any, Dict, Mapping, Tuple


SCHEMA = "numpilot.custom-operator-admission"
SCHEMA_VERSION = 1

_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_ADMISSION_BYTES = 512 * 1024
_MAX_EVIDENCE_BYTES = 8 * 1024 * 1024


class CustomOperatorAdmissionError(ValueError):
    """The admission file or one of its bound artifacts is invalid."""


@dataclass(frozen=True)
class CustomOperatorEntry:
    name: str
    source_api: str
    asnumpy_api: str
    contract: Tuple[Tuple[str, ...], int, frozenset, Tuple[str, ...]]
    adapter_manifest_path: str
    adapter_manifest_sha256: str
    adapter_build_receipt_path: str
    adapter_build_receipt_sha256: str
    wheel_sha256: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "source_api": self.source_api,
            "asnumpy_api": self.asnumpy_api,
            "contract": {
                "required": list(self.contract[0]),
                "positional_limit": self.contract[1],
                "allowed_keywords": sorted(self.contract[2]),
                "positional_names": list(self.contract[3]),
            },
            "adapter_manifest_sha256": self.adapter_manifest_sha256,
            "adapter_build_receipt_sha256": self.adapter_build_receipt_sha256,
            "wheel_sha256": self.wheel_sha256,
        }


@dataclass(frozen=True)
class CustomOperatorAdmission:
    path: str
    admission_id: str
    target_soc: str
    cann_version: str
    entries: Mapping[str, CustomOperatorEntry]

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(sorted(self.entries))

    @property
    def contracts(self) -> Mapping[str, tuple]:
        return MappingProxyType(
            {name: entry.contract for name, entry in self.entries.items()}
        )


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise CustomOperatorAdmissionError("%s must be a non-empty string" % label)
    if len(value) > 512 or any(ord(char) < 32 for char in value):
        raise CustomOperatorAdmissionError("%s is not a valid string" % label)
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if type(value) is not dict:
        raise CustomOperatorAdmissionError("%s must be an object" % label)
    return value


def _sha(value: Any, label: str) -> str:
    text = _text(value, label)
    if _SHA256.fullmatch(text) is None:
        raise CustomOperatorAdmissionError("%s must be a lowercase SHA-256" % label)
    return text


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_file(root: Path, value: Any, label: str) -> Tuple[str, Path]:
    text = _text(value, label)
    pure = PurePosixPath(text)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise CustomOperatorAdmissionError(
            "%s must be a relative path inside the admission directory" % label
        )
    target = root.joinpath(*pure.parts)
    try:
        real_root = root.resolve(strict=True)
        real_target = target.resolve(strict=True)
    except OSError as exc:
        raise CustomOperatorAdmissionError(
            "%s does not exist: %s" % (label, text)
        ) from exc
    if real_root != real_target and real_root not in real_target.parents:
        raise CustomOperatorAdmissionError(
            "%s escapes the admission directory" % label
        )
    if target.is_symlink() or not target.is_file():
        raise CustomOperatorAdmissionError("%s must be a regular non-symlink file" % label)
    if target.stat().st_size > _MAX_EVIDENCE_BYTES:
        raise CustomOperatorAdmissionError("%s is too large" % label)
    return text, target


def _evidence_file(
    root: Path, value: Any, label: str
) -> Tuple[str, str, Mapping[str, Any]]:
    evidence = _mapping(value, label)
    extra = set(evidence).difference({"path", "sha256"})
    if extra:
        raise CustomOperatorAdmissionError(
            "%s has unknown fields: %s" % (label, ", ".join(sorted(extra)))
        )
    relative, target = _relative_file(root, evidence.get("path"), label + ".path")
    expected = _sha(evidence.get("sha256"), label + ".sha256")
    actual = _sha256_file(target)
    if actual != expected:
        raise CustomOperatorAdmissionError(
            "%s SHA-256 mismatch: expected %s, found %s" % (label, expected, actual)
        )
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CustomOperatorAdmissionError("%s is not valid JSON" % label) from exc
    return relative, expected, _mapping(payload, label)


def _manifest_identity(manifest: Mapping[str, Any]) -> Tuple[str, str]:
    """Return the ``(numpy_api, asnumpy_api)`` identity an adapter manifest declares.

    Several identity shapes exist because adapters are produced by different
    lanes.  All of them are accepted here as long as they name both sides of
    the mapping explicitly; an ``api_name`` alone implies ``numpy.<api_name>``.
    """

    schema = manifest.get("schema")
    if schema == "asnumpy.adapter.manifest.v1":
        operator = manifest.get("operator")
        if isinstance(operator, dict):
            source = operator.get("source_api")
            public = operator.get("public_api")
            if isinstance(source, str) and isinstance(public, str):
                return source, public
    if schema == "numpilot.asnumpy-adapter.v1":
        source = manifest.get("qualified_api")
        public = manifest.get("public_api")
        if isinstance(source, str) and isinstance(public, str):
            return source, public
    if schema == "numpilot.asnumpy.adapter":
        operator = manifest.get("operator")
        if isinstance(operator, dict):
            source = operator.get("numpy_api")
            public = operator.get("asnumpy_api")
            if isinstance(source, str) and isinstance(public, str):
                return source, public
    if schema == "numpilot.asnumpy_adapter.operator":
        api = manifest.get("api")
        if isinstance(api, dict):
            source = api.get("numpy")
            public = api.get("asnumpy")
            if isinstance(source, str) and isinstance(public, str):
                return source, public
    if schema == "numpilot.asnumpy_adapter.test_manifest":
        operator = manifest.get("operator")
        if isinstance(operator, dict):
            source = operator.get("semantic_source")
            public = operator.get("public_api")
            if isinstance(source, str) and isinstance(public, str):
                return source, public
    if schema == "numpilot.asnumpy_adapter":
        operator = manifest.get("operator")
        if isinstance(operator, dict):
            source = operator.get("numpy_api")
            public = operator.get("asnumpy_api")
            if isinstance(source, str) and isinstance(public, str):
                return source, public
        source = manifest.get("numpy_api")
        if not isinstance(source, str):
            api_name = manifest.get("api_name")
            if isinstance(api_name, str):
                source = "numpy." + api_name
        public = manifest.get("asnumpy_api")
        if isinstance(source, str) and isinstance(public, str):
            return source, public
    raise CustomOperatorAdmissionError(
        "adapter manifest does not declare a recognized API identity"
    )


def _receipt_summary(receipt: Mapping[str, Any]) -> Tuple[str, str, int, int]:
    status = receipt.get("status")
    if status != "pass":
        raise CustomOperatorAdmissionError(
            "adapter build receipt status is not pass: %r" % (status,)
        )
    wheel = receipt.get("wheel_sha256")
    if not isinstance(wheel, str) or _SHA256.fullmatch(wheel) is None:
        raise CustomOperatorAdmissionError(
            "adapter build receipt has no valid wheel_sha256"
        )
    tests = receipt.get("tests", receipt.get("adapter_tests"))
    if not isinstance(tests, dict):
        raise CustomOperatorAdmissionError(
            "adapter build receipt has no test counts"
        )
    passed = tests.get("passed")
    failed = tests.get("failed")
    if not isinstance(passed, int) or not isinstance(failed, int):
        raise CustomOperatorAdmissionError(
            "adapter build receipt test counts must be integers"
        )
    if passed < 1 or failed != 0:
        raise CustomOperatorAdmissionError(
            "adapter build receipt must report passing tests and zero failures"
        )
    return status, wheel, passed, failed


def _contract(value: Any, label: str) -> Tuple[Tuple[str, ...], int, frozenset, Tuple[str, ...]]:
    contract = _mapping(value, label)
    extra = set(contract).difference(
        {"required", "positional_limit", "allowed_keywords", "positional_names"}
    )
    if extra:
        raise CustomOperatorAdmissionError(
            "%s has unknown fields: %s" % (label, ", ".join(sorted(extra)))
        )
    required = contract.get("required")
    keywords = contract.get("allowed_keywords")
    positional_names = contract.get("positional_names")
    limit = contract.get("positional_limit")
    if not isinstance(required, list) or not isinstance(keywords, list):
        raise CustomOperatorAdmissionError("%s lists are invalid" % label)
    if not isinstance(positional_names, list) or not isinstance(limit, int):
        raise CustomOperatorAdmissionError("%s positional data is invalid" % label)
    if limit < 1 or len(positional_names) != limit:
        raise CustomOperatorAdmissionError(
            "%s positional_names must match positional_limit" % label
        )
    for item in required:
        if item is not None and not isinstance(item, str):
            raise CustomOperatorAdmissionError("%s required names are invalid" % label)
    for item in keywords:
        if not isinstance(item, str) or _NAME.fullmatch(item) is None:
            raise CustomOperatorAdmissionError("%s keyword names are invalid" % label)
    for item in positional_names:
        if not isinstance(item, str) or _NAME.fullmatch(item) is None:
            raise CustomOperatorAdmissionError("%s positional names are invalid" % label)
    return (
        tuple(required),
        limit,
        frozenset(keywords),
        tuple(positional_names),
    )


def _entry(root: Path, value: Any, index: int) -> CustomOperatorEntry:
    label = "entries[%d]" % index
    entry = _mapping(value, label)
    extra = set(entry).difference(
        {"name", "source_api", "asnumpy_api", "contract", "evidence"}
    )
    if extra:
        raise CustomOperatorAdmissionError(
            "%s has unknown fields: %s" % (label, ", ".join(sorted(extra)))
        )
    name = _text(entry.get("name"), label + ".name")
    if _NAME.fullmatch(name) is None:
        raise CustomOperatorAdmissionError("%s.name is not a valid API name" % label)
    if entry.get("source_api") != "numpy." + name:
        raise CustomOperatorAdmissionError(
            "%s.source_api must be numpy.%s" % (label, name)
        )
    if entry.get("asnumpy_api") != "asnumpy." + name:
        raise CustomOperatorAdmissionError(
            "%s.asnumpy_api must be asnumpy.%s" % (label, name)
        )
    contract = _contract(entry.get("contract"), label + ".contract")

    evidence = _mapping(entry.get("evidence"), label + ".evidence")
    extra = set(evidence).difference(
        {"adapter_manifest", "adapter_build_receipt", "wheel_sha256"}
    )
    if extra:
        raise CustomOperatorAdmissionError(
            "%s.evidence has unknown fields: %s" % (label, ", ".join(sorted(extra)))
        )
    manifest_path, manifest_sha, manifest = _evidence_file(
        root, evidence.get("adapter_manifest"), label + ".evidence.adapter_manifest"
    )
    source_api, public_api = _manifest_identity(manifest)
    if (source_api, public_api) != ("numpy." + name, "asnumpy." + name):
        raise CustomOperatorAdmissionError(
            "%s adapter manifest identity mismatch: %s -> %s"
            % (label, source_api, public_api)
        )
    receipt_path, receipt_sha, receipt = _evidence_file(
        root,
        evidence.get("adapter_build_receipt"),
        label + ".evidence.adapter_build_receipt",
    )
    _, wheel_sha, _, _ = _receipt_summary(receipt)
    declared_wheel = _sha(
        evidence.get("wheel_sha256"), label + ".evidence.wheel_sha256"
    )
    if declared_wheel != wheel_sha:
        raise CustomOperatorAdmissionError(
            "%s wheel_sha256 does not match the build receipt" % label
        )
    return CustomOperatorEntry(
        name=name,
        source_api=source_api,
        asnumpy_api=public_api,
        contract=contract,
        adapter_manifest_path=manifest_path,
        adapter_manifest_sha256=manifest_sha,
        adapter_build_receipt_path=receipt_path,
        adapter_build_receipt_sha256=receipt_sha,
        wheel_sha256=wheel_sha,
    )


def _read_bytes(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise CustomOperatorAdmissionError(
            "admission file must be a regular non-symlink file"
        )
    size = path.stat().st_size
    if size > _MAX_ADMISSION_BYTES:
        raise CustomOperatorAdmissionError("admission file is too large")
    with open(path, "rb") as handle:
        return handle.read()


def load_custom_operator_admission(path: os.PathLike[str] | str) -> CustomOperatorAdmission:
    target = Path(path)
    raw = _read_bytes(target)
    digest = hashlib.sha256(raw).hexdigest()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CustomOperatorAdmissionError("admission file is not valid JSON") from exc
    payload = _mapping(payload, "admission")
    extra = set(payload).difference(
        {"schema", "schema_version", "target_soc", "cann_version", "entries"}
    )
    if extra:
        raise CustomOperatorAdmissionError(
            "admission has unknown fields: %s" % ", ".join(sorted(extra))
        )
    if payload.get("schema") != SCHEMA:
        raise CustomOperatorAdmissionError(
            "admission schema must be %s" % SCHEMA
        )
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise CustomOperatorAdmissionError(
            "admission schema_version must be %d" % SCHEMA_VERSION
        )
    target_soc = _text(payload.get("target_soc"), "admission.target_soc")
    cann_version = _text(payload.get("cann_version"), "admission.cann_version")
    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise CustomOperatorAdmissionError("admission.entries must be a non-empty list")
    root = target.resolve().parent
    entries: Dict[str, CustomOperatorEntry] = {}
    for index, item in enumerate(raw_entries):
        entry = _entry(root, item, index)
        if entry.name in entries:
            raise CustomOperatorAdmissionError(
                "duplicate admitted API name: %s" % entry.name
            )
        entries[entry.name] = entry
    return CustomOperatorAdmission(
        path=str(target.absolute()),
        admission_id=digest,
        target_soc=target_soc,
        cann_version=cann_version,
        entries=MappingProxyType(entries),
    )


__all__ = [
    "SCHEMA",
    "SCHEMA_VERSION",
    "CustomOperatorAdmission",
    "CustomOperatorAdmissionError",
    "CustomOperatorEntry",
    "load_custom_operator_admission",
]
