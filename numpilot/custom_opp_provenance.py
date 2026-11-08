"""Read-only provenance receipts for installed custom CANN OPP vendors."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping


SCHEMA = "numpilot.custom-opp-provenance"
SCHEMA_VERSION = 1
_MAX_FILE_BYTES = 64 * 1024 * 1024
_MAX_VERSION_BYTES = 16 * 1024
_OPERATOR_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_SYMBOL_RE = re.compile(r"^aclnn[A-Za-z0-9_]{1,127}$")


class CustomOppProvenanceError(ValueError):
    """The custom OPP vendor cannot be safely bound to a receipt."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _read_regular(path: Path, *, maximum: int, label: str) -> bytes:
    try:
        info = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise CustomOppProvenanceError(f"{label} is unavailable") from exc
    if path.is_symlink() or not path.is_file() or info.st_size <= 0:
        raise CustomOppProvenanceError(
            f"{label} must be a non-empty regular file"
        )
    if info.st_size > maximum:
        raise CustomOppProvenanceError(f"{label} exceeds the size bound")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise CustomOppProvenanceError(f"{label} cannot be read") from exc
    if len(payload) != info.st_size:
        raise CustomOppProvenanceError(f"{label} changed while being read")
    return payload


def _safe_root(value: str | os.PathLike[str]) -> Path:
    try:
        root = Path(value).expanduser().absolute()
    except (TypeError, ValueError, OSError) as exc:
        raise CustomOppProvenanceError("vendor root is invalid") from exc
    if root.is_symlink() or not root.is_dir():
        raise CustomOppProvenanceError("vendor root must be a real directory")
    try:
        return root.resolve(strict=True)
    except OSError as exc:
        raise CustomOppProvenanceError("vendor root cannot be resolved") from exc


def _parse_version_info(payload: bytes) -> Mapping[str, str]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CustomOppProvenanceError("version.info is not UTF-8") from exc
    values: dict[str, str] = {}
    for line_number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        if "=" not in line:
            raise CustomOppProvenanceError(
                f"version.info line {line_number} is not key=value"
            )
        key, value = (item.strip() for item in line.split("=", 1))
        if not key or not value or key in values:
            raise CustomOppProvenanceError(
                f"version.info line {line_number} is invalid"
            )
        if any(ord(ch) < 32 or ord(ch) == 127 for ch in key + value):
            raise CustomOppProvenanceError("version.info contains control characters")
        values[key] = value
    if not values:
        raise CustomOppProvenanceError("version.info is empty")
    return MappingProxyType(dict(sorted(values.items())))


def _operator(value: str, label: str = "operator") -> str:
    if not isinstance(value, str) or _OPERATOR_RE.fullmatch(value) is None:
        raise CustomOppProvenanceError(f"{label} is invalid")
    return value


def _symbol(value: str, label: str) -> str:
    if not isinstance(value, str) or _SYMBOL_RE.fullmatch(value) is None:
        raise CustomOppProvenanceError(f"{label} is invalid")
    return value


@dataclass(frozen=True)
class CustomOppProvenance:
    """Immutable, hash-bound identity of one installed custom OPP vendor."""

    vendor_root: str
    operator: str
    version_info: Mapping[str, str]
    version_info_sha256: str
    header_relative_path: str
    header_sha256: str
    library_relative_path: str
    library_sha256: str
    library_size: int
    get_workspace_size: str
    execute: str
    verification_scope: str = (
        "read-only-filesystem-hash-and-header-declaration"
    )

    def __post_init__(self) -> None:
        _operator(self.operator)
        _symbol(self.get_workspace_size, "GetWorkspaceSize symbol")
        _symbol(self.execute, "execute symbol")
        if self.get_workspace_size == self.execute:
            raise CustomOppProvenanceError("two-stage symbols must be distinct")
        if not isinstance(self.version_info, Mapping) or not self.version_info:
            raise CustomOppProvenanceError("version.info metadata is invalid")
        for value, label in (
            (self.version_info_sha256, "version.info SHA-256"),
            (self.header_sha256, "header SHA-256"),
            (self.library_sha256, "library SHA-256"),
        ):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(ch not in "0123456789abcdef" for ch in value)
            ):
                raise CustomOppProvenanceError(f"{label} is invalid")
        if not isinstance(self.library_size, int) or self.library_size <= 0:
            raise CustomOppProvenanceError("library size is invalid")

    @property
    def receipt_sha256(self) -> str:
        return _sha256(_canonical(self.to_dict(include_receipt=False)))

    def to_dict(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "vendor_root": self.vendor_root,
            "operator": self.operator,
            "version_info": dict(self.version_info),
            "version_info_sha256": self.version_info_sha256,
            "header": {
                "relative_path": self.header_relative_path,
                "sha256": self.header_sha256,
            },
            "library": {
                "relative_path": self.library_relative_path,
                "sha256": self.library_sha256,
                "size": self.library_size,
            },
            "aclnn": {
                "call_style": "two-stage",
                "get_workspace_size": self.get_workspace_size,
                "execute": self.execute,
            },
            "verification_scope": self.verification_scope,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


def inspect_custom_opp(
    vendor_root: str | os.PathLike[str],
    *,
    operator: str,
    get_workspace_size: str,
    execute: str,
    expected_compiler_version: str | None = None,
) -> CustomOppProvenance:
    """Inspect one vendor directory without installing or loading it.

    The expected layout is version.info,
    op_api/include/aclnn_<operator>.h and op_api/lib/libcust_opapi.so.
    The shared object is only hashed; no ELF loading or symbol execution occurs.
    """

    root = _safe_root(vendor_root)
    op = _operator(operator)
    get_symbol = _symbol(get_workspace_size, "GetWorkspaceSize symbol")
    exec_symbol = _symbol(execute, "execute symbol")
    if get_symbol == exec_symbol or not get_symbol.endswith("GetWorkspaceSize"):
        raise CustomOppProvenanceError(
            "ACLNN symbols do not describe a distinct two-stage interface"
        )
    version_path = root / "version.info"
    header_path = root / "op_api" / "include" / f"aclnn_{op}.h"
    library_path = root / "op_api" / "lib" / "libcust_opapi.so"
    version_payload = _read_regular(
        version_path, maximum=_MAX_VERSION_BYTES, label="version.info"
    )
    header_payload = _read_regular(
        header_path, maximum=_MAX_FILE_BYTES, label="ACLNN header"
    )
    library_payload = _read_regular(
        library_path, maximum=_MAX_FILE_BYTES, label="custom OPP library"
    )
    try:
        header_text = header_payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CustomOppProvenanceError("ACLNN header is not UTF-8") from exc
    for symbol in (get_symbol, exec_symbol):
        declaration = re.compile(
            r"(?<![A-Za-z0-9_])" + re.escape(symbol) + r"\s*\("
        )
        if declaration.search(header_text) is None:
            raise CustomOppProvenanceError(
                f"ACLNN header does not declare {symbol}"
            )
    version_info = _parse_version_info(version_payload)
    if expected_compiler_version is not None:
        if (
            not isinstance(expected_compiler_version, str)
            or not expected_compiler_version.strip()
            or version_info.get("custom_opp_compiler_version")
            != expected_compiler_version
        ):
            raise CustomOppProvenanceError(
                "custom OPP compiler version does not match the expected CANN version"
            )
    return CustomOppProvenance(
        vendor_root=str(root),
        operator=op,
        version_info=version_info,
        version_info_sha256=_sha256(version_payload),
        header_relative_path=f"op_api/include/aclnn_{op}.h",
        header_sha256=_sha256(header_payload),
        library_relative_path="op_api/lib/libcust_opapi.so",
        library_sha256=_sha256(library_payload),
        library_size=len(library_payload),
        get_workspace_size=get_symbol,
        execute=exec_symbol,
    )


__all__ = [
    "SCHEMA",
    "SCHEMA_VERSION",
    "CustomOppProvenance",
    "CustomOppProvenanceError",
    "inspect_custom_opp",
]
