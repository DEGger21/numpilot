"""Read-only discovery of installed CANN ACLNN adapter candidates.

This module deliberately does not import CANN, load a shared object, execute
user code, fetch a catalog, or mutate the toolkit.  It only reads a canonical,
explicit CANN root and inspects ELF dynamic-symbol tables with ``nm`` or
``readelf``.  A positive result means that an audited *adapter candidate* is
installed; it is never a claim of NumPy semantic equivalence.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import errno
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import selectors
import shutil
import stat
import subprocess
import time
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple


EXACT_CANDIDATE = "exact_candidate"
NO_MATCH = "no_match"
INCOMPATIBLE = "incompatible"
AMBIGUOUS = "ambiguous"
ERROR = "error"

RESOLUTION_SCHEMA_VERSION = 1
CATALOG_REVISION = "installed-cann-aclnn-adapter-candidates-v5"

_RESOLUTION_STATUSES = frozenset({
    EXACT_CANDIDATE,
    NO_MATCH,
    INCOMPATIBLE,
    AMBIGUOUS,
    ERROR,
})
_QUALIFIED_API = re.compile(
    r"^(?:numpy(?:\.[A-Za-z][A-Za-z0-9_]*)+|"
    r"operator\.[A-Za-z][A-Za-z0-9_]*)$"
)
_VERSION = re.compile(r"^[0-9][A-Za-z0-9_.+-]{0,63}$")
_TARGET_SOC = re.compile(r"^ascend(?:310|610|910)[a-z0-9]{0,32}$")
_PLATFORM = re.compile(r"^(?P<arch>aarch64|x86_64)-(?P<os>linux)$")
_HEADER = re.compile(r"^aclnn_[a-z0-9_]{1,128}\.h$")
_LIBRARY = re.compile(r"^libopapi(?:_[A-Za-z0-9_]+)?\.so$")
_SYMBOL = re.compile(r"^aclnn[A-Za-z0-9_]{1,255}$")
_CANDIDATE_ID = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

_MAX_VERSION_BYTES = 64 * 1024
_MAX_HEADER_BYTES = 2 * 1024 * 1024
_MAX_LIBRARY_BYTES = 512 * 1024 * 1024
_MAX_TOOL_OUTPUT = 32 * 1024 * 1024
_MAX_ROOT_ENTRIES = 256
_SYMBOL_TIMEOUT_SECONDS = 30
_IO_CHUNK_BYTES = 1024 * 1024


def _canonical_target_soc(value: object) -> str:
    """Normalize spelling only; never infer one SoC generation from another."""

    if not isinstance(value, str) or not value or len(value) > 64:
        return ""
    canonical = re.sub(r"[-_\s]", "", value.casefold())
    return canonical if _TARGET_SOC.fullmatch(canonical) else ""


def canonical_target_soc(value: object) -> str:
    """Return a spelling-normalized SoC identity without family inference."""

    return _canonical_target_soc(value)


class CannInventoryError(RuntimeError):
    """The installed toolkit could not be inventoried safely."""


class _InventoryIssue(CannInventoryError):
    def __init__(self, status: str, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code


@dataclass(frozen=True)
class AclnnAdapterCandidate:
    """Audited API-to-ACLNN artifact mapping, not a semantic equivalence."""

    candidate_id: str
    qualified_api: str
    header_name: str
    library_priority: Tuple[str, ...]
    required_symbols: Tuple[str, ...]
    audited_cann_versions: Tuple[str, ...]
    audited_target_socs: Tuple[str, ...]
    semantic_review: Tuple[str, ...]

    def __post_init__(self) -> None:
        if not _CANDIDATE_ID.fullmatch(self.candidate_id):
            raise ValueError("ACLNN candidate_id is invalid")
        if not _QUALIFIED_API.fullmatch(self.qualified_api):
            raise ValueError("ACLNN candidate qualified_api is invalid")
        if not _HEADER.fullmatch(self.header_name):
            raise ValueError("ACLNN candidate header is invalid")
        if (
            not self.library_priority
            or len(set(self.library_priority)) != len(self.library_priority)
            or any(not _LIBRARY.fullmatch(item) for item in self.library_priority)
        ):
            raise ValueError("ACLNN candidate library priority is invalid")
        if (
            len(self.required_symbols) != 2
            or len(set(self.required_symbols)) != len(self.required_symbols)
            or any(not _SYMBOL.fullmatch(item) for item in self.required_symbols)
        ):
            raise ValueError("ACLNN candidate must bind a validated two-stage API")
        if (
            not self.audited_cann_versions
            or len(set(self.audited_cann_versions))
            != len(self.audited_cann_versions)
            or any(not _VERSION.fullmatch(item) for item in self.audited_cann_versions)
        ):
            raise ValueError("ACLNN candidate audited versions are invalid")
        if (
            not self.audited_target_socs
            or len(set(self.audited_target_socs))
            != len(self.audited_target_socs)
            or any(
                _canonical_target_soc(item) != item
                for item in self.audited_target_socs
            )
        ):
            raise ValueError("ACLNN candidate audited target SoCs are invalid")
        if (
            not self.semantic_review
            or len(self.semantic_review) > 32
            or any(
                not isinstance(item, str) or not item.strip() or len(item) > 256
                for item in self.semantic_review
            )
        ):
            raise ValueError("ACLNN candidate semantic review is invalid")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "qualified_api": self.qualified_api,
            "header_name": self.header_name,
            "library_priority": list(self.library_priority),
            "required_symbols": list(self.required_symbols),
            "audited_cann_versions": list(self.audited_cann_versions),
            "audited_target_socs": list(self.audited_target_socs),
            "semantic_review": list(self.semantic_review),
        }

    @property
    def mapping_sha256(self) -> str:
        payload = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


_MATH_LIBRARIES = ("libopapi_math.so", "libopapi.so")
_NN_LIBRARIES = ("libopapi_nn.so", "libopapi.so")
# The first reviewed AsNumPy existing-module MatMul plan already links the
# canonical monolithic ``opapi`` target.  Prefer that exact DSO so the
# materialized candidate and the target link graph can share one hash-bound
# implementation identity; ``libopapi_nn.so`` remains a fallback for future
# reviewed target layouts that explicitly link it.
_ASNUMPY_MATMUL_LIBRARIES = ("libopapi.so", "libopapi_nn.so")
# ``concatenate`` is dispatched from the existing AsNumPy ``array`` module,
# whose reviewed target link graph also names the monolithic ``opapi`` DSO.
# ``libopapi_math.so`` is retained as a second audited provider because this
# toolkit exports the same two-stage Cat symbols from both libraries.
_ASNUMPY_ARRAY_LIBRARIES = ("libopapi.so", "libopapi_math.so")
# ``cbrt`` is dispatched from the existing AsNumPy ``math`` module.  Its
# reviewed target link graph names the monolithic ``opapi`` DSO, so prefer
# that exact library; the math DSO remains a second audited provider for
# the same two-stage ``aclnnPowTensorScalar`` symbols.
_ASNUMPY_MATH_LIBRARIES = ("libopapi.so", "libopapi_math.so")
# ``invert``, ``argsort`` and ``searchsorted`` are dispatched from existing
# AsNumPy modules whose reviewed binding target links the monolithic ``opapi``
# DSO.  This toolkit exports the same two-stage symbols from both audited
# libraries, so the math DSO remains a second provider.
_ASNUMPY_EXISTING_MODULE_LIBRARIES = ("libopapi.so", "libopapi_math.so")
# ``median`` is exported only from the monolithic ``opapi`` DSO; the math DSO
# lacks the symbol and must never be offered as a provider.
_ASNUMPY_OPAPI_ONLY_LIBRARIES = ("libopapi.so",)
_AUDITED_9_0 = ("9.0.0",)
# Concatenate was audited only against the exact toolkit and device profile
# recorded in docs/concatenate-official-audit.zh-CN.md.  Neither the broader
# ``ascend910b`` family nor an unaudited CANN release may inherit this entry.
_AUDITED_9_2_BETA2 = ("9.2.0-beta.2",)
# Only the exact device profile observed by the audit is eligible.  The
# broader ``ascend910b`` family must never imply B1/B2/B3/B4 equivalence.
_AUDITED_910B4 = ("ascend910b4",)
_AUDITED_910B3 = ("ascend910b3",)


ACLNN_ADAPTER_CANDIDATES: Tuple[AclnnAdapterCandidate, ...] = (
    # Two exact audit profiles for the same ACLNN mapping.  They are kept as
    # separate catalog entries because the profile fields are independent
    # membership sets; merging them would silently admit the unaudited
    # 9.0.0/910B3 and 9.2.0-beta.2/910B4 cross combinations.
    AclnnAdapterCandidate(
        candidate_id="numpy.cbrt-aclnn-pow-tensor-scalar",
        qualified_api="numpy.cbrt",
        header_name="aclnn_pow.h",
        library_priority=_MATH_LIBRARIES,
        required_symbols=(
            "aclnnPowTensorScalarGetWorkspaceSize",
            "aclnnPowTensorScalar",
        ),
        audited_cann_versions=_AUDITED_9_0,
        audited_target_socs=_AUDITED_910B4,
        semantic_review=(
            "adapt cbrt as sign(x) * pow(abs(x), 1/3); plain fractional "
            "pow is not equivalent for negative inputs",
            "validate signed zero, NaN, infinities, integer conversion, "
            "dtype promotion, and exponent precision",
        ),
    ),
    AclnnAdapterCandidate(
        candidate_id="numpy.cbrt-aclnn-pow-tensor-scalar-910b3",
        qualified_api="numpy.cbrt",
        header_name="aclnn_pow.h",
        library_priority=_ASNUMPY_MATH_LIBRARIES,
        required_symbols=(
            "aclnnPowTensorScalarGetWorkspaceSize",
            "aclnnPowTensorScalar",
        ),
        audited_cann_versions=_AUDITED_9_2_BETA2,
        audited_target_socs=_AUDITED_910B3,
        semantic_review=(
            "adapt cbrt as sign(x) * pow(abs(x), 1/3); plain fractional "
            "pow is not equivalent for negative inputs",
            "validate signed zero, NaN, infinities, integer conversion, "
            "dtype promotion, and exponent precision",
        ),
    ),
    AclnnAdapterCandidate(
        candidate_id="numpy.arange-aclnn-arange",
        qualified_api="numpy.arange",
        header_name="aclnn_arange.h",
        library_priority=_MATH_LIBRARIES,
        required_symbols=("aclnnArangeGetWorkspaceSize", "aclnnArange"),
        audited_cann_versions=_AUDITED_9_0,
        audited_target_socs=_AUDITED_910B4,
        semantic_review=(
            "validate start/stop/step normalization and empty ranges",
            "validate dtype inference, overflow, and output-size calculation",
        ),
    ),
    AclnnAdapterCandidate(
        candidate_id="numpy.concatenate-aclnn-cat",
        qualified_api="numpy.concatenate",
        header_name="aclnn_cat.h",
        library_priority=_ASNUMPY_ARRAY_LIBRARIES,
        required_symbols=("aclnnCatGetWorkspaceSize", "aclnnCat"),
        audited_cann_versions=_AUDITED_9_2_BETA2,
        audited_target_socs=_AUDITED_910B3,
        semantic_review=(
            "map an explicit integer axis to the ACLNN Cat ``dim`` argument; "
            "axis=None and the dtype/out/casting keywords are outside this "
            "reviewed mapping and must not enter the native path",
            "qualify only non-empty sequences of float32 device tensors with "
            "one shared rank, equal non-concatenated extents, and C-contiguous "
            "storage; allocate the pre-computed output shape explicitly",
            "validate negative-axis normalization, single-input copy behavior, "
            "empty sequences, mixed dtype, mismatched shapes, and "
            "non-contiguous inputs with a NumPy oracle before integration",
        ),
    ),
    AclnnAdapterCandidate(
        candidate_id="numpy.matmul-aclnn-matmul",
        qualified_api="numpy.matmul",
        header_name="aclnn_matmul.h",
        library_priority=_ASNUMPY_MATMUL_LIBRARIES,
        required_symbols=(
            "aclnnMatmulGetWorkspaceSize",
            "aclnnMatmul",
        ),
        audited_cann_versions=_AUDITED_9_0,
        audited_target_socs=_AUDITED_910B4,
        semantic_review=(
            "qualify two device inputs as 2-D float32 with matching inner "
            "dimensions and C-contiguous storage",
            "use cubeMathType=0 (KEEP_DTYPE); admit only a two-positional-"
            "argument call and keep keyword calls on the existing host path",
            "prove 1-D, batched, broadcast, mixed-dtype, empty, non-C-"
            "contiguous, and keyword variants do not enter the native path",
        ),
    ),
    AclnnAdapterCandidate(
        candidate_id="numpy.cumsum-aclnn-cumsum",
        qualified_api="numpy.cumsum",
        header_name="aclnn_cumsum.h",
        library_priority=_MATH_LIBRARIES,
        required_symbols=("aclnnCumsumGetWorkspaceSize", "aclnnCumsum"),
        audited_cann_versions=_AUDITED_9_0,
        audited_target_socs=_AUDITED_910B4,
        semantic_review=(
            "adapt NumPy axis=None flattening before calling ACLNN",
            "validate dtype promotion, overflow, shape, and empty tensors",
        ),
    ),
    AclnnAdapterCandidate(
        candidate_id="numpy.cumprod-aclnn-cumprod",
        qualified_api="numpy.cumprod",
        header_name="aclnn_cumprod.h",
        library_priority=_MATH_LIBRARIES,
        required_symbols=("aclnnCumprodGetWorkspaceSize", "aclnnCumprod"),
        audited_cann_versions=_AUDITED_9_0,
        audited_target_socs=_AUDITED_910B4,
        semantic_review=(
            "adapt NumPy axis=None flattening before calling ACLNN",
            "validate dtype promotion, overflow, shape, and empty tensors",
        ),
    ),
    AclnnAdapterCandidate(
        candidate_id="operator.erf-aclnn-erf",
        qualified_api="operator.erf",
        header_name="aclnn_erf.h",
        library_priority=_MATH_LIBRARIES,
        required_symbols=("aclnnErfGetWorkspaceSize", "aclnnErf"),
        audited_cann_versions=_AUDITED_9_0,
        audited_target_socs=_AUDITED_910B4,
        semantic_review=(
            "bind an explicit NumPy/SciPy oracle; NumPy has no top-level erf API",
            "validate supported dtypes, approximation error, NaN, and infinities",
        ),
    ),
    AclnnAdapterCandidate(
        candidate_id="numpy.random.normal-aclnn-inplace-normal",
        qualified_api="numpy.random.normal",
        header_name="aclnn_normal.h",
        library_priority=_MATH_LIBRARIES,
        required_symbols=(
            "aclnnInplaceNormalGetWorkspaceSize",
            "aclnnInplaceNormal",
        ),
        audited_cann_versions=_AUDITED_9_0,
        audited_target_socs=_AUDITED_910B4,
        semantic_review=(
            "allocate output explicitly because the ACLNN candidate is in-place",
            "validate RNG state, seeding, broadcasting, size, and dtype semantics",
        ),
    ),
    AclnnAdapterCandidate(
        candidate_id="operator.smooth-l1-loss-aclnn-smooth-l1-loss",
        qualified_api="operator.smooth_l1_loss",
        header_name="aclnn_smooth_l1_loss.h",
        library_priority=_NN_LIBRARIES,
        required_symbols=(
            "aclnnSmoothL1LossGetWorkspaceSize",
            "aclnnSmoothL1Loss",
        ),
        audited_cann_versions=_AUDITED_9_0,
        audited_target_socs=_AUDITED_910B4,
        semantic_review=(
            "validate beta/sigma normalization and reduction-code mapping",
            "normalize scalar output shape and validate dtype/shape constraints",
        ),
    ),
    # Four audited 9.2.0-beta.2 / Ascend910B3 mappings for NumPy APIs that the
    # pinned AsNumPy export surface does not implement.  Each entry records
    # only symbol presence plus a semantic-review checklist; equivalence is
    # decided later by the deterministic API gate and the qualification
    # receipt, never by this catalog.
    AclnnAdapterCandidate(
        candidate_id="numpy.invert-aclnn-bitwise-not",
        qualified_api="numpy.invert",
        header_name="aclnn_bitwise_not.h",
        library_priority=_ASNUMPY_EXISTING_MODULE_LIBRARIES,
        required_symbols=(
            "aclnnBitwiseNotGetWorkspaceSize",
            "aclnnBitwiseNot",
        ),
        audited_cann_versions=_AUDITED_9_2_BETA2,
        audited_target_socs=_AUDITED_910B3,
        semantic_review=(
            "np.invert is bitwise NOT for integer and bool inputs; float inputs "
            "must be rejected instead of silently cast",
            "preserve input dtype and shape; validate bool logical-not versus "
            "signed integer two's-complement patterns",
        ),
    ),
    AclnnAdapterCandidate(
        candidate_id="numpy.argsort-aclnn-argsort",
        qualified_api="numpy.argsort",
        header_name="aclnn_argsort.h",
        library_priority=_ASNUMPY_EXISTING_MODULE_LIBRARIES,
        required_symbols=(
            "aclnnArgsortGetWorkspaceSize",
            "aclnnArgsort",
        ),
        audited_cann_versions=_AUDITED_9_2_BETA2,
        audited_target_socs=_AUDITED_910B3,
        semantic_review=(
            "np.argsort returns int64 indices; axis must be normalized and "
            "negative axes resolved before dispatch",
            "stability differences are allowed only for tied values; validate "
            "that gathered values are sorted and indices are a permutation",
        ),
    ),
    AclnnAdapterCandidate(
        candidate_id="numpy.searchsorted-aclnn-search-sorted",
        qualified_api="numpy.searchsorted",
        header_name="aclnn_searchsorted.h",
        library_priority=_ASNUMPY_EXISTING_MODULE_LIBRARIES,
        required_symbols=(
            "aclnnSearchSortedGetWorkspaceSize",
            "aclnnSearchSorted",
        ),
        audited_cann_versions=_AUDITED_9_2_BETA2,
        audited_target_socs=_AUDITED_910B3,
        semantic_review=(
            "np.searchsorted defaults to side='left'; the ACLNN right flag must "
            "be mapped explicitly for left versus right",
            "validate int64 index dtype, broadcasting of the values operand, "
            "and the empty-sequence boundary",
        ),
    ),
    AclnnAdapterCandidate(
        candidate_id="numpy.median-aclnn-median",
        qualified_api="numpy.median",
        header_name="aclnn_median.h",
        library_priority=_ASNUMPY_OPAPI_ONLY_LIBRARIES,
        required_symbols=(
            "aclnnMedianGetWorkspaceSize",
            "aclnnMedian",
        ),
        audited_cann_versions=_AUDITED_9_2_BETA2,
        audited_target_socs=_AUDITED_910B3,
        semantic_review=(
            "aclnnMedian reduces the flattened tensor to one value; axis, out "
            "and keepdims must be rejected rather than approximated",
            "validate even-length midpoint averaging, odd-length selection, "
            "dtype preservation and NaN-free inputs",
        ),
    ),
)


@dataclass(frozen=True)
class FileEvidence:
    relative_path: str
    sha256: str
    size: int

    def __post_init__(self) -> None:
        path = PurePosixPath(self.relative_path)
        if (
            path.is_absolute()
            or not path.parts
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError("CANN evidence path is invalid")
        if not _SHA256.fullmatch(self.sha256):
            raise ValueError("CANN evidence SHA-256 is invalid")
        if isinstance(self.size, bool) or not isinstance(self.size, int) or self.size < 1:
            raise ValueError("CANN evidence size is invalid")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "size": self.size,
        }


@dataclass(frozen=True)
class VersionEvidence:
    file: FileEvidence
    version: str
    source_field: str

    def __post_init__(self) -> None:
        if not _VERSION.fullmatch(self.version):
            raise ValueError("CANN version evidence is invalid")
        if self.source_field not in {"version", "Version"}:
            raise ValueError("CANN version source field is invalid")

    def to_dict(self) -> Dict[str, Any]:
        result = self.file.to_dict()
        result.update({
            "version": self.version,
            "source_field": self.source_field,
        })
        return result


@dataclass(frozen=True)
class HeaderEvidence:
    file: FileEvidence
    declared_symbols: Tuple[str, ...]

    def __post_init__(self) -> None:
        normalized = tuple(sorted(set(self.declared_symbols)))
        if any(not _SYMBOL.fullmatch(item) for item in normalized):
            raise ValueError("CANN header declared-symbol evidence is invalid")
        object.__setattr__(self, "declared_symbols", normalized)

    def to_dict(self) -> Dict[str, Any]:
        result = self.file.to_dict()
        result["declared_symbols"] = list(self.declared_symbols)
        return result


@dataclass(frozen=True)
class SymbolProbeOutput:
    tool: str
    symbols: Tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.tool, str) or not self.tool.strip() or len(self.tool) > 512:
            raise ValueError("dynamic symbol probe tool is invalid")
        normalized = tuple(sorted(set(self.symbols)))
        if any(not _SYMBOL.fullmatch(item) for item in normalized):
            raise ValueError("dynamic symbol probe returned an invalid ACLNN symbol")
        object.__setattr__(self, "symbols", normalized)

    @property
    def symbols_sha256(self) -> str:
        payload = "".join(item + "\n" for item in self.symbols).encode("ascii")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class LibraryEvidence:
    file: FileEvidence
    probe_tool: str
    symbols: Tuple[str, ...]
    symbols_sha256: str

    def __post_init__(self) -> None:
        normalized = tuple(sorted(set(self.symbols)))
        if any(not _SYMBOL.fullmatch(item) for item in normalized):
            raise ValueError("CANN library symbol evidence is invalid")
        payload = "".join(item + "\n" for item in normalized).encode("ascii")
        if hashlib.sha256(payload).hexdigest() != self.symbols_sha256:
            raise ValueError("CANN library symbol-table hash is inconsistent")
        if not isinstance(self.probe_tool, str) or not self.probe_tool.strip():
            raise ValueError("CANN library symbol probe is invalid")
        object.__setattr__(self, "symbols", normalized)

    def to_dict(self) -> Dict[str, Any]:
        result = self.file.to_dict()
        result.update({
            "probe_tool": self.probe_tool,
            "aclnn_symbol_count": len(self.symbols),
            "aclnn_symbols_sha256": self.symbols_sha256,
        })
        return result


@dataclass(frozen=True)
class InstalledCannInventory:
    root: str
    platform: str
    architecture: str
    operating_system: str
    version: str
    version_evidence: Tuple[VersionEvidence, ...]
    headers: Tuple[HeaderEvidence, ...]
    libraries: Tuple[LibraryEvidence, ...]

    def __post_init__(self) -> None:
        platform = _PLATFORM.fullmatch(self.platform)
        if (
            platform is None
            or platform.group("arch") != self.architecture
            or platform.group("os") != self.operating_system
            or not _VERSION.fullmatch(self.version)
        ):
            raise ValueError("installed CANN inventory identity is invalid")
        if not Path(self.root).is_absolute():
            raise ValueError("installed CANN inventory root is invalid")
        if (
            not self.version_evidence
            or any(item.version != self.version for item in self.version_evidence)
        ):
            raise ValueError("installed CANN inventory version evidence is invalid")
        all_paths = tuple(
            item.file.relative_path
            for item in (*self.version_evidence, *self.headers, *self.libraries)
        )
        if len(set(all_paths)) != len(all_paths):
            raise ValueError("installed CANN inventory repeats an evidence path")
        header_names = tuple(Path(item.file.relative_path).name for item in self.headers)
        library_names = tuple(Path(item.file.relative_path).name for item in self.libraries)
        if (
            len(set(header_names)) != len(header_names)
            or any(not _HEADER.fullmatch(item) for item in header_names)
            or len(set(library_names)) != len(library_names)
            or any(not _LIBRARY.fullmatch(item) for item in library_names)
        ):
            raise ValueError("installed CANN inventory artifact names are invalid")

    def header(self, name: str) -> Optional[HeaderEvidence]:
        matches = [item for item in self.headers if Path(item.file.relative_path).name == name]
        return matches[0] if len(matches) == 1 else None

    def library(self, name: str) -> Optional[LibraryEvidence]:
        matches = [item for item in self.libraries if Path(item.file.relative_path).name == name]
        return matches[0] if len(matches) == 1 else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "root": self.root,
            "platform": self.platform,
            "architecture": self.architecture,
            "operating_system": self.operating_system,
            "version": self.version,
            "version_evidence": [item.to_dict() for item in self.version_evidence],
            "headers": [item.to_dict() for item in self.headers],
            "libraries": [item.to_dict() for item in self.libraries],
        }


@dataclass(frozen=True)
class InstalledAclnnResolution:
    status: str
    qualified_api: str
    expected_cann_version: str
    reason_code: str
    reason: str
    expected_target_soc: str = ""
    candidate: Optional[AclnnAdapterCandidate] = None
    inventory: Optional[InstalledCannInventory] = None
    header: Optional[HeaderEvidence] = None
    library: Optional[LibraryEvidence] = None
    missing_symbols: Tuple[str, ...] = ()
    alternatives: Tuple[str, ...] = ()
    schema_version: int = RESOLUTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RESOLUTION_SCHEMA_VERSION:
            raise ValueError("unsupported installed ACLNN resolution schema")
        if self.status not in _RESOLUTION_STATUSES:
            raise ValueError("installed ACLNN resolution status is invalid")
        if self.status == EXACT_CANDIDATE and (
            self.candidate is None or self.inventory is None
            or self.header is None or self.library is None
        ):
            raise ValueError("exact ACLNN candidate requires complete evidence")
        if self.status in {EXACT_CANDIDATE, NO_MATCH} and (
            self.candidate is None
            or _canonical_target_soc(self.expected_target_soc)
            not in self.candidate.audited_target_socs
        ):
            raise ValueError(
                "positive/negative ACLNN match requires an audited target SoC"
            )

    @property
    def adapter_candidate(self) -> bool:
        return self.status == EXACT_CANDIDATE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "qualified_api": self.qualified_api,
            "expected_cann_version": self.expected_cann_version,
            "expected_target_soc": self.expected_target_soc,
            "reason_code": self.reason_code,
            "reason": self.reason,
            "adapter_candidate": self.adapter_candidate,
            "semantic_equivalence": "unverified",
            "required_next_gate": (
                "generate_and_qualify_asnumpy_adapter"
                if self.adapter_candidate else None
            ),
            "catalog_revision": CATALOG_REVISION,
            "candidate": self.candidate.to_dict() if self.candidate else None,
            "candidate_mapping_sha256": (
                self.candidate.mapping_sha256 if self.candidate else None
            ),
            "inventory": self.inventory.to_dict() if self.inventory else None,
            "selected_header": self.header.to_dict() if self.header else None,
            "selected_library": self.library.to_dict() if self.library else None,
            "required_symbols": (
                list(self.candidate.required_symbols) if self.candidate else []
            ),
            "missing_symbols": list(self.missing_symbols),
            "alternatives": list(self.alternatives),
            "provenance": "installed_cann_toolkit_read_only_inventory",
        }


SymbolProbe = Callable[[Path], SymbolProbeOutput]
ToolFinder = Callable[[str], Optional[str]]
ToolRunner = Callable[..., Any]


def _canonical_root(value: os.PathLike[str] | str) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise _InventoryIssue(ERROR, "invalid_cann_root", "CANN root must be a path")
    try:
        supplied = Path(value)
    except TypeError as exc:
        raise _InventoryIssue(
            ERROR,
            "invalid_cann_root",
            "CANN root must be a text path",
        ) from exc
    if not supplied.is_absolute():
        raise _InventoryIssue(ERROR, "invalid_cann_root", "CANN root must be absolute")
    normalized = Path(os.path.normpath(str(supplied)))
    if supplied != normalized:
        raise _InventoryIssue(
            ERROR,
            "noncanonical_cann_root",
            "CANN root must not contain dot or parent traversal components",
        )
    return normalized


def _directory_flags() -> int:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory is None:
        raise _InventoryIssue(
            ERROR,
            "nofollow_unavailable",
            "this platform cannot safely anchor CANN directories",
        )
    return (
        os.O_RDONLY
        | nofollow
        | directory
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )


@dataclass(frozen=True)
class _DirectoryHop:
    parent_fd: int
    name: str
    fd: int
    initial_stat: os.stat_result


@dataclass(frozen=True)
class _RootAnchor:
    path: Path
    fd: int


def _directory_identity(metadata: os.stat_result) -> Tuple[int, int, int]:
    """Stable object identity without volatile directory timestamps/size."""

    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(stat.S_IFMT(metadata.st_mode)),
    )


def _verify_directory_hop(hop: _DirectoryHop) -> None:
    try:
        descriptor_stat = os.fstat(hop.fd)
        entry_stat = os.stat(
            hop.name,
            dir_fd=hop.parent_fd,
            follow_symlinks=False,
        )
    except OSError as exc:
        raise _InventoryIssue(
            ERROR,
            "directory_identity_changed",
            "CANN directory ancestry changed during inventory",
        ) from exc
    expected = _directory_identity(hop.initial_stat)
    if (
        _directory_identity(descriptor_stat) != expected
        or _directory_identity(entry_stat) != expected
        or not stat.S_ISDIR(descriptor_stat.st_mode)
        or not stat.S_ISDIR(entry_stat.st_mode)
    ):
        raise _InventoryIssue(
            ERROR,
            "directory_identity_changed",
            "CANN directory ancestry changed during inventory",
        )


def _open_directory_at(parent_fd: int, name: str, *, root: bool) -> _DirectoryHop:
    if not name or name in {".", ".."} or "/" in name or "\x00" in name:
        raise _InventoryIssue(ERROR, "invalid_evidence_path", "invalid CANN path component")
    try:
        descriptor = os.open(name, _directory_flags(), dir_fd=parent_fd)
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            code = "noncanonical_cann_root" if root else "symlink_evidence_rejected"
        elif exc.errno == errno.ENOENT:
            code = "invalid_cann_root" if root else "evidence_open_failed"
        else:
            code = "directory_open_failed"
        raise _InventoryIssue(
            ERROR,
            code,
            "cannot safely open CANN directory ancestry",
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise _InventoryIssue(
                ERROR,
                "invalid_evidence_type",
                "CANN directory ancestry contains a non-directory",
            )
        hop = _DirectoryHop(
            parent_fd=parent_fd,
            name=name,
            fd=descriptor,
            initial_stat=metadata,
        )
        _verify_directory_hop(hop)
        return hop
    except BaseException:
        os.close(descriptor)
        raise


@contextmanager
def _open_root_anchor(value: os.PathLike[str] | str):
    """Anchor an absolute CANN root from ``/`` using only no-follow openat."""

    root = _canonical_root(value)
    descriptors = []
    hops = []
    try:
        slash_fd = os.open("/", _directory_flags())
        descriptors.append(slash_fd)
        current_fd = slash_fd
        for component in root.parts[1:]:
            hop = _open_directory_at(current_fd, component, root=True)
            hops.append(hop)
            descriptors.append(hop.fd)
            current_fd = hop.fd
        yield _RootAnchor(path=root, fd=current_fd)
    finally:
        try:
            for hop in hops:
                _verify_directory_hop(hop)
        finally:
            for descriptor in reversed(descriptors):
                os.close(descriptor)


@contextmanager
def _open_relative_parent(root_fd: int, parent_parts: Sequence[str]):
    """Open and pin every directory below an already pinned CANN root."""

    hops = []
    current_fd = root_fd
    try:
        for component in parent_parts:
            hop = _open_directory_at(current_fd, component, root=False)
            hops.append(hop)
            current_fd = hop.fd
        yield current_fd
    finally:
        try:
            for hop in hops:
                _verify_directory_hop(hop)
        finally:
            for hop in reversed(hops):
                os.close(hop.fd)


def _relative_parts(root: Path, path: Path) -> Tuple[str, ...]:
    try:
        parts = path.relative_to(root).parts
    except ValueError as exc:
        raise _InventoryIssue(ERROR, "path_escape", "CANN evidence path escapes root") from exc
    if not parts or any(
        not item or item in {".", ".."} or "/" in item or "\x00" in item
        for item in parts
    ):
        raise _InventoryIssue(ERROR, "invalid_evidence_path", "invalid CANN evidence path")
    return tuple(parts)


def _regular_file_exists(root: Path, root_fd: int, path: Path) -> bool:
    parts = _relative_parts(root, path)
    with _open_relative_parent(root_fd, parts[:-1]) as parent_fd:
        try:
            metadata = os.stat(
                parts[-1],
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return False
        except OSError as exc:
            raise _InventoryIssue(
                ERROR,
                "evidence_stat_failed",
                "cannot inspect CANN evidence entry",
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise _InventoryIssue(
                ERROR,
                "symlink_evidence_rejected",
                "CANN evidence path contains a symlink",
            )
        if not stat.S_ISREG(metadata.st_mode):
            raise _InventoryIssue(
                ERROR,
                "invalid_evidence_type",
                "CANN evidence is not a regular file",
            )
        return True


def _bounded_children(directory_fd: int, limit: int) -> Tuple[str, ...]:
    try:
        children = []
        with os.scandir(directory_fd) as entries:
            for index, entry in enumerate(entries):
                if index >= limit:
                    raise _InventoryIssue(
                        ERROR,
                        "directory_entry_limit",
                        "CANN directory contains too many entries to inventory safely",
                    )
                name = entry.name
                if (
                    not isinstance(name, str)
                    or not name
                    or "/" in name
                    or "\x00" in name
                ):
                    raise _InventoryIssue(
                        ERROR,
                        "invalid_evidence_path",
                        "CANN directory contains an invalid entry name",
                    )
                children.append(name)
        return tuple(sorted(children))
    except _InventoryIssue:
        raise
    except OSError as exc:
        raise _InventoryIssue(
            ERROR,
            "directory_read_failed",
            "cannot read CANN directory",
        ) from exc


@dataclass(frozen=True)
class _OpenEvidence:
    fd: int
    path: Path
    relative_path: str
    parent_fd: int
    leaf_name: str
    initial_stat: os.stat_result


def _stable_identity(metadata: os.stat_result) -> Tuple[int, ...]:
    """Fields that must remain stable while evidence is consumed.

    Access time is intentionally excluded because reading the descriptor can
    update it.  Device/inode bind the object; type, size, mtime and ctime catch
    replacement and in-place modification during hashing or symbol probing.
    """

    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_mode),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def _verify_open_evidence(opened: _OpenEvidence) -> None:
    try:
        descriptor_stat = os.fstat(opened.fd)
        path_stat = os.stat(
            opened.leaf_name,
            dir_fd=opened.parent_fd,
            follow_symlinks=False,
        )
    except OSError as exc:
        raise _InventoryIssue(
            ERROR,
            "evidence_identity_changed",
            "CANN evidence disappeared while it was being inspected",
        ) from exc
    expected = _stable_identity(opened.initial_stat)
    if (
        _stable_identity(descriptor_stat) != expected
        or _stable_identity(path_stat) != expected
        or not stat.S_ISREG(descriptor_stat.st_mode)
        or not stat.S_ISREG(path_stat.st_mode)
    ):
        raise _InventoryIssue(
            ERROR,
            "evidence_identity_changed",
            "CANN evidence changed while it was being inspected",
        )


@contextmanager
def _open_regular_evidence(
    root: Path,
    path: Path,
    maximum: int,
    *,
    root_fd: Optional[int] = None,
):
    """Open evidence through a pinned root and no-follow openat chain."""

    if root_fd is None:
        with _open_root_anchor(root) as anchor:
            with _open_regular_evidence(
                anchor.path,
                path,
                maximum,
                root_fd=anchor.fd,
            ) as opened:
                yield opened
        return
    parts = _relative_parts(root, path)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise _InventoryIssue(
            ERROR,
            "nofollow_unavailable",
            "this platform cannot safely open CANN evidence",
        )
    flags = os.O_RDONLY | nofollow
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    with _open_relative_parent(root_fd, parts[:-1]) as parent_fd:
        try:
            descriptor = os.open(parts[-1], flags, dir_fd=parent_fd)
        except OSError as exc:
            code = (
                "symlink_evidence_rejected"
                if exc.errno in {errno.ELOOP, errno.ENOTDIR}
                else "evidence_open_failed"
            )
            raise _InventoryIssue(
                ERROR,
                code,
                "cannot safely open CANN evidence",
            ) from exc
        opened = None
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise _InventoryIssue(
                    ERROR,
                    "invalid_evidence_type",
                    "CANN evidence is not a regular file",
                )
            if metadata.st_size < 1 or metadata.st_size > maximum:
                raise _InventoryIssue(
                    ERROR,
                    "invalid_evidence_size",
                    "CANN evidence file has an invalid size",
                )
            opened = _OpenEvidence(
                fd=descriptor,
                path=path,
                relative_path=PurePosixPath(*parts).as_posix(),
                parent_fd=parent_fd,
                leaf_name=parts[-1],
                initial_stat=metadata,
            )
            _verify_open_evidence(opened)
            yield opened
        finally:
            try:
                if opened is not None:
                    _verify_open_evidence(opened)
            finally:
                os.close(descriptor)


def _read_open_bounded(opened: _OpenEvidence, maximum: int) -> bytes:
    try:
        os.lseek(opened.fd, 0, os.SEEK_SET)
        chunks = []
        total = 0
        while True:
            chunk = os.read(
                opened.fd,
                min(_IO_CHUNK_BYTES, maximum + 1 - total),
            )
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum:
                raise _InventoryIssue(
                    ERROR,
                    "invalid_evidence_size",
                    "CANN evidence exceeded the bounded read limit",
                )
        if total != opened.initial_stat.st_size:
            raise _InventoryIssue(
                ERROR,
                "evidence_identity_changed",
                "CANN evidence size changed while it was being read",
            )
        return b"".join(chunks)
    except _InventoryIssue:
        raise
    except OSError as exc:
        raise _InventoryIssue(
            ERROR,
            "evidence_read_failed",
            "cannot read CANN evidence",
        ) from exc


def _hash_open_bounded(
    opened: _OpenEvidence,
    maximum: int,
) -> FileEvidence:
    digest = hashlib.sha256()
    try:
        os.lseek(opened.fd, 0, os.SEEK_SET)
        total = 0
        while True:
            chunk = os.read(
                opened.fd,
                min(_IO_CHUNK_BYTES, maximum + 1 - total),
            )
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
            if total > maximum:
                raise _InventoryIssue(
                    ERROR,
                    "invalid_evidence_size",
                    "CANN evidence exceeded the bounded hash limit",
                )
        if total != opened.initial_stat.st_size:
            raise _InventoryIssue(
                ERROR,
                "evidence_identity_changed",
                "CANN evidence size changed while it was being hashed",
            )
    except _InventoryIssue:
        raise
    except OSError as exc:
        raise _InventoryIssue(
            ERROR,
            "evidence_read_failed",
            "cannot hash CANN evidence",
        ) from exc
    return FileEvidence(
        relative_path=opened.relative_path,
        sha256=digest.hexdigest(),
        size=total,
    )


def _read_bounded(
    root: Path,
    path: Path,
    maximum: int,
    *,
    root_fd: Optional[int] = None,
) -> bytes:
    with _open_regular_evidence(
        root,
        path,
        maximum,
        root_fd=root_fd,
    ) as opened:
        return _read_open_bounded(opened, maximum)


def _file_evidence(
    root: Path,
    path: Path,
    maximum: int,
    *,
    root_fd: Optional[int] = None,
) -> FileEvidence:
    with _open_regular_evidence(
        root,
        path,
        maximum,
        root_fd=root_fd,
    ) as opened:
        return _hash_open_bounded(opened, maximum)


def _key_values(payload: bytes) -> Mapping[str, str]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise _InventoryIssue(
            ERROR,
            "invalid_version_file",
            "CANN version file is not UTF-8",
        ) from exc
    values: Dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key in values and values[key] != value:
            raise _InventoryIssue(ERROR, "invalid_version_file", "CANN version field is duplicated")
        values[key] = value
    return values


def _version_evidence(
    root: Path,
    path: Path,
    field: str,
    *,
    root_fd: Optional[int] = None,
) -> VersionEvidence:
    payload = _read_bounded(
        root,
        path,
        _MAX_VERSION_BYTES,
        root_fd=root_fd,
    )
    value = _key_values(payload).get(field, "")
    if not _VERSION.fullmatch(value):
        raise _InventoryIssue(ERROR, "invalid_cann_version", "CANN version is missing or malformed")
    return VersionEvidence(
        file=FileEvidence(
            relative_path=path.relative_to(root).as_posix(),
            sha256=hashlib.sha256(payload).hexdigest(),
            size=len(payload),
        ),
        version=value,
        source_field=field,
    )


def _safe_tool(name: str, finder: ToolFinder) -> Optional[Path]:
    value = finder(name)
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        return None
    try:
        resolved = path.resolve(strict=True)
        mode = resolved.stat().st_mode
    except (OSError, RuntimeError):
        return None
    if not stat.S_ISREG(mode) or not os.access(resolved, os.X_OK):
        return None
    # Production discovery uses shutil.which, but must never execute a
    # user-controlled PATH shadow such as /tmp/nm.  Explicit injected finders
    # are reserved for hermetic tests and embedders that supply their runner.
    if finder is shutil.which:
        trusted_roots = tuple(
            item.resolve()
            for item in (Path("/usr/bin"), Path("/bin"))
            if item.is_dir()
        )
        if not any(
            resolved == root or root in resolved.parents
            for root in trusted_roots
        ):
            return None
        metadata = resolved.stat()
        if metadata.st_uid != 0 or metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            return None
    return resolved


def _symbols_from_output(output: str) -> Tuple[str, ...]:
    symbols = set()
    for line in output.splitlines():
        if len(line) > 8192:
            continue
        for token in reversed(line.split()):
            name = token.split("@", 1)[0]
            if _SYMBOL.fullmatch(name):
                symbols.add(name)
                break
    return tuple(sorted(symbols))


def _stop_process(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.kill()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _run_bounded_symbol_command(
    command: Sequence[str],
    *,
    inherited_fd: int,
) -> Tuple[int, bytes]:
    """Run a trusted ELF inspector with a strict in-memory output ceiling."""

    try:
        process = subprocess.Popen(
            list(command),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env={"LANG": "C", "LC_ALL": "C"},
            close_fds=True,
            pass_fds=(inherited_fd,),
        )
    except OSError as exc:
        raise CannInventoryError("cannot start the dynamic symbol tool") from exc
    if process.stdout is None:
        _stop_process(process)
        raise CannInventoryError("dynamic symbol tool has no bounded output stream")
    selector = selectors.DefaultSelector()
    output = bytearray()
    deadline = time.monotonic() + _SYMBOL_TIMEOUT_SECONDS
    try:
        selector.register(process.stdout, selectors.EVENT_READ)
        stream_open = True
        while stream_open:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, _SYMBOL_TIMEOUT_SECONDS)
            events = selector.select(remaining)
            if not events:
                raise subprocess.TimeoutExpired(command, _SYMBOL_TIMEOUT_SECONDS)
            for key, _ in events:
                chunk = os.read(key.fileobj.fileno(), 64 * 1024)
                if not chunk:
                    selector.unregister(key.fileobj)
                    stream_open = False
                    break
                if len(output) + len(chunk) > _MAX_TOOL_OUTPUT:
                    raise CannInventoryError(
                        "dynamic symbol tool output exceeded the safety limit"
                    )
                output.extend(chunk)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(command, _SYMBOL_TIMEOUT_SECONDS)
        return process.wait(timeout=remaining), bytes(output)
    except BaseException:
        _stop_process(process)
        raise
    finally:
        selector.close()
        process.stdout.close()


def _run_injected_symbol_command(
    command: Sequence[str],
    *,
    inherited_fd: int,
    runner: ToolRunner,
) -> Tuple[int, bytes]:
    """Compatibility boundary for hermetic injected runners used by tests."""

    result = runner(
        list(command),
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=False,
        timeout=_SYMBOL_TIMEOUT_SECONDS,
        env={"LANG": "C", "LC_ALL": "C"},
        close_fds=True,
        pass_fds=(inherited_fd,),
    )
    output = getattr(result, "stdout", b"") or b""
    if isinstance(output, str):
        output = output.encode("utf-8", errors="replace")
    if not isinstance(output, bytes):
        raise CannInventoryError("dynamic symbol runner returned invalid output")
    if len(output) > _MAX_TOOL_OUTPUT:
        raise CannInventoryError(
            "dynamic symbol tool output exceeded the safety limit"
        )
    return int(getattr(result, "returncode", 1)), output


def _probe_dynamic_symbols_fd(
    inherited_fd: int,
    *,
    finder: ToolFinder,
    runner: ToolRunner,
) -> SymbolProbeOutput:
    try:
        metadata = os.fstat(inherited_fd)
    except OSError as exc:
        raise CannInventoryError("dynamic symbol descriptor is invalid") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise CannInventoryError("dynamic symbol descriptor is not a regular file")
    descriptor_path = Path("/proc/self/fd/%d" % inherited_fd)
    if not descriptor_path.exists():
        raise CannInventoryError("procfs descriptor access is unavailable")
    attempts = (
        ("nm", ("-D", "--defined-only", "--format=posix")),
        ("readelf", ("--dyn-syms", "--wide")),
    )
    available = False
    for name, arguments in attempts:
        executable = _safe_tool(name, finder)
        if executable is None:
            continue
        available = True
        command = [str(executable), *arguments, str(descriptor_path)]
        try:
            if runner is subprocess.run:
                code, output = _run_bounded_symbol_command(
                    command,
                    inherited_fd=inherited_fd,
                )
            else:
                code, output = _run_injected_symbol_command(
                    command,
                    inherited_fd=inherited_fd,
                    runner=runner,
                )
        except (OSError, subprocess.SubprocessError):
            continue
        if code != 0:
            continue
        return SymbolProbeOutput(
            tool="%s:%s" % (name, executable),
            symbols=_symbols_from_output(
                output.decode("utf-8", errors="replace")
            ),
        )
    if not available:
        raise CannInventoryError("neither nm nor readelf is safely available")
    raise CannInventoryError("nm/readelf could not inspect the CANN library")


def probe_dynamic_symbols(
    library: Path,
    *,
    finder: ToolFinder = shutil.which,
    runner: ToolRunner = subprocess.run,
    inherited_fd: Optional[int] = None,
) -> SymbolProbeOutput:
    """Inspect one ELF without loading it, preferring ``nm`` over ``readelf``.

    Inventory callers pass the already-hashed descriptor.  Standalone callers
    get the same no-follow and identity checks through a private descriptor.
    An injected ``runner`` is a trusted hermetic-test boundary; production uses
    the bounded ``Popen`` reader above.
    """

    if inherited_fd is not None:
        return _probe_dynamic_symbols_fd(
            inherited_fd,
            finder=finder,
            runner=runner,
        )
    target = Path(library)
    root = target.parent
    with _open_regular_evidence(root, target, _MAX_LIBRARY_BYTES) as opened:
        return _probe_dynamic_symbols_fd(
            opened.fd,
            finder=finder,
            runner=runner,
        )


def inventory_installed_cann(
    cann_root: os.PathLike[str] | str,
    *,
    candidates: Sequence[AclnnAdapterCandidate] = ACLNN_ADAPTER_CANDIDATES,
    symbol_probe: Optional[SymbolProbe] = None,
) -> InstalledCannInventory:
    """Build a bounded, targeted inventory under one explicit CANN root.

    Expected absence of a particular candidate header/library is represented
    by an incomplete inventory.  Unsafe paths, ambiguous layouts, malformed
    version files, and failed symbol probes raise :class:`CannInventoryError`.
    """

    try:
        catalog = tuple(candidates)
    except TypeError as exc:
        raise CannInventoryError(
            "ACLNN adapter candidate catalog is invalid"
        ) from exc
    if (
        not catalog
        or any(not isinstance(item, AclnnAdapterCandidate) for item in catalog)
    ):
        raise CannInventoryError("ACLNN adapter candidate catalog is invalid")
    with _open_root_anchor(cann_root) as anchor:
        return _inventory_installed_cann_anchored(
            anchor,
            catalog=catalog,
            symbol_probe=symbol_probe,
        )


def _inventory_installed_cann_anchored(
    anchor: _RootAnchor,
    *,
    catalog: Tuple[AclnnAdapterCandidate, ...],
    symbol_probe: Optional[SymbolProbe],
) -> InstalledCannInventory:
    root = anchor.path
    root_fd = anchor.fd
    platform_dirs = []
    for child_name in _bounded_children(root_fd, _MAX_ROOT_ENTRIES):
        match = _PLATFORM.fullmatch(child_name)
        if match is None:
            continue
        child = root / child_name
        install_info = child / "ascend_toolkit_install.info"
        if _regular_file_exists(root, root_fd, install_info):
            platform_dirs.append((child, match, install_info))
    if not platform_dirs:
        raise _InventoryIssue(
            ERROR,
            "missing_cann_platform",
            "CANN root has no platform install manifest",
        )
    if len(platform_dirs) != 1:
        raise _InventoryIssue(
            AMBIGUOUS,
            "ambiguous_cann_platform",
            "CANN root contains multiple installed platform manifests",
        )

    platform, platform_match, install_info = platform_dirs[0]
    install_payload = _read_bounded(
        root,
        install_info,
        _MAX_VERSION_BYTES,
        root_fd=root_fd,
    )
    install_values = _key_values(install_payload)
    install_version = install_values.get("version", "")
    if not _VERSION.fullmatch(install_version):
        raise _InventoryIssue(ERROR, "invalid_cann_version", "CANN install version is invalid")
    if install_values.get("package_name") != "Ascend-cann-toolkit":
        raise _InventoryIssue(
            ERROR,
            "invalid_cann_package",
            "CANN install manifest names another package",
        )
    if (
        install_values.get("arch") != platform_match.group("arch")
        or install_values.get("os") != platform_match.group("os")
    ):
        raise _InventoryIssue(
            ERROR,
            "platform_manifest_mismatch",
            "CANN platform manifest is inconsistent",
        )
    versions = [VersionEvidence(
        file=FileEvidence(
            relative_path=install_info.relative_to(root).as_posix(),
            sha256=hashlib.sha256(install_payload).hexdigest(),
            size=len(install_payload),
        ),
        version=install_version,
        source_field="version",
    )]
    opp_version = root / "opp" / "version.info"
    if _regular_file_exists(root, root_fd, opp_version):
        versions.append(_version_evidence(
            root,
            opp_version,
            "Version",
            root_fd=root_fd,
        ))
    if len({item.version for item in versions}) != 1:
        raise _InventoryIssue(ERROR, "conflicting_cann_versions", "CANN version evidence conflicts")

    candidate_headers = tuple(sorted({item.header_name for item in catalog}))
    candidate_symbols: Dict[str, set[str]] = {}
    for candidate in catalog:
        candidate_symbols.setdefault(candidate.header_name, set()).update(
            candidate.required_symbols
        )
    header_dir = platform / "include" / "aclnnop"
    headers = []
    for name in candidate_headers:
        path = header_dir / name
        if not _regular_file_exists(root, root_fd, path):
            continue
        payload = _read_bounded(
            root,
            path,
            _MAX_HEADER_BYTES,
            root_fd=root_fd,
        )
        declared = tuple(sorted(
            symbol
            for symbol in candidate_symbols[name]
            if re.search(
                rb"\b" + re.escape(symbol.encode("ascii")) + rb"\b",
                payload,
            )
        ))
        headers.append(HeaderEvidence(
            file=FileEvidence(
                relative_path=path.relative_to(root).as_posix(),
                sha256=hashlib.sha256(payload).hexdigest(),
                size=len(payload),
            ),
            declared_symbols=declared,
        ))

    library_names = tuple(sorted({
        name
        for candidate in catalog
        for name in candidate.library_priority
    }))
    library_dir = platform / "lib64"
    libraries = []
    for name in library_names:
        path = library_dir / name
        if not _regular_file_exists(root, root_fd, path):
            continue
        with _open_regular_evidence(
            root,
            path,
            _MAX_LIBRARY_BYTES,
            root_fd=root_fd,
        ) as opened:
            evidence = _hash_open_bounded(opened, _MAX_LIBRARY_BYTES)
            try:
                if symbol_probe is None:
                    output = probe_dynamic_symbols(
                        path,
                        inherited_fd=opened.fd,
                    )
                else:
                    # Explicitly trusted test/embedder boundary.  The open
                    # descriptor and pathname identity are still checked
                    # before and after this call by the context manager.
                    output = symbol_probe(path)
            except CannInventoryError:
                raise
            except Exception as exc:
                raise _InventoryIssue(
                    ERROR,
                    "symbol_probe_failed",
                    "dynamic symbol probe failed",
                ) from exc
            if not isinstance(output, SymbolProbeOutput):
                raise _InventoryIssue(
                    ERROR,
                    "invalid_symbol_probe_output",
                    "dynamic symbol probe returned the wrong contract",
                )
            libraries.append(LibraryEvidence(
                file=evidence,
                probe_tool=output.tool,
                symbols=output.symbols,
                symbols_sha256=output.symbols_sha256,
            ))

    return InstalledCannInventory(
        root=str(root),
        platform=platform.name,
        architecture=platform_match.group("arch"),
        operating_system=platform_match.group("os"),
        version=install_version,
        version_evidence=tuple(versions),
        headers=tuple(headers),
        libraries=tuple(libraries),
    )


def _resolution(
    status: str,
    qualified_api: str,
    expected_cann_version: str,
    reason_code: str,
    reason: str,
    *,
    expected_target_soc: str = "",
    **kwargs: Any,
) -> InstalledAclnnResolution:
    return InstalledAclnnResolution(
        status=status,
        qualified_api=qualified_api,
        expected_cann_version=expected_cann_version,
        reason_code=reason_code,
        reason=reason,
        expected_target_soc=expected_target_soc,
        **kwargs,
    )


def _candidates_for_exact_profile(
    matches: Sequence[AclnnAdapterCandidate],
    version: str,
    target_soc: str,
) -> Tuple[AclnnAdapterCandidate, ...]:
    """Select catalog entries audited for this exact version/SoC pair.

    The two audit fields are independent membership sets.  A catalog can hold
    one entry per exact audited profile for the same API without implying the
    cross product of those claims.
    """

    return tuple(
        item for item in matches
        if version in item.audited_cann_versions
        and target_soc in item.audited_target_socs
    )


def _unique_partial_profile_match(
    matches: Sequence[AclnnAdapterCandidate],
    version: str,
    target_soc: str,
) -> Optional[AclnnAdapterCandidate]:
    """Pick the only entry that can explain a non-matching exact profile.

    This preserves the specific ``candidate_target_soc_not_audited`` and
    ``candidate_version_not_audited`` diagnostics when the catalog contains
    several entries for one API.  If neither side is unique, the caller must
    report ``catalog_multiple_candidates`` instead of guessing.
    """

    version_matches = tuple(
        item for item in matches if version in item.audited_cann_versions
    )
    soc_matches = tuple(
        item for item in matches if target_soc in item.audited_target_socs
    )
    if len(version_matches) == 1 and not soc_matches:
        return version_matches[0]
    if len(soc_matches) == 1 and not version_matches:
        return soc_matches[0]
    return None


def resolve_installed_aclnn_from_inventory(
    qualified_api: str,
    *,
    inventory: InstalledCannInventory,
    expected_cann_version: str,
    expected_target_soc: Optional[str] = None,
    candidates: Sequence[AclnnAdapterCandidate] = ACLNN_ADAPTER_CANDIDATES,
) -> InstalledAclnnResolution:
    """Resolve one API from a previously captured immutable inventory.

    Callers handling multiple gaps should inventory the CANN root once and
    reuse this function.  Resolution performs no filesystem access and runs no
    subprocess, so large ``libopapi`` files are not repeatedly hashed or
    inspected.
    """

    api = qualified_api if isinstance(qualified_api, str) else ""
    version = expected_cann_version if isinstance(expected_cann_version, str) else ""
    target_soc = expected_target_soc if isinstance(expected_target_soc, str) else ""
    if (
        not _QUALIFIED_API.fullmatch(api)
        or not _VERSION.fullmatch(version)
        or not isinstance(inventory, InstalledCannInventory)
    ):
        return _resolution(
            ERROR,
            api[:256],
            version[:64],
            "invalid_request",
            "qualified API, expected CANN version, or inventory is invalid",
            expected_target_soc=target_soc[:64],
        )
    if not target_soc:
        return _resolution(
            AMBIGUOUS,
            api,
            version,
            "target_soc_required",
            "an explicit target SoC is required before ACLNN resolution",
            expected_target_soc="",
            inventory=inventory,
        )
    canonical_target_soc = _canonical_target_soc(target_soc)
    if not canonical_target_soc:
        return _resolution(
            AMBIGUOUS,
            api,
            version,
            "target_soc_unrecognized",
            "the target SoC is not a recognized canonical identifier",
            expected_target_soc=target_soc[:64],
            inventory=inventory,
        )
    if canonical_target_soc == "ascend910b":
        return _resolution(
            AMBIGUOUS,
            api,
            version,
            "target_soc_family_not_exact",
            "the Ascend 910B family does not identify an exact audited SoC",
            expected_target_soc=target_soc,
            inventory=inventory,
        )
    try:
        catalog = tuple(candidates)
        if any(not isinstance(item, AclnnAdapterCandidate) for item in catalog):
            raise TypeError
        matches = tuple(item for item in catalog if item.qualified_api == api)
    except (AttributeError, TypeError):
        return _resolution(
            ERROR,
            api,
            version,
            "invalid_candidate_catalog",
            "the ACLNN adapter candidate catalog is invalid",
            expected_target_soc=target_soc,
            inventory=inventory,
        )
    if not matches:
        return _resolution(
            AMBIGUOUS,
            api,
            version,
            "catalog_unreviewed_api",
            "the bounded ACLNN catalog has not reviewed this API",
            expected_target_soc=target_soc,
            inventory=inventory,
        )
    profile_matches = _candidates_for_exact_profile(
        matches, version, canonical_target_soc
    )
    if len(profile_matches) > 1:
        return _resolution(
            AMBIGUOUS,
            api,
            version,
            "catalog_multiple_candidates",
            "multiple audited ACLNN mappings require an explicit selection",
            expected_target_soc=target_soc,
            inventory=inventory,
            alternatives=tuple(item.candidate_id for item in profile_matches),
        )
    if profile_matches:
        candidate = profile_matches[0]
    elif len(matches) == 1:
        candidate = matches[0]
    else:
        candidate = _unique_partial_profile_match(
            matches, version, canonical_target_soc
        )
        if candidate is None:
            return _resolution(
                AMBIGUOUS,
                api,
                version,
                "catalog_multiple_candidates",
                "multiple audited ACLNN mappings require an explicit selection",
                expected_target_soc=target_soc,
                inventory=inventory,
                alternatives=tuple(item.candidate_id for item in matches),
            )
    if canonical_target_soc not in candidate.audited_target_socs:
        return _resolution(
            INCOMPATIBLE,
            api,
            version,
            "candidate_target_soc_not_audited",
            "the mapping has not been audited for this exact target SoC",
            expected_target_soc=target_soc,
            candidate=candidate,
            inventory=inventory,
        )
    if inventory.version != version:
        return _resolution(
            INCOMPATIBLE,
            api,
            version,
            "cann_version_mismatch",
            "installed CANN version does not exactly match the requested version",
            expected_target_soc=target_soc,
            candidate=candidate,
            inventory=inventory,
        )
    if inventory.version not in candidate.audited_cann_versions:
        return _resolution(
            INCOMPATIBLE,
            api,
            version,
            "candidate_version_not_audited",
            "the mapping has not been audited for this exact CANN version",
            expected_target_soc=target_soc,
            candidate=candidate,
            inventory=inventory,
        )

    header = inventory.header(candidate.header_name)
    header_symbols = set(header.declared_symbols) if header else set()
    header_complete = set(candidate.required_symbols).issubset(header_symbols)
    selected_library = None
    partial_symbols = set()
    for name in candidate.library_priority:
        library = inventory.library(name)
        if library is None:
            continue
        available = set(library.symbols)
        partial_symbols.update(available.intersection(candidate.required_symbols))
        if set(candidate.required_symbols).issubset(available):
            selected_library = library
            break
    missing_symbols = tuple(sorted(
        set(candidate.required_symbols) - partial_symbols
    ))
    if header_complete and selected_library is not None:
        return _resolution(
            EXACT_CANDIDATE,
            api,
            version,
            "header_and_dynamic_symbols_present",
            "installed artifacts match one audited adapter mapping; semantics remain unverified",
            expected_target_soc=target_soc,
            candidate=candidate,
            inventory=inventory,
            header=header,
            library=selected_library,
        )
    if header is None and not partial_symbols:
        return _resolution(
            NO_MATCH,
            api,
            version,
            "candidate_artifacts_absent",
            "the audited candidate header and required dynamic symbols are absent",
            expected_target_soc=target_soc,
            candidate=candidate,
            inventory=inventory,
            missing_symbols=tuple(candidate.required_symbols),
        )
    return _resolution(
        INCOMPATIBLE,
        api,
        version,
        "candidate_artifacts_incomplete",
        "the candidate header declarations and dynamic symbols are incomplete",
        expected_target_soc=target_soc,
        candidate=candidate,
        inventory=inventory,
        header=header,
        library=selected_library,
        missing_symbols=missing_symbols,
    )


def resolve_installed_aclnn(
    qualified_api: str,
    *,
    cann_root: os.PathLike[str] | str,
    expected_cann_version: str,
    expected_target_soc: Optional[str] = None,
    candidates: Sequence[AclnnAdapterCandidate] = ACLNN_ADAPTER_CANDIDATES,
    symbol_probe: Optional[SymbolProbe] = None,
) -> InstalledAclnnResolution:
    """Resolve one API to installed ACLNN evidence without executing it.

    ``exact_candidate`` means only that one audited mapping has matching CANN
    version, target SoC, header declarations, and exported two-stage ACLNN
    symbols.  The adapter still needs model generation plus the normal
    correctness and performance qualification gates.
    """

    api = qualified_api if isinstance(qualified_api, str) else ""
    version = expected_cann_version if isinstance(expected_cann_version, str) else ""
    target_soc = expected_target_soc if isinstance(expected_target_soc, str) else ""
    if not _QUALIFIED_API.fullmatch(api) or not _VERSION.fullmatch(version):
        return _resolution(
            ERROR,
            api[:256],
            version[:64],
            "invalid_request",
            "qualified API or expected CANN version is invalid",
            expected_target_soc=target_soc[:64],
        )
    if not target_soc:
        return _resolution(
            AMBIGUOUS,
            api,
            version,
            "target_soc_required",
            "an explicit target SoC is required before ACLNN resolution",
            expected_target_soc="",
        )
    canonical_target_soc = _canonical_target_soc(target_soc)
    if not canonical_target_soc:
        return _resolution(
            AMBIGUOUS,
            api,
            version,
            "target_soc_unrecognized",
            "the target SoC is not a recognized canonical identifier",
            expected_target_soc=target_soc[:64],
        )
    if canonical_target_soc == "ascend910b":
        return _resolution(
            AMBIGUOUS,
            api,
            version,
            "target_soc_family_not_exact",
            "the Ascend 910B family does not identify an exact audited SoC",
            expected_target_soc=target_soc,
        )
    try:
        catalog = tuple(candidates)
        if any(not isinstance(item, AclnnAdapterCandidate) for item in catalog):
            raise TypeError
        matches = tuple(item for item in catalog if item.qualified_api == api)
    except (AttributeError, TypeError):
        return _resolution(
            ERROR,
            api,
            version,
            "invalid_candidate_catalog",
            "the ACLNN adapter candidate catalog is invalid",
            expected_target_soc=target_soc,
        )
    if not matches:
        return _resolution(
            AMBIGUOUS,
            api,
            version,
            "catalog_unreviewed_api",
            "the bounded ACLNN catalog has not reviewed this API",
            expected_target_soc=target_soc,
        )
    profile_matches = _candidates_for_exact_profile(
        matches, version, canonical_target_soc
    )
    if len(profile_matches) > 1:
        return _resolution(
            AMBIGUOUS,
            api,
            version,
            "catalog_multiple_candidates",
            "multiple audited ACLNN mappings require an explicit selection",
            expected_target_soc=target_soc,
            alternatives=tuple(item.candidate_id for item in profile_matches),
        )
    if profile_matches:
        candidate = profile_matches[0]
    elif len(matches) == 1:
        candidate = matches[0]
    else:
        candidate = _unique_partial_profile_match(
            matches, version, canonical_target_soc
        )
        if candidate is None:
            return _resolution(
                AMBIGUOUS,
                api,
                version,
                "catalog_multiple_candidates",
                "multiple audited ACLNN mappings require an explicit selection",
                expected_target_soc=target_soc,
                alternatives=tuple(item.candidate_id for item in matches),
            )
    if canonical_target_soc not in candidate.audited_target_socs:
        return _resolution(
            INCOMPATIBLE,
            api,
            version,
            "candidate_target_soc_not_audited",
            "the mapping has not been audited for this exact target SoC",
            expected_target_soc=target_soc,
            candidate=candidate,
        )
    try:
        inventory = inventory_installed_cann(
            cann_root,
            candidates=catalog,
            symbol_probe=symbol_probe,
        )
    except _InventoryIssue as exc:
        return _resolution(
            exc.status,
            api,
            version,
            exc.code,
            str(exc),
            expected_target_soc=target_soc,
            candidate=candidate,
        )
    except CannInventoryError as exc:
        return _resolution(
            ERROR,
            api,
            version,
            "cann_inventory_error",
            str(exc)[:512],
            expected_target_soc=target_soc,
            candidate=candidate,
        )
    except (OSError, ValueError) as exc:
        return _resolution(
            ERROR,
            api,
            version,
            "cann_inventory_error",
            "CANN inventory failed safely: %s" % type(exc).__name__,
            expected_target_soc=target_soc,
            candidate=candidate,
        )

    return resolve_installed_aclnn_from_inventory(
        api,
        inventory=inventory,
        expected_cann_version=version,
        expected_target_soc=target_soc,
        candidates=catalog,
    )


__all__ = [
    "ACLNN_ADAPTER_CANDIDATES",
    "AMBIGUOUS",
    "AclnnAdapterCandidate",
    "CannInventoryError",
    "canonical_target_soc",
    "ERROR",
    "EXACT_CANDIDATE",
    "FileEvidence",
    "HeaderEvidence",
    "INCOMPATIBLE",
    "InstalledAclnnResolution",
    "InstalledCannInventory",
    "LibraryEvidence",
    "NO_MATCH",
    "SymbolProbeOutput",
    "VersionEvidence",
    "inventory_installed_cann",
    "probe_dynamic_symbols",
    "resolve_installed_aclnn",
    "resolve_installed_aclnn_from_inventory",
]
