"""Safe providers for generating an unverified Ascend C operator draft."""

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
from typing import Callable, Dict, List, Optional, Tuple

from numpilot.asnumpy_skill import (
    ASNUMPY_INTEGRATION_SKILL_FILES,
    ASNUMPY_OFFICIAL_ADAPTER_SKILL_FILES,
)
from numpilot.cannbot_skills import (
    CANNBOT_GENERATION_SKILL_FILES,
    CANNBOT_REPAIR_SKILL_FILES,
)
from numpilot.gaps.models import (
    ARTIFACT_SCHEMA_VERSION,
    GapSpec,
    OperatorDraft,
)
from numpilot import model_contract as _model_contract
from numpilot.llm_config import (
    OPENAI_CHAT_COMPLETIONS_PROTOCOL,
)
from numpilot.gaps.generated_adapter import (
    CUSTOM_OPAPI_SONAME,
    SYMBOL_RESOLUTION_STRATEGY,
    GeneratedAdapterContractError,
    validate_operator_manifest_for_overlay,
)


class DraftGenerationError(RuntimeError):
    """The model response is invalid or cannot be published safely."""

    def __init__(self, message: str):
        # These fields are deliberately non-authoritative telemetry only.  A
        # provider fills them after a bounded attempt chain; callers must still
        # rely on the signed artifact/receipt for any claim of remote work.
        # Initialising them here keeps failure handling predictable for local
        # validation errors and for custom provider implementations.
        self.request_count = 0
        self.receipt_count = 0
        self.actual_remote_call = False
        super().__init__(message)


class DraftRepairError(DraftGenerationError):
    """A bounded remote operation failed with separate request/receipt counts."""

    def __init__(
        self,
        message: str,
        *,
        receipt_count: int = 0,
        request_count: Optional[int] = None,
    ):
        if request_count is None:
            # Preserve the historical constructor behavior for callers that
            # only supplied a receipt count.  Provider call sites now pass the
            # independently observed transport-attempt count explicitly.
            request_count = receipt_count
        if (
            isinstance(receipt_count, bool)
            or not isinstance(receipt_count, int)
            or receipt_count < 0
            or isinstance(request_count, bool)
            or not isinstance(request_count, int)
            or request_count < receipt_count
        ):
            raise ValueError("invalid remote request/receipt counts")
        super().__init__(message)
        self.receipt_count = receipt_count
        self.request_count = request_count
        self.actual_remote_call = request_count > 0


_ALLOWED_ROOT_FILES = frozenset({
    "CMakeLists.txt",
    "README.md",
    "build.sh",
})
_ALLOWED_TOP_DIRS = frozenset({
    "op_host",
    "op_kernel",
    "op_api",
    "csrc",
    "tests",
    "asnumpy_adapter",
    "docs",
})
# An AsNumpy adapter is an overlay for a pre-existing source tree, not an
# operator-project draft.  Keep its writable surface separate so supporting
# the legacy/manual and existing-module layouts cannot silently broaden the
# paths accepted from ordinary operator generation.
_ASNUMPY_ADAPTER_ROOT_FILES = frozenset({
    "CMakeLists.txt",
})
_ASNUMPY_ADAPTER_TOP_DIRS = frozenset({
    "bindings",
    "csrc",
    "include",
    "src",
    "tests",
})
# Manifest discovery reads these directories as hash-bound prompt context,
# but generated/repaired overlays must never be allowed to write them.
_ASNUMPY_ADAPTER_CONTEXT_TOP_DIRS = (
    _ASNUMPY_ADAPTER_TOP_DIRS | frozenset({"cmake", "scripts"})
)
_ALLOWED_SUFFIXES = frozenset({
    ".asc",
    ".cmake",
    ".cpp",
    ".h",
    ".hpp",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".txt",
    ".yaml",
    ".yml",
})
_MAX_FILES = 64
_MAX_TOTAL_BYTES = 2 * 1024 * 1024
_MAX_RESPONSE_BYTES = _MAX_TOTAL_BYTES + 256 * 1024
_MAX_SUMMARY_CHARS = 4096
_MAX_PATH_CHARS = 240
_MAX_PATH_DEPTH = 8
# Direct generation has a fixed 17-file plan and permits three bounded
# requests per file.  Repair chains are independently bounded to eight
# rounds, each of which may repair all 16 admitted paths with three requests.
# ``repair_to_candidate`` reserves that round's worst case before any remote
# call, so every candidate it publishes remains sealable.
_MAX_FRESH_TRANSPORT_ATTEMPTS = 51
_MAX_REPAIR_PATHS = 16
_MAX_REPAIR_CHAIN_DEPTH = 8
_MAX_REPAIR_ATTEMPTS_PER_ROUND = _MAX_REPAIR_PATHS * 3
_MAX_TRANSPORT_ATTEMPTS = (
    _MAX_FRESH_TRANSPORT_ATTEMPTS
    + _MAX_REPAIR_CHAIN_DEPTH * _MAX_REPAIR_ATTEMPTS_PER_ROUND
)
_MAX_METADATA_BYTES = 2 * 1024 * 1024
_MAX_OPERATOR_TREE_ENTRIES = 256
_MAX_ADAPTER_CHAIN_DEPTH = 32
_MAX_ADAPTER_FILES = 8
_MAX_ADAPTER_TREE_ENTRIES = 64
_MAX_ADAPTER_FILE_BYTES = 512 * 1024
_MAX_ADAPTER_TOTAL_BYTES = 2 * 1024 * 1024
# Exact-edit responses are model output, not a bounded chat answer.  Keep them
# within the same per-file/total envelope as the resulting overlay instead of
# imposing a smaller framework-specific completion ceiling.
_MAX_OFFICIAL_EXACT_EDIT_RESPONSE_BYTES = _MAX_ADAPTER_TOTAL_BYTES
_MAX_OFFICIAL_EXACT_EDIT_FRAGMENT_BYTES = _MAX_ADAPTER_FILE_BYTES
_MAX_OFFICIAL_EXACT_EDITS = 256
_MAX_OFFICIAL_EXACT_EDIT_PROMPT_BYTES = _MAX_ADAPTER_TOTAL_BYTES
_MAX_OFFICIAL_CONTEXT_FILES = 12
# NumPilot does not impose a completion-token ceiling on operator generation
# or repair.  The request omits ``max_tokens`` unless the operator explicitly
# sets NUMPILOT_LLM_MAX_TOKENS, so the connected model/gateway applies its own
# maximum.  Thinking mode is likewise left at the provider default.


_OFFICIAL_MATMUL_LINK_BLOCK = (
    "target_link_libraries(_core\n"
    "    PUBLIC\n"
    "    ascendcl\n"
    "    runtime\n"
    "    nnopbase\n"
    "    opapi\n"
    "    Python::NumPy\n"
    ")"
)
_OFFICIAL_EXACT_EDIT_TARGET_WINDOWS = {
    "csrc/linalg/product.cpp": (
        (
            "matmul_implementation",
            "NPUArray Matmul(const NPUArray& x1, const NPUArray& x2) {",
            "\nNPUArray Einsum(",
        ),
    ),
    # The dispatch module is already small and every helper/import may be
    # relevant to preserving the compatibility fallback.
    "src/asnumpy/linalg/direct.py": (("complete_dispatch_module", None, None),),
    "tests/asnumpy_tests/linalg_tests/test_product.py": (
        (
            "imports_and_helpers",
            None,
            "\n# ==========================================================================\n"
            "# 1. dot 点积测试",
        ),
        (
            "matmul_tests",
            "# ==========================================================================\n"
            "# 5. matmul 矩阵乘法测试",
            "\n# ==========================================================================\n"
            "# 6. matrix_power 矩阵幂测试",
        ),
    ),
    # np.concatenate is implemented in the existing array module.  Only the
    # include block and the tail after Linspace are editable in basic.cpp, so
    # the model cannot rewrite unrelated creation functions.
    "include/asnumpy/array/basic.hpp": (
        ("complete_array_header", None, None),
    ),
    "csrc/array/basic.cpp": (
        (
            "includes",
            "#include <asnumpy/array/basic.hpp>",
            "namespace asnumpy {",
        ),
        (
            "concatenate_implementation",
            "NPUArray Linspace(const py::object& start, "
            "const py::object& end, const py::object& steps, "
            "const py::object& dtype) {",
            None,
        ),
    ),
    "bindings/python/bind_array.cpp": (
        ("complete_binding_module", None, None),
    ),
    "src/asnumpy/array.py": (
        ("complete_array_module", None, None),
    ),
    "src/asnumpy/__init__.py": (
        (
            "type_checking_array_imports",
            "    from .array import (",
            "    from .io import load, save, savez, savez_compressed",
        ),
        (
            "lazy_array_exports",
            "    # .array\n",
            "    # .linalg\n",
        ),
    ),
    "tests/asnumpy_tests/creation_tests/test_basic.py": (
        ("complete_creation_test_module", None, None),
    ),
}
_GENERATION_FIELDS_V1 = frozenset({
    "mode",
    "provider",
    "model",
    "configuration_source",
    "origin_action",
    "attempt_count",
    "transport_attempt_count",
    "successful_http_response_count",
    "content_response_count",
    "response_sha256",
    "remote_receipts",
    "local_validation",
    "fallback",
})
_GENERATION_FIELDS_V2 = _GENERATION_FIELDS_V1 | frozenset({
    "protocol",
    "response_model_aliases",
})
_ARTIFACT_FIELDS = frozenset({
    "schema_version",
    "artifact_id",
    "gap_id",
    "workspace",
    "summary",
    "status",
    "files",
    "generation",
    "gates",
})
_REPAIR_REPORT_FIELDS_V1 = frozenset({
    "schema_version",
    "status",
    "source_artifact_id",
    "source_workspace",
    "signed_source_workspace",
    "parent_repair_report_sha256",
    "gap_id",
    "model",
    "provider",
    "configuration_source",
    "attempt_count",
    "paths",
    "before_sha256",
    "after_sha256",
    "candidate_files_sha256",
    "diagnostic_sha256",
    "generation_request_sha256",
    "skill_bundle_sha256",
    "cannbot_repair_skill_files",
    "response_sha256",
    "remote_receipts",
    "actual_remote_call",
    "gates",
})
_REPAIR_REPORT_FIELDS_V2 = _REPAIR_REPORT_FIELDS_V1 | frozenset({
    "protocol",
    "response_model_aliases",
})
_GENERATED_ADAPTER_REPORT_FIELDS_V2 = frozenset({
    "schema_version",
    "status",
    "adapter_artifact_id",
    "source_artifact_id",
    "model",
    "provider",
    "configuration_source",
    "attempt_count",
    "files_sha256",
    "adapter_contract_sha256",
    "asnumpy_context_sha256",
    "skill_bundle_sha256",
    "integration_skill_files",
    "response_sha256",
    "remote_receipts",
    "actual_remote_call",
    "gates",
})
_GENERATED_ADAPTER_REPORT_FIELDS_V3 = (
    _GENERATED_ADAPTER_REPORT_FIELDS_V2 | frozenset({
        "protocol",
        "response_model_aliases",
    })
)
_GENERATED_ADAPTER_REPORT_FIELDS_V4 = (
    _GENERATED_ADAPTER_REPORT_FIELDS_V3 | frozenset({
        "asnumpy_adapter_context_plan",
    })
)
_REPAIRED_ADAPTER_REPORT_FIELDS_V2 = frozenset({
    "schema_version",
    "status",
    "adapter_artifact_id",
    "source_artifact_id",
    "source_adapter_root",
    "parent_adapter_artifact_id",
    "parent_adapter_report_sha256",
    "model",
    "provider",
    "configuration_source",
    "repair_profile",
    "attempt_count",
    "paths",
    "before_sha256",
    "after_sha256",
    "files_sha256",
    "diagnostic_sha256",
    "asnumpy_context_sha256",
    "skill_bundle_sha256",
    "integration_skill_files",
    "response_sha256",
    "remote_receipts",
    "actual_remote_call",
    "gates",
})
_REPAIRED_ADAPTER_REPORT_FIELDS_V3 = (
    _REPAIRED_ADAPTER_REPORT_FIELDS_V2 | frozenset({
        "protocol",
        "response_model_aliases",
    })
)
_REPAIRED_ADAPTER_REPORT_FIELDS_V4 = (
    _REPAIRED_ADAPTER_REPORT_FIELDS_V3 | frozenset({
        "asnumpy_adapter_context_plan",
    })
)
_EXPECTED_UNVERIFIED_GATES = {
    "spec": "prepared",
    "generation": "passed",
    "build": "pending",
    "unit_test": "pending",
    "numerical_validation": "pending",
    "activation": "pending",
}

_SYSTEM_PROMPT = """You generate an UNVERIFIED Ascend C registry-invoke operator draft.
Return exactly one NumPilot line bundle, without Markdown fences or commentary:
NUMPILOT_BUNDLE_V1
SUMMARY: one-line summary
FILE: relative/path
CONTENT:
raw file content (do not JSON-escape quotes or newlines)
END_FILE
END_BUNDLE

Follow these rules:
1. Target only the SoC/CANN version in the supplied GapSpec.
2. Use the registry-invoke structure: CMakeLists.txt, build.sh, op_host,
   op_kernel, op_api, tests, and asnumpy_adapter.
3. Kernel parameters are inputs, outputs, workspace, tiling in that order.
4. Include shape/dtype checks, tiling, CPU golden tests, ACLNN two-stage API,
   and an explicit AsNumpy name mapping manifest.
5. Do not invent undocumented Ascend C or ACLNN APIs. If an API cannot be
   verified from the supplied contract, leave a compile-failing
   TODO_VERIFY_CANN_API marker and explain it in README.md.
6. Do not access the network, install packages, alter system paths, or embed
   credentials. build.sh may build only this workspace.
7. This output is a draft. Never claim build, unit test, numerical validation,
   installation, or NPU execution succeeded.
8. Produce a complete, self-contained, build-ready local candidate with at most
   17 files and below 14000 characters. Include all build/host/kernel/API/test/
   AsNumpy-adapter files required by the supplied official Skill. Never put a
   line equal to END_FILE inside file content.
9. Do not emit TODO_VERIFY_CANN_API, placeholder kernels, stub success/failure
   returns, or defer missing implementation to a later model call. The local
   build and qualification gates will reject any unverified API usage.
10. When the GapSpec selects direct generation, do not fetch, clone, copy, or
    require an external operator implementation repository. Generate against
    the embedded Skill and the semantic contract, and keep all output local.
"""

_STAGED_SYSTEM_PROMPT = """You generate one bounded stage of an UNVERIFIED
Ascend C registry-invoke operator project. Return exactly one
NUMPILOT_BUNDLE_V1 line bundle and exactly the paths requested by the user.
The literal envelope is:
NUMPILOT_BUNDLE_V1
SUMMARY: one short stage summary
FILE: relative/path
CONTENT:
raw complete file content
END_FILE
(repeat FILE/CONTENT/END_FILE for every requested path)
END_BUNDLE

Use the supplied NumPy oracle as semantic truth and the selected official
Huawei CANNBot files as engineering truth. Emit complete source: no TODO,
placeholder kernel, stub return, external repository dependency, network
operation, system install, or success claim. Never put a line equal to
END_FILE inside file content.
"""

_REPAIR_SYSTEM_PROMPT = """You repair exactly one file in an UNVERIFIED
Ascend C registry-invoke operator candidate. The local CANN compiler diagnostic
is authoritative. The supplied NumPy oracle is semantic truth and the selected
official Huawei CANNBot files are engineering truth.

Return exactly one NUMPILOT_BUNDLE_V1 line bundle containing only the requested
path. The literal envelope is:
NUMPILOT_BUNDLE_V1
SUMMARY: one short repair summary
FILE: the/exact/requested/path
CONTENT:
raw complete replacement file
END_FILE
END_BUNDLE

Emit the complete replacement file, not a diff. Preserve correct behavior
outside the reported failure. Do not emit TODO, placeholders, stub returns,
external repository dependencies, network operations, system installation, or
success claims. Never put a line equal to END_FILE inside file content.
"""

_ASNUMPY_ADAPTER_SYSTEM_PROMPT = """You generate exactly one file for an
UNVERIFIED AsNumpy operator integration overlay. The signed route-neutral
operator artifact/candidate and its adapter contract are semantic and ABI truth. The supplied
AsNumpy architecture files are integration truth, and the supplied
NumPilot AsNumpy integration Skill is workflow and testing truth.

Return exactly one NUMPILOT_BUNDLE_V1 line bundle containing only the requested
path. The literal envelope is:
NUMPILOT_BUNDLE_V1
SUMMARY: one short adapter summary
FILE: the/exact/requested/path
CONTENT:
raw complete file content
END_FILE
END_BUNDLE

The first line MUST be exactly NUMPILOT_BUNDLE_V1 and the last line MUST be
exactly END_BUNDLE. Emit the complete file, not a diff. Do not edit generated AsNumpy files,
invent APIs, add CPU fallbacks, disable tests, access the network, install
software, or claim build/NPU success. Never put a line equal to END_FILE inside
file content.
"""

_ASNUMPY_OFFICIAL_EXACT_EDIT_SYSTEM_PROMPT = """You propose bounded exact
edits to exactly one EXISTING file for an UNVERIFIED official ACLNN AsNumpy
integration overlay. The signed route-neutral official_binary candidate,
adapter contract, selected CANN header, target file, and existing-module
AsNumpy references are authoritative. The route-specific Skill is engineering
and testing truth, except that its older complete-file output instruction is
superseded by this exact-edit protocol.

Never return a complete file or a NUMPILOT_BUNDLE_V1 envelope. Return exactly:
NUMPILOT_EXACT_EDIT_V1
PATH: the/exact/requested/path
BASE_SHA256: lowercase SHA-256 supplied for that target
EDIT_COUNT: N
BEGIN_EDIT 1
OLD:
an exact non-empty substring copied verbatim from the supplied target
END_OLD 1
NEW:
its exact replacement (which may be empty)
END_NEW 1
END_EDIT 1
END_EXACT_EDIT

For N edits, repeat the numbered block consecutively from 1 through N. Every
OLD block must occur exactly once in the original target, and OLD intervals
must not overlap or leave the supplied target edit windows. Use the smallest
sufficient OLD/NEW blocks. Do not use
Markdown fences, ellipses, line numbers, unified diffs, commentary, TODOs,
CPU fallbacks, network operations, installation, or build/NPU success claims.
"""

_ASNUMPY_OFFICIAL_EXACT_EDIT_REPAIR_SYSTEM_PROMPT = """You repair exactly one
EXISTING file in an UNVERIFIED official ACLNN AsNumpy integration overlay. The
raw local build or test diagnostic is authoritative. The current target file,
its hash-bound dependencies, the selected official CANN header, and the
supplied AsNumpy references are the only edit context.

Never return a complete file or a NUMPILOT_BUNDLE_V1 envelope. Return exactly:
NUMPILOT_EXACT_EDIT_V1
PATH: the/exact/requested/path
BASE_SHA256: lowercase SHA-256 supplied for that target
EDIT_COUNT: N
BEGIN_EDIT 1
OLD:
an exact non-empty substring copied verbatim from the supplied target
END_OLD 1
NEW:
its exact replacement (which may be empty)
END_NEW 1
END_EDIT 1
END_EXACT_EDIT

For N edits, repeat the numbered block consecutively from 1 through N. Every
OLD block must occur exactly once in the original target, and OLD intervals
must not overlap or leave the supplied target edit windows. Use the smallest
sufficient OLD/NEW blocks. Do not use Markdown fences, ellipses, line numbers,
unified diffs, commentary, TODOs, CPU fallbacks, network operations,
installation, or build/NPU success claims.

FRAMEWORK OWNERSHIP INVARIANT: `aclCreateTensorList` stores the caller's
`aclTensor*` descriptors, and `aclDestroyTensorList` destroys every tensor in
the list. An AsNumpy `NPUArray` owns its `tensorPtr` descriptor and its device
buffer; its destructor calls `aclDestroyTensor` on `tensorPtr` and `aclrtFree`
on `device_address()`. Passing an input `NPUArray`'s `tensorPtr` into a list
that is later destroyed therefore double-destroys the input descriptor and can
crash with SIGSEGV. When ACLNN requires a tensor list, create one disposable
alias descriptor per input with
`aclCreateTensor(input.shape.data(), input.shape.size(), input.aclDtype,
input.strides.data(), 0, ACL_FORMAT_ND, input.shape.data(),
input.shape.size(), input.device_address())`,
build the list from those aliases, and destroy only the list. The last
`aclCreateTensor` argument is the device data pointer, so it must be
`input.device_address()`, never `input.tensorPtr`; do not use a NumPy/pybind
dtype or an invented field for the `aclDataType` argument. On a failure before
list creation, destroy the aliases individually. Do not add redundant null
checks or unrelated guards.

PYTHON VALIDATION INVARIANT: A Python wrapper must reject every unsupported
input before it calls native code. The native path is not a validator and may
crash or corrupt memory on malformed inputs. For `concatenate`, require a
non-empty sequence of AsNumpy `ndarray` inputs; reject a zero-dimensional
input; normalize `axis` with `operator.index` (a non-integer, including
`None`, is a TypeError) and require it in `[-rank, rank)`; require every input
to have the same rank and dtype and the same extent on every non-axis
dimension; reject non-positive extents; and reject inputs that are not
C-contiguous. Compute contiguity from `itemsize`, `shape`, and `strides`,
treating a dimension of extent 1 as unconstrained. Raise TypeError or
ValueError before `_concatenate` is entered; do not wrap the native call in a
fallback that hides validation failures.

TEST FILE INVARIANT (only when the target is a test file): The pinned AsNumpy
`ndarray` exposes `shape`, `strides`, `ndim`, `itemsize`, `dtype`, and
`to_numpy()`, but it does not provide NumPy convenience attributes such as
`.T` and it does not provide `transpose`. A non-C-contiguous test input must
be a small `ndarray` subclass that overrides `strides`; do not use `.T`,
delete the case, or weaken the rejection assertion.
"""

_ASNUMPY_ADAPTER_REPAIR_SYSTEM_PROMPT = """You repair exactly one file in an
UNVERIFIED AsNumpy operator integration overlay. The raw local build or
test diagnostic is authoritative. The current overlay and supplied AsNumpy
architecture files are integration truth, and the supplied NumPilot AsNumpy
integration Skill files are engineering truth.

Return exactly one NUMPILOT_BUNDLE_V1 line bundle containing only the requested
path. The literal envelope is:
NUMPILOT_BUNDLE_V1
SUMMARY: one short repair summary
FILE: the/exact/requested/path
CONTENT:
raw complete replacement file
END_FILE
END_BUNDLE

The first line MUST be exactly NUMPILOT_BUNDLE_V1 and the last line MUST be
exactly END_BUNDLE. Do not use Markdown fences or add commentary before or
after the envelope. Emit the complete replacement file, not a diff. Do not edit
generated AsNumpy files, invent APIs, add CPU fallbacks, disable tests, access
the network, install software, or claim build/NPU success. Never put a line
equal to END_FILE inside file content.
"""

_SKILL_PREFIX = "ops/ascendc-registry-invoke-template"


def _strip_json_fence(payload: str) -> str:
    text = payload.strip()
    fence = chr(96) * 3
    if not text.startswith(fence):
        return text
    first_newline = text.find("\n")
    if first_newline < 0 or not text.endswith(fence):
        return text
    return text[first_newline + 1:-len(fence)].strip()


def _safe_relative_path(value: object) -> PurePosixPath:
    if not isinstance(value, str) or not value.strip():
        raise DraftGenerationError("draft file path must be a non-empty string")
    if len(value) > _MAX_PATH_CHARS or any(
        ord(character) < 32 for character in value
    ):
        raise DraftGenerationError("draft file path is too long or unsafe")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise DraftGenerationError("unsafe draft path: %r" % value)
    if not path.parts:
        raise DraftGenerationError("empty draft path")
    if len(path.parts) > _MAX_PATH_DEPTH:
        raise DraftGenerationError("draft path is too deep")
    if len(path.parts) == 1:
        if path.name not in _ALLOWED_ROOT_FILES:
            raise DraftGenerationError("unexpected root draft file: %s" % path)
    elif path.parts[0] not in _ALLOWED_TOP_DIRS:
        raise DraftGenerationError("unexpected draft directory: %s" % path)
    if path.name not in _ALLOWED_ROOT_FILES and path.suffix not in _ALLOWED_SUFFIXES:
        raise DraftGenerationError("unsupported draft file type: %s" % path)
    return path


def _safe_asnumpy_adapter_relative_path(value: object) -> PurePosixPath:
    """Validate one writable path in an isolated AsNumpy overlay.

    This intentionally does not reuse or extend ``_ALLOWED_TOP_DIRS``.  The
    adapter may replace existing AsNumpy binding/public-source files that an
    operator-project draft must not be able to emit.
    """

    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > _MAX_PATH_CHARS
        or "\\" in value
        or any(ord(character) < 32 for character in value)
    ):
        raise DraftGenerationError("AsNumpy adapter path is invalid")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or len(path.parts) > _MAX_PATH_DEPTH
        or any(part in {"", ".", ".."} for part in value.split("/"))
        or str(path) != value
    ):
        raise DraftGenerationError("unsafe AsNumpy adapter path: %r" % value)
    if len(path.parts) == 1:
        if path.name not in _ASNUMPY_ADAPTER_ROOT_FILES:
            raise DraftGenerationError(
                "unexpected AsNumpy adapter root file: %s" % path
            )
    elif path.parts[0] not in _ASNUMPY_ADAPTER_TOP_DIRS:
        raise DraftGenerationError(
            "unexpected AsNumpy adapter directory: %s" % path
        )
    if (
        path.name not in _ASNUMPY_ADAPTER_ROOT_FILES
        and path.suffix not in _ALLOWED_SUFFIXES
    ):
        raise DraftGenerationError(
            "unsupported AsNumpy adapter file type: %s" % path
        )
    return path


def _safe_asnumpy_adapter_context_path(value: object) -> PurePosixPath:
    """Validate a content-only path in an adapter context-plan receipt."""

    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > _MAX_PATH_CHARS
        or "\\" in value
        or any(ord(character) < 32 for character in value)
    ):
        raise DraftGenerationError("AsNumpy adapter context path is invalid")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or len(path.parts) > _MAX_PATH_DEPTH
        or any(part in {"", ".", ".."} for part in value.split("/"))
        or str(path) != value
    ):
        raise DraftGenerationError("unsafe AsNumpy adapter context path: %r" % value)
    if len(path.parts) == 1:
        if path.name not in _ASNUMPY_ADAPTER_ROOT_FILES:
            raise DraftGenerationError(
                "unexpected AsNumpy adapter context root file: %s" % path
            )
    elif path.parts[0] not in _ASNUMPY_ADAPTER_CONTEXT_TOP_DIRS:
        raise DraftGenerationError(
            "unexpected AsNumpy adapter context directory: %s" % path
        )
    if (
        path.name not in _ASNUMPY_ADAPTER_ROOT_FILES
        and path.suffix not in _ALLOWED_SUFFIXES
    ):
        raise DraftGenerationError(
            "unsupported AsNumpy adapter context file type: %s" % path
        )
    return path


def _parse_line_bundle(payload: str) -> Tuple[str, List[Dict[str, str]]]:
    text = payload.strip().replace("\r\n", "\n").replace("\r", "\n")
    prefix = "NUMPILOT_BUNDLE_V1\nSUMMARY: "
    if not text.startswith(prefix):
        raise DraftGenerationError("model output has no NumPilot bundle header")
    summary_end = text.find("\n", len(prefix))
    if summary_end < 0:
        raise DraftGenerationError("model output has no bounded summary")
    summary = text[len(prefix):summary_end]
    cursor = summary_end + 1
    files: List[Dict[str, str]] = []
    while True:
        if text[cursor:] == "END_BUNDLE":
            break
        if not text.startswith("FILE: ", cursor):
            raise DraftGenerationError("model line bundle is missing FILE marker")
        path_end = text.find("\n", cursor)
        if path_end < 0:
            raise DraftGenerationError("model line bundle has an incomplete path")
        path = text[cursor + len("FILE: "):path_end]
        content_start_marker = "CONTENT:\n"
        content_marker_start = path_end + 1
        if not text.startswith(content_start_marker, content_marker_start):
            raise DraftGenerationError("model line bundle is missing CONTENT marker")
        content_start = content_marker_start + len(content_start_marker)
        boundary = "\nEND_FILE\n"
        content_end = text.find(boundary, content_start)
        if content_end < 0:
            raise DraftGenerationError("model line bundle has an unterminated file")
        files.append({"path": path, "content": text[content_start:content_end]})
        cursor = content_end + len(boundary)
    return summary, files


def _validate_bundle_files(
    payload: str,
    *,
    path_validator: Callable[[object], PurePosixPath] = _safe_relative_path,
) -> Tuple[str, Tuple[Tuple[PurePosixPath, str], ...]]:
    if not isinstance(payload, str):
        raise DraftGenerationError("model output must be text")
    if len(payload.encode("utf-8")) > _MAX_RESPONSE_BYTES:
        raise DraftGenerationError("model output exceeds the response size limit")
    stripped = _strip_json_fence(payload)
    if stripped.startswith("NUMPILOT_BUNDLE_V1"):
        summary, raw_files = _parse_line_bundle(stripped)
        data = {"summary": summary, "files": raw_files}
    else:
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise DraftGenerationError(
                "model output is neither a line bundle nor valid JSON: %s; "
                "position=%d; characters=%d; utf8_bytes=%d"
                % (exc.msg, exc.pos, len(payload), len(payload.encode("utf-8")))
            ) from exc
        if not isinstance(data, dict):
            raise DraftGenerationError("model output must be a JSON object")
        summary = data.get("summary", "")
    if not isinstance(summary, str) or len(summary) > _MAX_SUMMARY_CHARS:
        raise DraftGenerationError("summary must be a bounded string")
    raw_files = data.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise DraftGenerationError("files must be a non-empty list")
    if len(raw_files) > _MAX_FILES:
        raise DraftGenerationError("too many generated files")

    seen = set()
    files: List[Tuple[PurePosixPath, str]] = []
    total = 0
    for item in raw_files:
        if not isinstance(item, dict):
            raise DraftGenerationError("each generated file must be an object")
        path = path_validator(item.get("path"))
        content = item.get("content")
        if not isinstance(content, str):
            raise DraftGenerationError("file content must be a string: %s" % path)
        if str(path) in seen:
            raise DraftGenerationError("duplicate generated path: %s" % path)
        seen.add(str(path))
        total += len(content.encode("utf-8"))
        if total > _MAX_TOTAL_BYTES:
            raise DraftGenerationError("generated draft exceeds size limit")
        files.append((path, content))

    forbidden_markers = (
        "TODO_VERIFY_CANN_API",
        "TODO_IMPLEMENT_OPERATOR",
    )
    for path, content in files:
        if any(marker in content for marker in forbidden_markers):
            raise DraftGenerationError(
                "generated draft contains a forbidden placeholder marker: %s"
                % path
            )

    return summary.strip(), tuple(files)


_ASC_TWO_STAGE_MANIFEST_SCHEMA = "asnumpy.adapter.manifest.v1"
_ASC_TWO_STAGE_SYMBOL = re.compile(r"^aclnn[A-Za-z0-9_]{1,127}$")
_ASNUMPY_PUBLIC_SYMBOL = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def _validate_asc_two_stage_manifest_payload(payload: str, label: str) -> None:
    """Validate the semantic contract for a layout-independent ASC artifact.

    An ASC implementation is accepted only when its adjacent AsNumpy manifest
    proves the ACLNN two-stage entry points and public AsNumpy mapping.  This
    keeps the file layout flexible while preventing arbitrary ``.asc`` files
    from bypassing the operator contract.
    """

    try:
        data = json.loads(payload)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise DraftGenerationError(
            "ASC implementation manifest is not valid JSON: %s" % label
        ) from exc
    if not isinstance(data, dict) or data.get("schema") != _ASC_TWO_STAGE_MANIFEST_SCHEMA:
        raise DraftGenerationError(
            "ASC implementation requires %s: %s"
            % (_ASC_TWO_STAGE_MANIFEST_SCHEMA, label)
        )
    # When the CANNBot manifest carries an operator identity, bind the target
    # toolchain and ACLNN boundary before accepting the ASC candidate. Older
    # fixtures without an operator object remain supported by the narrow
    # layout contract; injected/generated manifests must carry these fields.
    if "operator" in data:
        try:
            validate_operator_manifest_for_overlay(data)
        except GeneratedAdapterContractError as exc:
            raise DraftGenerationError(str(exc)) from exc
    name_mapping = data.get("name_mapping")
    if not isinstance(name_mapping, dict):
        raise DraftGenerationError(
            "ASC implementation manifest name_mapping is invalid: %s" % label
        )
    python_mapping = name_mapping.get("python")
    if not isinstance(python_mapping, dict):
        raise DraftGenerationError(
            "ASC implementation manifest Python mapping is invalid: %s" % label
        )
    public_symbol = python_mapping.get("public_symbol")
    if (
        python_mapping.get("module") != "asnumpy"
        or not isinstance(public_symbol, str)
        or _ASNUMPY_PUBLIC_SYMBOL.fullmatch(public_symbol) is None
        or python_mapping.get("public_qualified_name") != "asnumpy.%s" % public_symbol
    ):
        raise DraftGenerationError(
            "ASC implementation manifest AsNumpy mapping is invalid: %s" % label
        )
    aclnn_mapping = name_mapping.get("aclnn")
    if not isinstance(aclnn_mapping, dict):
        raise DraftGenerationError(
            "ASC implementation manifest ACLNN mapping is invalid: %s" % label
        )
    get_workspace = aclnn_mapping.get("get_workspace_size")
    execute = aclnn_mapping.get("execute")
    if (
        aclnn_mapping.get("call_style") != "two-stage"
        or not isinstance(get_workspace, str)
        or _ASC_TWO_STAGE_SYMBOL.fullmatch(get_workspace) is None
        or not isinstance(execute, str)
        or _ASC_TWO_STAGE_SYMBOL.fullmatch(execute) is None
        or get_workspace == execute
        or not get_workspace.endswith("GetWorkspaceSize")
    ):
        raise DraftGenerationError(
            "ASC implementation manifest must declare distinct ACLNN "
            "GetWorkspaceSize/execute symbols: %s" % label
        )
    symbol_resolution = aclnn_mapping.get("symbol_resolution")
    if (
        not isinstance(symbol_resolution, dict)
        or symbol_resolution.get("strategy") != SYMBOL_RESOLUTION_STRATEGY
        or symbol_resolution.get("library_soname") != CUSTOM_OPAPI_SONAME
        or symbol_resolution.get("global_symbol_call") is not False
    ):
        raise DraftGenerationError(
            "ASC implementation manifest must use explicit-handle-dlsym "
            "for custom ACLNN symbols: %s" % label
        )
    cpp_mapping = name_mapping.get("cpp_api")
    if isinstance(cpp_mapping, dict):
        if (
            cpp_mapping.get("l2_get_workspace_size") != get_workspace
            or cpp_mapping.get("l2_execute") != execute
        ):
            raise DraftGenerationError(
                "ASC implementation manifest C++/ACLNN mapping differs: %s"
                % label
            )


def _validate_asc_two_stage_layout(
    names,
    *,
    file_contents: Optional[Dict[str, str]] = None,
    root: Optional[Path] = None,
) -> bool:
    """Return whether files form the explicit ASC two-stage layout.

    This is an additive escape hatch for cannbot-skill outputs.  The existing
    17-file registry layout remains the default; ASC is admitted only with an
    AsNumpy manifest and tests, and only after the manifest's interface
    contract passes the checks above.
    """

    normalized = {str(name) for name in names}
    asc_paths = sorted(
        name for name in normalized if PurePosixPath(name).suffix == ".asc"
    )
    if not asc_paths:
        return False
    if not any(PurePosixPath(name).parts[:1] == ("tests",) for name in normalized):
        raise DraftGenerationError(
            "ASC implementation layout requires tests/"
        )
    manifest_paths = sorted(
        name
        for name in normalized
        if PurePosixPath(name).parts[:1] == ("asnumpy_adapter",)
        and PurePosixPath(name).suffix == ".json"
    )
    if not manifest_paths:
        raise DraftGenerationError(
            "ASC implementation layout requires an AsNumpy manifest"
        )
    if len(manifest_paths) != 1:
        raise DraftGenerationError(
            "ASC implementation layout requires exactly one AsNumpy manifest"
        )
    for manifest_path in manifest_paths:
        if file_contents is not None and manifest_path in file_contents:
            payload = file_contents[manifest_path]
        elif root is not None:
            try:
                payload, _ = _read_bounded_text(
                    root / _safe_relative_path(manifest_path),
                    _MAX_METADATA_BYTES,
                    "ASC implementation manifest",
                )
            except OSError as exc:
                raise DraftGenerationError(
                    "cannot read ASC implementation manifest: %s" % manifest_path
                ) from exc
        else:
            raise DraftGenerationError(
                "ASC implementation manifest contents are unavailable"
            )
        try:
            _validate_asc_two_stage_manifest_payload(payload, manifest_path)
        except DraftGenerationError:
            # A modern operator manifest carries target/toolchain identity and
            # must fail with its concrete contract error. Legacy manifests
            # without an operator object retain the historical generic ASC
            # rejection so older diagnostics remain stable.
            try:
                manifest_data = json.loads(payload)
            except (json.JSONDecodeError, RecursionError):
                manifest_data = {}
            if isinstance(manifest_data, dict) and "operator" in manifest_data:
                raise
            continue
        return True
    raise DraftGenerationError(
        "no AsNumpy manifest declares a valid ACLNN two-stage ASC interface"
    )


def _validate_bundle(payload: str) -> Tuple[str, Tuple[Tuple[PurePosixPath, str], ...]]:
    summary, files = _validate_bundle_files(payload)
    file_contents = {str(path): content for path, content in files}
    names = set(file_contents)
    for required in ("CMakeLists.txt", "build.sh"):
        if required not in names:
            raise DraftGenerationError("generated draft is missing %s" % required)
    _require_project_layout(file_contents, file_contents=file_contents)
    return summary, files


def _validate_stage_bundle(
    payload: str,
    required_paths: Tuple[str, ...],
    *,
    path_validator: Callable[[object], PurePosixPath] = _safe_relative_path,
) -> Tuple[str, Tuple[Tuple[PurePosixPath, str], ...]]:
    summary, files = _validate_bundle_files(
        payload,
        path_validator=path_validator,
    )
    actual = {str(path) for path, _content in files}
    required = set(required_paths)
    if len(files) != len(required_paths) or actual != required:
        missing = sorted(required - actual)
        unexpected = sorted(actual - required)
        raise DraftGenerationError(
            "staged response file set mismatch; missing=%s unexpected=%s"
            % (missing, unexpected)
        )
    return summary, files


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _apply_official_exact_edit_response(
    payload: str,
    *,
    expected_path: str,
    base_content: str,
    allowed_intervals: Optional[Tuple[Tuple[int, int], ...]] = None,
) -> str:
    """Parse and mechanically apply one bounded existing-file edit response."""

    normalized_expected = str(
        _safe_asnumpy_adapter_relative_path(expected_path)
    )
    if normalized_expected != expected_path:
        raise DraftGenerationError(
            "official exact-edit target path is not normalized"
        )
    if (
        not isinstance(base_content, str)
        or not base_content
        or "\x00" in base_content
    ):
        raise DraftGenerationError("official exact-edit base content is invalid")
    if not isinstance(payload, str):
        raise DraftGenerationError("official exact-edit response must be text")
    try:
        base_bytes = base_content.encode("utf-8")
        response_bytes = payload.encode("utf-8")
    except UnicodeError as exc:
        raise DraftGenerationError(
            "official exact-edit input is not valid UTF-8 text"
        ) from exc
    if len(base_bytes) > _MAX_ADAPTER_FILE_BYTES:
        raise DraftGenerationError(
            "official exact-edit base content exceeds its size limit"
        )
    if not response_bytes:
        raise DraftGenerationError("official exact-edit response is empty")
    if len(response_bytes) > _MAX_OFFICIAL_EXACT_EDIT_RESPONSE_BYTES:
        raise DraftGenerationError(
            "official exact-edit response exceeds its size limit"
        )
    if "\x00" in payload:
        raise DraftGenerationError(
            "official exact-edit response contains a NUL byte"
        )
    if "\r" in payload:
        raise DraftGenerationError(
            "official exact-edit response contains a carriage return"
        )
    text = payload[:-1] if payload.endswith("\n") else payload
    prefix = "NUMPILOT_EXACT_EDIT_V1\nPATH: "
    if not text.startswith(prefix):
        raise DraftGenerationError(
            "official exact-edit response has no exact-edit header"
        )
    path_end = text.find("\n", len(prefix))
    if path_end < 0:
        raise DraftGenerationError("official exact-edit response path is incomplete")
    response_path = str(
        _safe_asnumpy_adapter_relative_path(text[len(prefix):path_end])
    )
    if response_path != expected_path:
        raise DraftGenerationError("official exact-edit response path mismatch")

    hash_prefix = "BASE_SHA256: "
    hash_start = path_end + 1
    if not text.startswith(hash_prefix, hash_start):
        raise DraftGenerationError("official exact-edit base hash is missing")
    hash_end = text.find("\n", hash_start)
    if hash_end < 0:
        raise DraftGenerationError("official exact-edit base hash is incomplete")
    base_sha256 = text[hash_start + len(hash_prefix):hash_end]
    if (
        not re.fullmatch(r"[0-9a-f]{64}", base_sha256)
        or base_sha256 != _sha256_bytes(base_bytes)
    ):
        raise DraftGenerationError("official exact-edit base hash mismatch")

    count_prefix = "EDIT_COUNT: "
    count_start = hash_end + 1
    if not text.startswith(count_prefix, count_start):
        raise DraftGenerationError("official exact-edit count is missing")
    count_end = text.find("\n", count_start)
    if count_end < 0:
        raise DraftGenerationError("official exact-edit count is incomplete")
    raw_count = text[count_start + len(count_prefix):count_end]
    if not re.fullmatch(r"[1-9][0-9]*", raw_count):
        raise DraftGenerationError("official exact-edit count is invalid")
    edit_count = int(raw_count)
    if edit_count > _MAX_OFFICIAL_EXACT_EDITS:
        raise DraftGenerationError("official exact-edit count exceeds its limit")

    cursor = count_end + 1
    raw_edits = []
    for index in range(1, edit_count + 1):
        edit_prefix = "BEGIN_EDIT %d\nOLD:\n" % index
        if not text.startswith(edit_prefix, cursor):
            raise DraftGenerationError(
                "official exact-edit block %d is malformed" % index
            )
        old_start = cursor + len(edit_prefix)
        old_marker = "\nEND_OLD %d\nNEW:\n" % index
        old_end = text.find(old_marker, old_start)
        if old_end < 0:
            raise DraftGenerationError(
                "official exact-edit OLD block %d is unterminated" % index
            )
        old = text[old_start:old_end]
        if (
            not old
            or len(old.encode("utf-8"))
            > _MAX_OFFICIAL_EXACT_EDIT_FRAGMENT_BYTES
        ):
            raise DraftGenerationError(
                "official exact-edit OLD block %d is empty or too large" % index
            )
        new_start = old_end + len(old_marker)
        empty_new_marker = "END_NEW %d\nEND_EDIT %d\n" % (index, index)
        new_marker = "\n" + empty_new_marker
        if text.startswith(empty_new_marker, new_start):
            new = ""
            cursor = new_start + len(empty_new_marker)
        else:
            new_end = text.find(new_marker, new_start)
            if new_end < 0:
                raise DraftGenerationError(
                    "official exact-edit NEW block %d is unterminated" % index
                )
            new = text[new_start:new_end]
            cursor = new_end + len(new_marker)
        if (
            len(new.encode("utf-8"))
            > _MAX_OFFICIAL_EXACT_EDIT_FRAGMENT_BYTES
            or new == old
        ):
            raise DraftGenerationError(
                "official exact-edit NEW block %d is invalid" % index
            )
        raw_edits.append((old, new))
    if text[cursor:] != "END_EXACT_EDIT":
        raise DraftGenerationError(
            "official exact-edit response has trailing or missing content"
        )

    located = []
    for index, (old, new) in enumerate(raw_edits, start=1):
        start = base_content.find(old)
        if start < 0 or base_content.find(old, start + 1) >= 0:
            raise DraftGenerationError(
                "official exact-edit OLD block %d must occur exactly once"
                % index
            )
        located.append((start, start + len(old), new, index))
    located.sort(key=lambda item: (item[0], item[1], item[3]))
    normalized_allowed = None
    if allowed_intervals is not None:
        if not isinstance(allowed_intervals, tuple) or not allowed_intervals:
            raise DraftGenerationError(
                "official exact-edit target windows are invalid"
            )
        normalized_allowed = []
        allowed_previous_end = -1
        for interval in allowed_intervals:
            if (
                not isinstance(interval, tuple)
                or len(interval) != 2
                or any(isinstance(value, bool) for value in interval)
                or any(not isinstance(value, int) for value in interval)
            ):
                raise DraftGenerationError(
                    "official exact-edit target windows are invalid"
                )
            allowed_start, allowed_end = interval
            if (
                allowed_start < 0
                or allowed_start >= allowed_end
                or allowed_end > len(base_content)
                or allowed_start < allowed_previous_end
            ):
                raise DraftGenerationError(
                    "official exact-edit target windows are invalid"
                )
            normalized_allowed.append((allowed_start, allowed_end))
            allowed_previous_end = allowed_end
    previous_end = -1
    for start, end, _new, _index in located:
        if start < previous_end:
            raise DraftGenerationError("official exact-edit intervals overlap")
        if normalized_allowed is not None and not any(
            start >= allowed_start and end <= allowed_end
            for allowed_start, allowed_end in normalized_allowed
        ):
            raise DraftGenerationError(
                "official exact-edit OLD block leaves its target edit windows"
            )
        previous_end = end

    rebuilt = []
    cursor = 0
    for start, end, new, _index in located:
        rebuilt.extend((base_content[cursor:start], new))
        cursor = end
    rebuilt.append(base_content[cursor:])
    result = "".join(rebuilt)
    if not result:
        raise DraftGenerationError("official exact-edit rebuilt file is empty")
    if (
        "\x00" in result
        or len(result.encode("utf-8")) > _MAX_ADAPTER_FILE_BYTES
    ):
        raise DraftGenerationError(
            "official exact-edit rebuilt file exceeds its size limit"
        )
    if any(
        marker in result
        for marker in ("TODO_VERIFY_CANN_API", "TODO_IMPLEMENT_OPERATOR")
    ):
        raise DraftGenerationError(
            "official exact-edit rebuilt file contains a forbidden placeholder"
        )
    return result


def _bounded_json_payload(value: object, label: str) -> bytes:
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    if len(payload) > _MAX_METADATA_BYTES:
        raise DraftGenerationError("%s exceeds the size limit" % label)
    return payload


def _read_bounded_bytes(path: Path, maximum: int, label: str) -> bytes:
    """Read at most ``maximum`` bytes plus one sentinel byte."""

    descriptor = -1
    try:
        if path.is_symlink():
            raise DraftGenerationError("%s must not be a symlink" % label)
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise DraftGenerationError("%s must be a regular file" % label)
        if metadata.st_size > maximum:
            raise DraftGenerationError("%s exceeds the size limit" % label)
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            payload = handle.read(maximum + 1)
    except OSError as exc:
        raise DraftGenerationError("cannot read %s" % label) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if len(payload) > maximum:
        raise DraftGenerationError("%s exceeds the size limit" % label)
    return payload


def _read_bounded_text(
    path: Path,
    maximum: int,
    label: str,
) -> Tuple[str, bytes]:
    payload = _read_bounded_bytes(path, maximum, label)
    try:
        return payload.decode("utf-8"), payload
    except UnicodeError as exc:
        raise DraftGenerationError("%s is not UTF-8" % label) from exc


def _hash_bounded_file(
    path: Path,
    maximum: int,
    label: str,
) -> Tuple[str, int]:
    """Hash one regular file with a hard streaming byte bound."""

    digest = hashlib.sha256()
    total = 0
    descriptor = -1
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise DraftGenerationError("%s must be a regular file" % label)
        if metadata.st_size > maximum:
            raise DraftGenerationError("%s exceeds the size limit" % label)
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            while True:
                chunk = handle.read(64 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > maximum:
                    raise DraftGenerationError(
                        "%s exceeds the size limit" % label
                    )
                digest.update(chunk)
    except OSError as exc:
        raise DraftGenerationError("cannot read %s" % label) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return digest.hexdigest(), total


def _bounded_tree_hashes(
    root: Path,
    *,
    excluded: Tuple[str, ...],
    maximum_files: int,
    maximum_entries: int,
    maximum_file_bytes: int,
    maximum_total_bytes: int,
    label: str,
    path_validator: Callable[[object], PurePosixPath] = _safe_relative_path,
) -> Dict[str, str]:
    """Stream, bound and hash every non-metadata regular file below ``root``."""

    excluded_names = set(excluded)
    hashes: Dict[str, str] = {}
    entry_count = 0
    total_bytes = 0
    try:
        candidates = root.rglob("*")
        for candidate in candidates:
            entry_count += 1
            if entry_count > maximum_entries:
                raise DraftGenerationError("%s exceeds the entry limit" % label)
            if candidate.is_symlink():
                raise DraftGenerationError(
                    "%s contains a symlink: %s"
                    % (label, candidate.relative_to(root).as_posix())
                )
            try:
                metadata = candidate.stat()
            except OSError as exc:
                raise DraftGenerationError("cannot inspect %s" % label) from exc
            if stat.S_ISDIR(metadata.st_mode):
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise DraftGenerationError("%s contains a special file" % label)
            relative = candidate.relative_to(root).as_posix()
            if relative in excluded_names:
                continue
            if len(hashes) >= maximum_files:
                raise DraftGenerationError("%s exceeds the file-count limit" % label)
            safe = str(path_validator(relative))
            if safe != relative:
                raise DraftGenerationError("%s path is not normalized" % label)
            if metadata.st_size > maximum_file_bytes:
                raise DraftGenerationError("%s file exceeds the size limit" % label)
            if total_bytes + metadata.st_size > maximum_total_bytes:
                raise DraftGenerationError("%s exceeds the total-size limit" % label)
            digest, actual_size = _hash_bounded_file(
                candidate,
                maximum_file_bytes,
                "%s file %s" % (label, relative),
            )
            if actual_size != metadata.st_size:
                raise DraftGenerationError(
                    "%s file changed while being validated" % label
                )
            total_bytes += actual_size
            if total_bytes > maximum_total_bytes:
                raise DraftGenerationError("%s exceeds the total-size limit" % label)
            hashes[relative] = digest
    except OSError as exc:
        raise DraftGenerationError("cannot enumerate %s" % label) from exc
    return hashes


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_exact_int(
    value: object,
    name: str,
    *,
    minimum: int = 0,
    maximum: int = 2 ** 63 - 1,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or value > maximum
    ):
        raise DraftGenerationError("%s is invalid" % name)
    return value


_RESPONSE_NOT_PROVIDED = object()


def _translate_model_contract(callable_, *args, **kwargs):
    try:
        return callable_(*args, **kwargs)
    except _model_contract.ModelContractError as exc:
        raise DraftGenerationError(str(exc)) from exc


def _validate_remote_receipt(
    receipt: object, response: object = _RESPONSE_NOT_PROVIDED
) -> Dict[str, object]:
    if response is _RESPONSE_NOT_PROVIDED:
        return _translate_model_contract(
            _model_contract.validate_remote_receipt, receipt
        )
    return _translate_model_contract(
        _model_contract.validate_remote_receipt, receipt, response
    )


def validate_remote_receipt(
    receipt: object, response: object = _RESPONSE_NOT_PROVIDED
) -> Dict[str, object]:
    """Public validation boundary for a redacted real-model receipt."""

    return _validate_remote_receipt(receipt, response)


def require_model_configuration(
    provider: object,
    model: object,
    *,
    protocol: object = OPENAI_CHAT_COMPLETIONS_PROTOCOL,
    response_model_aliases: object = (),
):
    """Return a provider-neutral, validated configured-model identity."""

    return _translate_model_contract(
        _model_contract.require_model_configuration,
        provider,
        model,
        protocol=protocol,
        response_model_aliases=response_model_aliases,
    )


def validate_configured_model_receipt(
    receipt: object,
    response: object = _RESPONSE_NOT_PROVIDED,
    *,
    expected_model: object,
    expected_provider: object,
    expected_protocol: object = OPENAI_CHAT_COMPLETIONS_PROTOCOL,
    response_model_aliases: object = (),
) -> Dict[str, object]:
    kwargs = {
        "expected_model": expected_model,
        "expected_provider": expected_provider,
        "expected_protocol": expected_protocol,
        "response_model_aliases": response_model_aliases,
    }
    if response is _RESPONSE_NOT_PROVIDED:
        return _translate_model_contract(
            _model_contract.validate_configured_model_receipt,
            receipt,
            **kwargs,
        )
    return _translate_model_contract(
        _model_contract.validate_configured_model_receipt,
        receipt,
        response,
        **kwargs,
    )


def validate_configured_model_attempt_receipt(
    receipt: object,
    response: object = _RESPONSE_NOT_PROVIDED,
    *,
    expected_model: object,
    expected_provider: object,
    expected_protocol: object = OPENAI_CHAT_COMPLETIONS_PROTOCOL,
    response_model_aliases: object = (),
) -> Dict[str, object]:
    """Validate a failed/successful attempt without accepting its response."""

    kwargs = {
        "expected_model": expected_model,
        "expected_provider": expected_provider,
        "expected_protocol": expected_protocol,
        "response_model_aliases": response_model_aliases,
    }
    if response is _RESPONSE_NOT_PROVIDED:
        return _translate_model_contract(
            _model_contract.validate_configured_model_attempt_receipt,
            receipt,
            **kwargs,
        )
    return _translate_model_contract(
        _model_contract.validate_configured_model_attempt_receipt,
        receipt,
        response,
        **kwargs,
    )


def require_glm_configuration(provider: object, model: object) -> str:
    """Compatibility alias; provider is now an audit label, not a vendor gate."""

    return require_model_configuration(provider, model).model


def validate_glm_remote_receipt(
    receipt: object,
    response: object = _RESPONSE_NOT_PROVIDED,
    *,
    expected_model: object,
) -> Dict[str, object]:
    """Read-only compatibility boundary for exact legacy schema-1 receipts."""

    if not isinstance(receipt, dict) or receipt.get("schema_version") != 1:
        raise DraftGenerationError(
            "validate_glm_remote_receipt only accepts legacy schema 1"
        )
    return validate_configured_model_receipt(
        receipt,
        response,
        expected_model=expected_model,
        expected_provider="glm",
        expected_protocol=OPENAI_CHAT_COMPLETIONS_PROTOCOL,
    )


def _validate_generation_base_receipt(
    receipt: object,
    response: object,
    generation: Dict[str, object],
) -> Dict[str, object]:
    """Validate one call against the exact signed/configured generation identity."""

    return validate_configured_model_receipt(
        receipt,
        response,
        expected_model=generation.get("model"),
        expected_provider=generation.get("provider"),
        expected_protocol=generation.get(
            "protocol", OPENAI_CHAT_COMPLETIONS_PROTOCOL
        ),
        response_model_aliases=generation.get("response_model_aliases", ()),
    )


def _validate_generation_attempt_receipt(
    receipt: object,
    response: object,
    generation: Dict[str, object],
) -> Dict[str, object]:
    """Validate configured/requested identity for one historical attempt."""

    return validate_configured_model_attempt_receipt(
        receipt,
        response,
        expected_model=generation.get("model"),
        expected_provider=generation.get("provider"),
        expected_protocol=generation.get(
            "protocol", OPENAI_CHAT_COMPLETIONS_PROTOCOL
        ),
        response_model_aliases=generation.get("response_model_aliases", ()),
    )


def _record_generation_attempt(
    receipts: List[Dict[str, object]],
    response_hashes: List[str],
    receipt: object,
    response: object,
    generation: Dict[str, object],
) -> Dict[str, object]:
    """Retain a trustworthy attempt, then enforce final response identity."""

    attempted = _validate_generation_attempt_receipt(
        receipt, response, generation
    )
    receipts.append(attempted)
    if isinstance(response, str):
        response_hashes.append(_sha256_bytes(response.encode("utf-8")))
    accepted = _validate_generation_base_receipt(
        attempted, response, generation
    )
    receipts[-1] = accepted
    return accepted


def _generation_identity_snapshot(value: Dict[str, object]):
    identity = require_model_configuration(
        value.get("provider"),
        value.get("model"),
        protocol=value.get(
            "protocol", OPENAI_CHAT_COMPLETIONS_PROTOCOL
        ),
        response_model_aliases=value.get("response_model_aliases", ()),
    )
    return (
        identity.model,
        identity.provider,
        identity.protocol,
        identity.response_model_aliases,
        value.get("configuration_source"),
    )


def _adapter_artifact_id(report: Dict[str, object]) -> str:
    """Return the reproducible identity for an AsNumpy adapter report.

    Schema 1 generated reports used a narrower identity and remain readable.
    A schema 1 repair reused its parent's ID and therefore cannot prove a
    content/parent identity; those reports are deliberately rejected below.
    Schema 2 binds every auditable field except the relocatable parent path.
    """

    if (
        report.get("schema_version") == 1
        and report.get("status") == "generated_unverified"
    ):
        identity = {
            "source_artifact_id": report.get("source_artifact_id"),
            "files": report.get("files_sha256"),
            "response_sha256": report.get("response_sha256"),
        }
    else:
        identity = {
            name: report[name]
            for name in sorted(report)
            if name not in {"adapter_artifact_id", "source_adapter_root"}
        }
    payload = json.dumps(
        identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return "adapter-" + _sha256_bytes(payload)[:20]


def _validate_adapter_hash_map(
    value: object,
    label: str,
    *,
    maximum: int,
    paths: bool,
) -> Dict[str, str]:
    if not isinstance(value, dict) or not value or len(value) > maximum:
        raise DraftGenerationError("AsNumpy adapter %s is invalid" % label)
    normalized: Dict[str, str] = {}
    for raw_name, digest in value.items():
        if paths:
            name = str(_safe_asnumpy_adapter_relative_path(raw_name))
        else:
            if (
                not isinstance(raw_name, str)
                or not raw_name
                or len(raw_name) > _MAX_PATH_CHARS
                or any(ord(character) < 32 for character in raw_name)
            ):
                raise DraftGenerationError(
                    "AsNumpy adapter %s label is invalid" % label
                )
            name = raw_name
        if name in normalized or not _is_sha256(digest):
            raise DraftGenerationError(
                "AsNumpy adapter %s digest is invalid" % label
            )
        normalized[name] = digest
    return normalized


def _validate_adapter_context_plan(value: object) -> Dict[str, object]:
    """Validate the content-free AsNumpy layout receipt bound to an adapter."""

    base_fields = {
        "schema_version", "architecture", "mode", "reference_root",
        "context_sha256", "expected_overlay_paths", "total_bytes",
        "plan_sha256",
    }
    if not isinstance(value, dict):
        raise DraftGenerationError("AsNumpy adapter context plan is invalid")
    schema_version = value.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version not in (1, 2, 3)
    ):
        raise DraftGenerationError("AsNumpy adapter context plan schema is invalid")
    if schema_version == 3:
        fields = base_fields | {
            "prompt_dependencies", "official_library_binding",
        }
    elif schema_version == 2:
        fields = base_fields | {"prompt_dependencies"}
    else:
        fields = base_fields
    if set(value) != fields:
        raise DraftGenerationError("AsNumpy adapter context plan is invalid")
    architecture = value.get("architecture")
    mode = value.get("mode")
    if (architecture, mode) not in {
        ("manifest_generator", "manifest"),
        ("legacy_manual", "manual"),
        ("official_aclnn_existing_module", "existing_module"),
    }:
        raise DraftGenerationError("AsNumpy adapter architecture/mode is invalid")
    root = value.get("reference_root")
    if (
        not isinstance(root, str)
        or not root
        or len(root) > 4096
        or any(ord(character) < 32 for character in root)
        or not Path(root).is_absolute()
    ):
        raise DraftGenerationError("AsNumpy adapter reference root is invalid")
    raw_contexts = value.get("context_sha256")
    maximum_context_files = (
        _MAX_OFFICIAL_CONTEXT_FILES if schema_version in (2, 3) else 8
    )
    if (
        not isinstance(raw_contexts, dict)
        or not raw_contexts
        or len(raw_contexts) > maximum_context_files
    ):
        raise DraftGenerationError(
            "AsNumpy adapter context-plan hash map is invalid"
        )
    contexts: Dict[str, str] = {}
    for raw_name, digest in raw_contexts.items():
        name = str(_safe_asnumpy_adapter_context_path(raw_name))
        if name in contexts or name != raw_name or not _is_sha256(digest):
            raise DraftGenerationError(
                "AsNumpy adapter context-plan path/hash is invalid"
            )
        contexts[name] = digest
    raw_paths = value.get("expected_overlay_paths")
    if (
        not isinstance(raw_paths, list)
        or not raw_paths
        or len(raw_paths) > _MAX_ADAPTER_FILES
    ):
        raise DraftGenerationError("AsNumpy adapter context-plan paths are invalid")
    paths = tuple(
        str(_safe_asnumpy_adapter_relative_path(item)) for item in raw_paths
    )
    if len(set(paths)) != len(paths) or list(paths) != raw_paths:
        raise DraftGenerationError("AsNumpy adapter context-plan paths are invalid")
    prompt_dependencies = None
    if schema_version in (2, 3):
        if (architecture, mode) != (
            "official_aclnn_existing_module", "existing_module"
        ):
            raise DraftGenerationError(
                "AsNumpy adapter projected plan is not an official existing module"
            )
        raw_dependencies = value.get("prompt_dependencies")
        if (
            not isinstance(raw_dependencies, dict)
            or set(raw_dependencies) != set(paths)
        ):
            raise DraftGenerationError(
                "AsNumpy adapter prompt dependencies are invalid"
            )
        prompt_dependencies = {}
        generated_before = set()
        for target in paths:
            raw_items = raw_dependencies.get(target)
            if (
                not isinstance(raw_items, list)
                or not raw_items
                or len(raw_items) > _MAX_ADAPTER_FILES
            ):
                raise DraftGenerationError(
                    "AsNumpy adapter prompt dependencies are invalid"
                )
            items = []
            seen = set()
            for raw_item in raw_items:
                if not isinstance(raw_item, dict) or set(raw_item) != {
                    "source", "path",
                }:
                    raise DraftGenerationError(
                        "AsNumpy adapter prompt dependency is invalid"
                    )
                source = raw_item.get("source")
                raw_path = raw_item.get("path")
                if source == "reference":
                    dependency_path = str(
                        _safe_asnumpy_adapter_context_path(raw_path)
                    )
                    valid = (
                        dependency_path == raw_path
                        and dependency_path in contexts
                        and dependency_path != target
                    )
                elif source == "signed_official_header":
                    valid = (
                        isinstance(raw_path, str)
                        and re.fullmatch(
                            r"SIGNED_OFFICIAL_HEADER/[A-Za-z0-9_.-]{1,128}",
                            raw_path,
                        )
                        is not None
                    )
                    dependency_path = raw_path if valid else ""
                elif source == "generated_overlay":
                    dependency_path = str(
                        _safe_asnumpy_adapter_relative_path(raw_path)
                    )
                    valid = (
                        dependency_path == raw_path
                        and dependency_path in generated_before
                    )
                else:
                    valid = False
                    dependency_path = ""
                identity = (source, dependency_path)
                if not valid or identity in seen:
                    raise DraftGenerationError(
                        "AsNumpy adapter prompt dependency is invalid"
                    )
                seen.add(identity)
                items.append({"source": source, "path": dependency_path})
            prompt_dependencies[target] = items
            generated_before.add(target)
    official_library_binding = None
    if schema_version == 3:
        raw_binding = value.get("official_library_binding")
        binding_fields = {
            "schema_version",
            "proof_kind",
            "cmake_path",
            "cmake_sha256",
            "target",
            "link_item",
            "selected_library_relative_path",
            "selected_library_basename",
            "selected_library_sha256",
        }
        if not isinstance(raw_binding, dict) or set(raw_binding) != binding_fields:
            raise DraftGenerationError(
                "AsNumpy official library binding is invalid"
            )
        cmake_path = str(_safe_asnumpy_adapter_context_path(
            raw_binding.get("cmake_path")
        ))
        selected_path_value = raw_binding.get(
            "selected_library_relative_path"
        )
        if not isinstance(selected_path_value, str):
            raise DraftGenerationError(
                "AsNumpy official library binding is invalid"
            )
        selected_path = PurePosixPath(selected_path_value)
        target = raw_binding.get("target")
        link_item = raw_binding.get("link_item")
        basename = raw_binding.get("selected_library_basename")
        if (
            raw_binding.get("schema_version") != 1
            or raw_binding.get("proof_kind")
            != "reviewed_exact_cmake_target_link_item"
            or cmake_path != raw_binding.get("cmake_path")
            or cmake_path not in contexts
            or raw_binding.get("cmake_sha256") != contexts[cmake_path]
            or not isinstance(target, str)
            or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", target) is None
            or not isinstance(link_item, str)
            or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", link_item) is None
            or not isinstance(basename, str)
            or basename != "lib%s.so" % link_item
            or not selected_path_value
            or "\\" in selected_path_value
            or selected_path.is_absolute()
            or any(
                part in {"", ".", ".."}
                for part in selected_path_value.split("/")
            )
            or str(selected_path) != selected_path_value
            or selected_path.name != basename
            or not _is_sha256(raw_binding.get("selected_library_sha256"))
            or raw_binding.get("selected_library_sha256") == "0" * 64
        ):
            raise DraftGenerationError(
                "AsNumpy official library binding is invalid"
            )
        official_library_binding = {
            "schema_version": 1,
            "proof_kind": "reviewed_exact_cmake_target_link_item",
            "cmake_path": cmake_path,
            "cmake_sha256": contexts[cmake_path],
            "target": target,
            "link_item": link_item,
            "selected_library_relative_path": selected_path_value,
            "selected_library_basename": basename,
            "selected_library_sha256": raw_binding[
                "selected_library_sha256"
            ],
        }
    total_bytes = _require_exact_int(
        value.get("total_bytes"),
        "AsNumpy adapter context-plan byte count",
        minimum=1,
        maximum=256 * 1024,
    )
    identity = {
        "schema_version": schema_version,
        "architecture": architecture,
        "mode": mode,
        "context_sha256": contexts,
        "expected_overlay_paths": list(paths),
    }
    if schema_version in (2, 3):
        identity["prompt_dependencies"] = prompt_dependencies
    if schema_version == 3:
        identity["official_library_binding"] = official_library_binding
    expected_digest = _sha256_bytes(json.dumps(
        identity, sort_keys=True, separators=(",", ":")
    ).encode("utf-8"))
    if value.get("plan_sha256") != expected_digest:
        raise DraftGenerationError("AsNumpy adapter context-plan digest is invalid")
    return {
        "schema_version": schema_version,
        "architecture": architecture,
        "mode": mode,
        "reference_root": root,
        "context_sha256": contexts,
        "expected_overlay_paths": list(paths),
        "total_bytes": total_bytes,
        "plan_sha256": expected_digest,
        **(
            {
                "prompt_dependencies": prompt_dependencies,
                **(
                    {"official_library_binding": official_library_binding}
                    if schema_version == 3
                    else {}
                ),
            }
            if schema_version in (2, 3)
            else {}
        ),
    }


def _validate_official_exact_edit_inputs(
    *,
    operator_artifact: Dict[str, object],
    adapter_contract: str,
    asnumpy_context: Dict[str, str],
    normalized_paths: Tuple[str, ...],
    normalized_plan: Dict[str, object],
    skill_files: Tuple[str, ...],
) -> None:
    """Prove that an official existing-module generation is route-bound."""

    from numpilot.asnumpy_adapter_context import (
        reviewed_official_adapter_prompt_dependencies,
    )

    try:
        expected_prompt_dependencies = (
            reviewed_official_adapter_prompt_dependencies(normalized_paths)
        )
    except ValueError as exc:
        raise DraftGenerationError(
            "official exact-edit overlay shape is not reviewed"
        ) from exc
    if (
        operator_artifact.get("artifact_type")
        != "official_binary_adapter_input"
        or operator_artifact.get("strategy")
        != "official_aclnn_existing_module"
        or operator_artifact.get("origin_kind") != "official_binary"
        or not isinstance(operator_artifact.get("artifact_id"), str)
        or operator_artifact.get("artifact_id")
        != operator_artifact.get("candidate_id")
        or operator_artifact.get("model_invocation_allowed") is not True
        or operator_artifact.get("context_plan_status") != "prepared"
        or operator_artifact.get("new_kernel_generation_allowed") is not False
        or operator_artifact.get("custom_opp_generation_allowed") is not False
        or operator_artifact.get("activation_allowed") is not False
        or operator_artifact.get("asnumpy_adapter_context_plan")
        != normalized_plan
        or normalized_plan.get("schema_version") != 3
        or normalized_plan.get("prompt_dependencies")
        != expected_prompt_dependencies
        or tuple(expected_prompt_dependencies) != normalized_paths
        or any(
            path not in _OFFICIAL_EXACT_EDIT_TARGET_WINDOWS
            for path in normalized_paths
        )
        or tuple(skill_files) != ASNUMPY_OFFICIAL_ADAPTER_SKILL_FILES
    ):
        raise DraftGenerationError(
            "official exact-edit generation inputs are not route-bound"
        )
    try:
        contract_payload = adapter_contract.encode("utf-8")
    except UnicodeError as exc:
        raise DraftGenerationError(
            "official adapter contract is not UTF-8"
        ) from exc
    if (
        not contract_payload
        or len(contract_payload) > 512 * 1024
        or operator_artifact.get("adapter_contract_sha256")
        != _sha256_bytes(contract_payload)
    ):
        raise DraftGenerationError("official adapter contract binding mismatch")

    def strict_pairs(pairs):
        result = {}
        for name, value in pairs:
            if name in result:
                raise DraftGenerationError(
                    "official adapter contract contains a duplicate key"
                )
            result[name] = value
        return result

    def reject_constant(value):
        raise DraftGenerationError(
            "official adapter contract contains a non-standard constant: %s"
            % value
        )

    try:
        contract = json.loads(
            adapter_contract,
            object_pairs_hook=strict_pairs,
            parse_constant=reject_constant,
        )
    except DraftGenerationError:
        raise
    except (json.JSONDecodeError, RecursionError) as exc:
        raise DraftGenerationError(
            "official adapter contract is not strict JSON"
        ) from exc
    if (
        not isinstance(contract, dict)
        or contract.get("contract_type")
        != "official_binary_asnumpy_adapter"
        or contract.get("strategy") != "official_aclnn_existing_module"
        or contract.get("activation_allowed") is not False
    ):
        raise DraftGenerationError("official adapter contract strategy mismatch")
    source_candidate = contract.get("source_candidate")
    api = contract.get("api")
    target = contract.get("target")
    permissions = contract.get("permissions")
    if (
        not isinstance(source_candidate, dict)
        or source_candidate.get("origin_kind") != "official_binary"
        or source_candidate.get("artifact_id")
        != operator_artifact.get("artifact_id")
        or source_candidate.get("candidate_id")
        != operator_artifact.get("candidate_id")
        or not isinstance(api, dict)
        or api.get("gap_id") != operator_artifact.get("gap_id")
        or api.get("qualified_api")
        != operator_artifact.get("qualified_api")
        or api.get("asnumpy_api") != operator_artifact.get("asnumpy_api")
        or not isinstance(target, dict)
        or target.get("soc") != operator_artifact.get("target_soc")
        or target.get("cann_version")
        != operator_artifact.get("cann_version")
        or not isinstance(permissions, dict)
        or permissions.get("reuse_installed_aclnn") is not True
        or permissions.get("generate_new_kernel") is not False
        or permissions.get("generate_custom_opp") is not False
        or permissions.get("activation_allowed") is not False
    ):
        raise DraftGenerationError(
            "official adapter manifest/contract identity mismatch"
        )
    official = contract.get("official_binary")
    header = official.get("selected_header") if isinstance(official, dict) else None
    if not isinstance(header, dict):
        raise DraftGenerationError("official adapter selected header is missing")
    header_path = header.get("relative_path")
    header_sha256 = header.get("sha256")
    if (
        not isinstance(header_path, str)
        or not header_path
        or "\\" in header_path
        or PurePosixPath(header_path).is_absolute()
        or any(part in {"", ".", ".."} for part in header_path.split("/"))
        or not _is_sha256(header_sha256)
        or header_sha256 == "0" * 64
    ):
        raise DraftGenerationError("official adapter selected header is invalid")
    header_name = PurePosixPath(header_path).name
    expected_header_label = "SIGNED_OFFICIAL_HEADER/" + header_name
    header_labels = tuple(
        sorted(
            label for label in asnumpy_context
            if label.startswith("SIGNED_OFFICIAL_HEADER/")
        )
    )
    if header_labels != (expected_header_label,):
        raise DraftGenerationError(
            "official adapter signed header context is missing or ambiguous"
        )
    header_content = asnumpy_context[expected_header_label]
    if _sha256_bytes(header_content.encode("utf-8")) != header_sha256:
        raise DraftGenerationError("official adapter signed header hash mismatch")

    library = (
        official.get("selected_library") if isinstance(official, dict) else None
    )
    library_binding = normalized_plan.get("official_library_binding")
    if not isinstance(library, dict) or not isinstance(library_binding, dict):
        raise DraftGenerationError(
            "official adapter selected-library link binding is missing"
        )
    library_path = library.get("relative_path")
    library_sha256 = library.get("sha256")
    cmake_path = library_binding.get("cmake_path")
    cmake_content = asnumpy_context.get(cmake_path)
    if (
        not isinstance(library_path, str)
        or not library_path
        or "\\" in library_path
        or PurePosixPath(library_path).is_absolute()
        or any(part in {"", ".", ".."} for part in library_path.split("/"))
        or PurePosixPath(library_path).name != "libopapi.so"
        or not _is_sha256(library_sha256)
        or library_sha256 == "0" * 64
        or library_binding.get("proof_kind")
        != "reviewed_exact_cmake_target_link_item"
        or cmake_path != "bindings/python/CMakeLists.txt"
        or library_binding.get("target") != "_core"
        or library_binding.get("link_item") != "opapi"
        or library_binding.get("selected_library_relative_path")
        != library_path
        or library_binding.get("selected_library_basename") != "libopapi.so"
        or library_binding.get("selected_library_sha256") != library_sha256
        or not isinstance(cmake_content, str)
        or _sha256_bytes(cmake_content.encode("utf-8"))
        != library_binding.get("cmake_sha256")
        or cmake_content.count(_OFFICIAL_MATMUL_LINK_BLOCK) != 1
        or cmake_content.count("target_link_libraries(_core") != 2
        or re.search(
            r"\b(?:opapi_nn|opapi_math|cust_opapi)\b", cmake_content
        )
        is not None
    ):
        raise DraftGenerationError(
            "official selected library does not match the reviewed AsNumPy "
            "link graph"
        )

    plan_hashes = normalized_plan["context_sha256"]
    if any(
        path not in plan_hashes
        or path not in asnumpy_context
        or _sha256_bytes(asnumpy_context[path].encode("utf-8"))
        != plan_hashes[path]
        for path in normalized_paths
    ):
        raise DraftGenerationError(
            "official exact-edit target is absent from the context plan"
        )


def _render_official_exact_edit_projection(
    *,
    target_path: str,
    asnumpy_context: Dict[str, str],
    normalized_plan: Dict[str, object],
    collected: Dict[str, str],
    collected_provenance: Dict[str, Dict[str, str]],
) -> str:
    """Render only the reviewed, hash-bound dependencies for one target."""

    raw_dependencies = normalized_plan["prompt_dependencies"][target_path]
    dependency_manifest = []
    content_sections = []
    for dependency in raw_dependencies:
        source = dependency["source"]
        path = dependency["path"]
        if source == "reference":
            content = asnumpy_context.get(path)
            digest = normalized_plan["context_sha256"].get(path)
            if (
                not isinstance(content, str)
                or not _is_sha256(digest)
                or _sha256_bytes(content.encode("utf-8")) != digest
            ):
                raise DraftGenerationError(
                    "official exact-edit reference dependency drifted"
                )
            metadata = {
                "source": source,
                "path": path,
                "sha256": digest,
                "binding": "asnumpy_adapter_context_plan.context_sha256",
            }
        elif source == "signed_official_header":
            content = asnumpy_context.get(path)
            if not isinstance(content, str) or not content:
                raise DraftGenerationError(
                    "official exact-edit signed-header dependency is missing"
                )
            metadata = {
                "source": source,
                "path": path,
                "sha256": _sha256_bytes(content.encode("utf-8")),
                "binding": "adapter_contract.official_binary.selected_header.sha256",
            }
        elif source == "generated_overlay":
            content = collected.get(path)
            provenance = collected_provenance.get(path)
            if not isinstance(content, str) or not isinstance(provenance, dict):
                raise DraftGenerationError(
                    "official exact-edit generated dependency is unavailable"
                )
            digest = _sha256_bytes(content.encode("utf-8"))
            required_hashes = (
                "base_sha256",
                "rebuilt_sha256",
                "response_sha256",
                "request_body_sha256",
                "response_body_sha256",
                "receipt_content_sha256",
            )
            if (
                provenance.get("rebuilt_sha256") != digest
                or any(not _is_sha256(provenance.get(name)) for name in required_hashes)
            ):
                raise DraftGenerationError(
                    "official exact-edit generated dependency binding is invalid"
                )
            metadata = {
                "source": source,
                "path": path,
                **{name: provenance[name] for name in required_hashes},
            }
        else:  # The schema-2 plan validator already rejects this branch.
            raise DraftGenerationError(
                "official exact-edit dependency source is invalid"
            )
        dependency_manifest.append(metadata)
        content_sections.append(
            "HASH-BOUND DEPENDENCY `%s` (%s):\n%s"
            % (path, source, content)
        )

    projection_identity = {
        "schema_version": 1,
        "target_path": target_path,
        "context_plan_sha256": normalized_plan["plan_sha256"],
        "dependencies": dependency_manifest,
    }
    projection_sha256 = _sha256_bytes(json.dumps(
        projection_identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8"))
    projection = {
        **projection_identity,
        "projection_sha256": projection_sha256,
    }
    return "\n\n".join((
        "HASH-BOUND TARGET DEPENDENCY PROJECTION:\n" + json.dumps(
            projection, ensure_ascii=False, indent=2, sort_keys=True
        ),
        *content_sections,
    ))


def _render_official_exact_edit_repair_dependencies(
    *,
    target_path: str,
    asnumpy_context: Dict[str, str],
    base_context_hashes: Dict[str, str],
    normalized_plan: Dict[str, object],
    current_contents: Dict[str, str],
    current_hashes: Dict[str, str],
) -> str:
    """Render only the reviewed dependencies of one repair target.

    Repair does not regenerate the other overlay files, so their original
    per-file generation provenance is unavailable.  Bind generated
    dependencies to the validated parent report instead of fabricating
    request/response hashes that the repair request never observed.
    """

    raw_dependencies = normalized_plan["prompt_dependencies"].get(target_path)
    if not raw_dependencies:
        raise DraftGenerationError(
            "official exact-edit repair dependency plan is missing"
        )

    manifest = []
    content_sections = []
    for dependency in raw_dependencies:
        source = dependency.get("source")
        path = dependency.get("path")
        if not isinstance(path, str) or not path:
            raise DraftGenerationError(
                "official exact-edit repair dependency path is invalid"
            )
        if source == "reference":
            content = asnumpy_context.get(path)
            digest = normalized_plan["context_sha256"].get(path)
            if (
                not isinstance(content, str)
                or not content
                or not _is_sha256(digest)
                or _sha256_bytes(content.encode("utf-8")) != digest
            ):
                raise DraftGenerationError(
                    "official exact-edit repair reference dependency drifted"
                )
            binding = "asnumpy_adapter_context_plan.context_sha256"
        elif source == "signed_official_header":
            content = asnumpy_context.get(path)
            if isinstance(content, str) and content:
                digest = _sha256_bytes(content.encode("utf-8"))
                binding = "repair_request.asnumpy_context_sha256"
            else:
                # The repair request intentionally replays only the reviewed
                # context-plan references.  The signed CANN header was read
                # during generation but is not part of that plan, so bind it
                # to the parent report's recorded digest instead of pretending
                # the repair request carried its bytes.
                digest = base_context_hashes.get(path)
                if not _is_sha256(digest):
                    raise DraftGenerationError(
                        "official exact-edit repair signed-header dependency "
                        "is missing"
                    )
                binding = "parent_adapter_report.asnumpy_context_sha256"
        elif source == "generated_overlay":
            content = current_contents.get(path)
            digest = current_hashes.get(path)
            if (
                not isinstance(content, str)
                or not content
                or not _is_sha256(digest)
                or _sha256_bytes(content.encode("utf-8")) != digest
            ):
                raise DraftGenerationError(
                    "official exact-edit repair generated dependency drifted"
                )
            binding = "parent_adapter_report.files_sha256"
        else:
            raise DraftGenerationError(
                "official exact-edit repair dependency source is invalid"
            )
        manifest.append({
            "source": source,
            "path": path,
            "sha256": digest,
            "binding": binding,
            "content_supplied": bool(content),
        })
        if isinstance(content, str) and content:
            content_sections.append(
                "HASH-BOUND REPAIR DEPENDENCY `%s` (%s):\n%s"
                % (path, source, content)
            )

    identity = {
        "schema_version": 1,
        "target_path": target_path,
        "context_plan_sha256": normalized_plan["plan_sha256"],
        "dependencies": manifest,
    }
    projection = {
        **identity,
        "projection_sha256": _sha256_bytes(json.dumps(
            identity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")),
    }
    return "\n\n".join((
        "HASH-BOUND REPAIR DEPENDENCY PROJECTION:\n" + json.dumps(
            projection, ensure_ascii=False, indent=2, sort_keys=True
        ),
        *content_sections,
    ))


def _render_official_exact_edit_target_windows(
    *, target_path: str, base_content: str,
) -> Tuple[str, Tuple[Tuple[int, int], ...]]:
    """Render the reviewed target slices and return their strict edit bounds."""

    specifications = _OFFICIAL_EXACT_EDIT_TARGET_WINDOWS.get(target_path)
    if not specifications:
        raise DraftGenerationError(
            "official exact-edit target window plan is missing"
        )
    windows = []
    intervals = []
    previous_end = -1
    sections = []
    for label, start_marker, end_marker in specifications:
        if start_marker is None:
            start = 0
        else:
            if base_content.count(start_marker) != 1:
                raise DraftGenerationError(
                    "official exact-edit target start anchor is ambiguous"
                )
            start = base_content.find(start_marker)
        if end_marker is None:
            end = len(base_content)
        else:
            if base_content.count(end_marker) != 1:
                raise DraftGenerationError(
                    "official exact-edit target end anchor is ambiguous"
                )
            end = base_content.find(end_marker)
        if start < 0 or end <= start or start < previous_end:
            raise DraftGenerationError(
                "official exact-edit target window bounds are invalid"
            )
        content = base_content[start:end]
        digest = _sha256_bytes(content.encode("utf-8"))
        windows.append({
            "label": label,
            "start_character": start,
            "end_character": end,
            "sha256": digest,
        })
        intervals.append((start, end))
        sections.append(
            "TARGET EDIT WINDOW `%s` [%d,%d) SHA-256 %s:\n%s"
            % (label, start, end, digest, content)
        )
        previous_end = end
    identity = {
        "schema_version": 1,
        "target_path": target_path,
        "full_base_sha256": _sha256_bytes(base_content.encode("utf-8")),
        "windows": windows,
    }
    manifest = {
        **identity,
        "window_plan_sha256": _sha256_bytes(json.dumps(
            identity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")),
    }
    return "\n\n".join((
        "HASH-BOUND TARGET EDIT WINDOWS:\n" + json.dumps(
            manifest, ensure_ascii=False, indent=2, sort_keys=True
        ),
        *sections,
    )), tuple(intervals)


def _enforce_official_exact_edit_prompt_budget(
    *,
    system_prompt: str,
    user_prompt: str,
    model: str,
    max_tokens: Optional[int],
    thinking_mode: Optional[str],
    reasoning_effort: Optional[str],
    stream: bool,
) -> int:
    """Reject an oversized exact-edit request before counting/calling it."""

    request_data = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
    }
    if max_tokens is not None:
        request_data["max_tokens"] = max_tokens
    if thinking_mode is not None:
        request_data["thinking"] = {"type": thinking_mode}
    if reasoning_effort is not None:
        request_data["reasoning_effort"] = reasoning_effort
    if stream:
        request_data["stream"] = True
    try:
        size = len(json.dumps(request_data).encode("utf-8"))
    except (TypeError, UnicodeError, ValueError) as exc:
        raise DraftGenerationError(
            "official exact-edit prompt is not serializable"
        ) from exc
    if size > _MAX_OFFICIAL_EXACT_EDIT_PROMPT_BYTES:
        raise DraftGenerationError(
            "official exact-edit prompt exceeds its 64 KiB request budget: %d"
            % size
        )
    return size


def _validate_adapter_report(
    report: object,
    *,
    candidate_hashes: Optional[Dict[str, str]] = None,
    parent_report: Optional[Dict[str, object]] = None,
    parent_report_sha256: Optional[str] = None,
) -> Dict[str, object]:
    """Validate a complete adapter report and, when supplied, its parent edge."""

    if not isinstance(report, dict):
        raise DraftGenerationError("AsNumpy adapter report must be an object")
    schema_version = report.get("schema_version")
    status = report.get("status")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version not in (1, 2, 3, 4)
    ):
        raise DraftGenerationError("AsNumpy adapter report schema is unsupported")
    if status == "generated_unverified":
        expected_fields = (
            _GENERATED_ADAPTER_REPORT_FIELDS_V4
            if schema_version == 4
            else _GENERATED_ADAPTER_REPORT_FIELDS_V3
            if schema_version == 3
            else _GENERATED_ADAPTER_REPORT_FIELDS_V2
        )
    elif status == "repaired_unverified":
        if schema_version == 1:
            raise DraftGenerationError(
                "legacy AsNumpy adapter repair cannot prove a distinct artifact "
                "identity and parent chain; regenerate it (fail-closed)"
            )
        expected_fields = (
            _REPAIRED_ADAPTER_REPORT_FIELDS_V4
            if schema_version == 4
            else _REPAIRED_ADAPTER_REPORT_FIELDS_V3
            if schema_version == 3
            else _REPAIRED_ADAPTER_REPORT_FIELDS_V2
        )
    else:
        raise DraftGenerationError("AsNumpy adapter report status is invalid")
    if frozenset(report) != expected_fields:
        if (
            status == "repaired_unverified"
            and schema_version == 2
            and "repair_profile" not in report
        ):
            raise DraftGenerationError(
                "legacy AsNumpy adapter repair cannot prove its repair-profile "
                "binding; rerun the repair (fail-closed)"
            )
        raise DraftGenerationError("AsNumpy adapter report schema is invalid")

    for name in (
        "source_artifact_id",
        "model",
        "provider",
        "configuration_source",
    ):
        value = report.get(name)
        if (
            not isinstance(value, str)
            or not value
            or len(value) > 512
            or any(ord(character) < 32 for character in value)
        ):
            raise DraftGenerationError(
                "AsNumpy adapter report field is invalid: %s" % name
            )
    _require_exact_int(
        report.get("attempt_count"),
        "AsNumpy adapter attempt count",
        minimum=1,
        maximum=_MAX_TRANSPORT_ATTEMPTS,
    )
    files = _validate_adapter_hash_map(
        report.get("files_sha256"),
        "file map",
        maximum=_MAX_ADAPTER_FILES,
        paths=True,
    )
    contexts = _validate_adapter_hash_map(
        report.get("asnumpy_context_sha256"),
        "context map",
        maximum=64,
        paths=False,
    )
    if not contexts or not _is_sha256(report.get("skill_bundle_sha256")):
        raise DraftGenerationError("AsNumpy adapter context/skill binding is invalid")
    skill_files = report.get("integration_skill_files")
    if (
        not isinstance(skill_files, list)
        or not skill_files
        or len(skill_files) > 64
        or any(
            not isinstance(item, str)
            or not item
            or len(item) > _MAX_PATH_CHARS
            or any(ord(character) < 32 for character in item)
            for item in skill_files
        )
    ):
        raise DraftGenerationError("AsNumpy adapter skill file binding is invalid")
    if len(set(skill_files)) != len(skill_files):
        raise DraftGenerationError("AsNumpy adapter skill file binding is invalid")
    for item in skill_files:
        skill_path = PurePosixPath(item)
        if (
            skill_path.is_absolute()
            or not skill_path.parts
            or ".." in skill_path.parts
            or "." in skill_path.parts
            or str(skill_path) != item
        ):
            raise DraftGenerationError(
                "AsNumpy adapter skill file binding is invalid"
            )

    response_hashes = report.get("response_sha256")
    receipts = report.get("remote_receipts")
    attempt_count = report["attempt_count"]
    if (
        not isinstance(response_hashes, list)
        or not response_hashes
        or len(response_hashes) > attempt_count
        or any(not _is_sha256(item) for item in response_hashes)
        or not isinstance(receipts, list)
        or not receipts
        or len(receipts) > attempt_count
        or report.get("actual_remote_call") is not True
        or report.get("gates") != _EXPECTED_UNVERIFIED_GATES
    ):
        raise DraftGenerationError("AsNumpy adapter transport evidence is invalid")
    identity = require_model_configuration(
        report.get("provider"),
        report.get("model"),
        protocol=report.get(
            "protocol", OPENAI_CHAT_COMPLETIONS_PROTOCOL
        ),
        response_model_aliases=report.get("response_model_aliases", ()),
    )
    receipt_validator = (
        validate_configured_model_attempt_receipt
        if schema_version in {3, 4}
        else validate_configured_model_receipt
    )
    normalized_receipts = [
        receipt_validator(
            item,
            expected_model=identity.model,
            expected_provider=identity.provider,
            expected_protocol=identity.protocol,
            response_model_aliases=identity.response_model_aliases,
        )
        for item in receipts
    ]
    content_hashes = [
        item["content_sha256"]
        for item in normalized_receipts
        if item["content_sha256"] is not None
    ]
    if (
        content_hashes != response_hashes
    ):
        raise DraftGenerationError(
            "AsNumpy adapter receipt/content hashes are inconsistent"
        )
    if schema_version in {3, 4}:
        validate_configured_model_receipt(
            normalized_receipts[-1],
            expected_model=identity.model,
            expected_provider=identity.provider,
            expected_protocol=identity.protocol,
            response_model_aliases=identity.response_model_aliases,
        )
        if normalized_receipts[-1]["content_sha256"] != response_hashes[-1]:
            raise DraftGenerationError(
                "AsNumpy adapter final receipt is not the accepted response"
            )
    if candidate_hashes is not None and files != candidate_hashes:
        raise DraftGenerationError(
            "AsNumpy adapter candidate content differs from its report"
        )

    context_plan = None
    if schema_version == 4:
        context_plan = _validate_adapter_context_plan(
            report.get("asnumpy_adapter_context_plan")
        )
        if set(context_plan["expected_overlay_paths"]) != set(files):
            raise DraftGenerationError(
                "AsNumpy adapter overlay differs from its context plan"
            )
        if any(
            contexts.get(label) != digest
            for label, digest in context_plan["context_sha256"].items()
        ):
            raise DraftGenerationError(
                "AsNumpy adapter context differs from its context plan"
            )

    normalized = dict(report)
    normalized["files_sha256"] = files
    normalized["asnumpy_context_sha256"] = contexts
    normalized["integration_skill_files"] = list(skill_files)
    normalized["response_sha256"] = list(response_hashes)
    normalized["remote_receipts"] = normalized_receipts
    if context_plan is not None:
        normalized["asnumpy_adapter_context_plan"] = context_plan
    if status == "generated_unverified":
        if not _is_sha256(report.get("adapter_contract_sha256")):
            raise DraftGenerationError(
                "AsNumpy adapter contract binding is invalid"
            )
    else:
        repair_profile = report.get("repair_profile")
        if (
            not isinstance(repair_profile, str)
            or repair_profile not in {"compile", "precision", "runtime"}
        ):
            raise DraftGenerationError(
                "AsNumpy adapter repair profile is invalid"
            )
        source_root = report.get("source_adapter_root")
        if (
            not isinstance(source_root, str)
            or not source_root
            or len(source_root) > 4096
            or any(ord(character) < 32 for character in source_root)
            or not Path(source_root).is_absolute()
            or not _is_sha256(report.get("parent_adapter_report_sha256"))
            or not _is_sha256(report.get("diagnostic_sha256"))
        ):
            raise DraftGenerationError(
                "AsNumpy adapter repair provenance is invalid"
            )
        parent_id = report.get("parent_adapter_artifact_id")
        if (
            not isinstance(parent_id, str)
            or not re.fullmatch(r"adapter-[0-9a-f]{20}", parent_id)
        ):
            raise DraftGenerationError(
                "AsNumpy adapter parent artifact identity is invalid"
            )
        raw_paths = report.get("paths")
        if (
            not isinstance(raw_paths, list)
            or len(raw_paths) != 1
        ):
            raise DraftGenerationError("AsNumpy adapter repair path is invalid")
        repaired_paths = tuple(
            str(_safe_asnumpy_adapter_relative_path(item))
            for item in raw_paths
        )
        if len(set(repaired_paths)) != len(repaired_paths):
            raise DraftGenerationError("AsNumpy adapter repair path is duplicated")
        before = _validate_adapter_hash_map(
            report.get("before_sha256"),
            "before map",
            maximum=_MAX_ADAPTER_FILES,
            paths=True,
        )
        after = _validate_adapter_hash_map(
            report.get("after_sha256"),
            "after map",
            maximum=_MAX_ADAPTER_FILES,
            paths=True,
        )
        if (
            set(before) != set(repaired_paths)
            or set(after) != set(repaired_paths)
            or any(files.get(path) != after[path] for path in repaired_paths)
        ):
            raise DraftGenerationError(
                "AsNumpy adapter repair content hashes are inconsistent"
            )
        normalized["paths"] = list(repaired_paths)
        normalized["before_sha256"] = before
        normalized["after_sha256"] = after
        if parent_report is not None:
            if not _is_sha256(parent_report_sha256):
                raise DraftGenerationError(
                    "AsNumpy adapter parent report hash is unavailable"
                )
            parent_files = parent_report["files_sha256"]
            expected_files = dict(parent_files)
            expected_files.update(after)
            if (
                report["parent_adapter_report_sha256"] != parent_report_sha256
                or parent_id != parent_report["adapter_artifact_id"]
                or report["source_artifact_id"] != parent_report["source_artifact_id"]
                or report["skill_bundle_sha256"]
                != parent_report["skill_bundle_sha256"]
                or _generation_identity_snapshot(report)
                != _generation_identity_snapshot(parent_report)
                or (
                    schema_version == 4
                    and context_plan
                    != parent_report.get("asnumpy_adapter_context_plan")
                )
                or any(parent_files.get(path) != before[path] for path in repaired_paths)
                or expected_files != files
            ):
                raise DraftGenerationError(
                    "AsNumpy adapter repair does not derive from its parent"
                )

    artifact_id = report.get("adapter_artifact_id")
    if (
        not isinstance(artifact_id, str)
        or artifact_id != _adapter_artifact_id(normalized)
    ):
        raise DraftGenerationError("AsNumpy adapter artifact_id mismatch")
    if status == "repaired_unverified" and artifact_id == report.get(
        "parent_adapter_artifact_id"
    ):
        raise DraftGenerationError(
            "AsNumpy adapter repair must have a distinct artifact_id"
        )
    return normalized


def _validate_adapter_skill_binding(report: Dict[str, object], bundle) -> None:
    """Bind report skill claims to the verified in-memory bundle in this run."""

    if report["skill_bundle_sha256"] != getattr(bundle, "bundle_sha256", None):
        raise DraftGenerationError("AsNumpy adapter Skill bundle hash mismatch")
    files = getattr(bundle, "files", None)
    if not isinstance(files, dict):
        raise DraftGenerationError("AsNumpy adapter Skill bundle is invalid")
    for relative in report["integration_skill_files"]:
        content = files.get(relative)
        if not isinstance(content, str) or not content:
            raise DraftGenerationError(
                "AsNumpy adapter Skill file is unavailable: %s" % relative
            )


def _read_bounded_adapter_text(path: Path, relative: str) -> Tuple[str, int, str]:
    """Read prompt context with the same hard byte bound used by reloading."""

    content, payload = _read_bounded_text(
        path,
        _MAX_ADAPTER_FILE_BYTES,
        "AsNumpy adapter base file %s" % relative,
    )
    return content, len(payload), _sha256_bytes(payload)


def _load_asnumpy_adapter_report(
    root: Path,
    *,
    _seen=None,
    _depth: int = 0,
) -> Tuple[Dict[str, object], bytes, Dict[str, str]]:
    """Reload and recursively validate an adapter report and candidate tree."""

    root = Path(root)
    if _depth >= _MAX_ADAPTER_CHAIN_DEPTH:
        raise DraftGenerationError("AsNumpy adapter parent chain is too deep")
    report_path = root / "asnumpy-adapter-report.json"
    candidate_root = root / "candidate"
    if (
        root.is_symlink()
        or not root.is_dir()
        or report_path.is_symlink()
        or not report_path.is_file()
        or candidate_root.is_symlink()
        or not candidate_root.is_dir()
    ):
        raise DraftGenerationError("AsNumpy adapter base is invalid")
    try:
        identity = (root.stat().st_dev, root.stat().st_ino)
    except OSError as exc:
        raise DraftGenerationError("cannot inspect AsNumpy adapter base") from exc
    seen = set() if _seen is None else _seen
    if identity in seen:
        raise DraftGenerationError("AsNumpy adapter parent chain contains a cycle")
    seen.add(identity)
    report_text, payload = _read_bounded_text(
        report_path,
        _MAX_METADATA_BYTES,
        "AsNumpy adapter base report",
    )
    try:
        raw_report = json.loads(report_text)
    except json.JSONDecodeError as exc:
        raise DraftGenerationError(
            "AsNumpy adapter base report is unreadable"
        ) from exc

    report = _validate_adapter_report(raw_report)
    actual_hashes = _bounded_tree_hashes(
        candidate_root,
        excluded=(),
        maximum_files=_MAX_ADAPTER_FILES,
        maximum_entries=_MAX_ADAPTER_TREE_ENTRIES,
        maximum_file_bytes=_MAX_ADAPTER_FILE_BYTES,
        maximum_total_bytes=_MAX_ADAPTER_TOTAL_BYTES,
        label="AsNumpy adapter base",
        path_validator=_safe_asnumpy_adapter_relative_path,
    )
    if set(actual_hashes) != set(report["files_sha256"]):
        raise DraftGenerationError(
            "AsNumpy adapter base files differ from its report"
        )

    parent = None
    parent_sha256 = None
    if report["status"] == "repaired_unverified":
        parent, parent_payload, _parent_hashes = _load_asnumpy_adapter_report(
            Path(report["source_adapter_root"]),
            _seen=seen,
            _depth=_depth + 1,
        )
        parent_sha256 = _sha256_bytes(parent_payload)
    report = _validate_adapter_report(
        report,
        candidate_hashes=actual_hashes,
        parent_report=parent,
        parent_report_sha256=parent_sha256,
    )
    return report, payload, actual_hashes


def _validate_generation(
    generation: object,
    *,
    maximum_attempts: int = _MAX_TRANSPORT_ATTEMPTS,
) -> Dict[str, object]:
    maximum_attempts = _require_exact_int(
        maximum_attempts,
        "generation validation attempt limit",
        minimum=1,
        maximum=_MAX_TRANSPORT_ATTEMPTS,
    )
    if (
        not isinstance(generation, dict)
        or frozenset(generation)
        not in {_GENERATION_FIELDS_V1, _GENERATION_FIELDS_V2}
    ):
        raise DraftGenerationError(
            "existing draft generation evidence schema is invalid"
        )

    mode = generation.get("mode")
    if mode not in {"remote_llm", "injected_test_double"}:
        raise DraftGenerationError("existing draft generation mode is invalid")
    for name in ("provider", "model", "configuration_source"):
        value = generation.get(name)
        if (
            not isinstance(value, str)
            or not value
            or len(value) > 512
            or any(ord(character) < 32 for character in value)
        ):
            raise DraftGenerationError(
                "existing draft generation field is invalid: %s" % name
            )
    generic_schema = frozenset(generation) == _GENERATION_FIELDS_V2
    protocol = generation.get(
        "protocol", OPENAI_CHAT_COMPLETIONS_PROTOCOL
    )
    aliases = generation.get("response_model_aliases", ())
    try:
        identity = require_model_configuration(
            generation.get("provider"),
            generation.get("model"),
            protocol=protocol,
            response_model_aliases=aliases,
        )
    except DraftGenerationError as exc:
        raise DraftGenerationError(
            "existing draft configured-model identity is invalid: %s" % exc
        ) from exc
    if generation.get("local_validation") != "passed":
        raise DraftGenerationError(
            "existing draft must prove local validation passed"
        )
    if generation.get("fallback") != "disabled":
        raise DraftGenerationError("existing draft must prove fallback was disabled")

    attempt_count = _require_exact_int(
        generation.get("attempt_count"),
        "existing draft attempt count",
        minimum=1,
        maximum=maximum_attempts,
    )
    transport_count = _require_exact_int(
        generation.get("transport_attempt_count"),
        "existing draft transport attempt count",
        minimum=1,
        maximum=maximum_attempts,
    )
    successful_http_count = _require_exact_int(
        generation.get("successful_http_response_count"),
        "existing draft successful HTTP response count",
        maximum=maximum_attempts,
    )
    content_count = _require_exact_int(
        generation.get("content_response_count"),
        "existing draft content response count",
        maximum=maximum_attempts,
    )
    if transport_count != attempt_count:
        raise DraftGenerationError(
            "existing draft transport/attempt counts do not match"
        )

    response_hashes = generation.get("response_sha256")
    if (
        not isinstance(response_hashes, list)
        or not response_hashes
        or len(response_hashes) != content_count
        or content_count > transport_count
        or successful_http_count > transport_count
        or any(not _is_sha256(item) for item in response_hashes)
    ):
        raise DraftGenerationError(
            "existing draft response hashes/counters are inconsistent"
        )

    raw_receipts = generation.get("remote_receipts")
    if (
        not isinstance(raw_receipts, list)
        or len(raw_receipts) > maximum_attempts
    ):
        raise DraftGenerationError("existing draft remote receipts are invalid")
    if mode == "remote_llm":
        try:
            if generic_schema:
                receipts = [
                    validate_configured_model_attempt_receipt(
                        item,
                        expected_model=identity.model,
                        expected_provider=identity.provider,
                        expected_protocol=identity.protocol,
                        response_model_aliases=identity.response_model_aliases,
                    )
                    for item in raw_receipts
                ]
            else:
                receipts = [
                    validate_configured_model_receipt(
                        item,
                        expected_model=identity.model,
                        expected_provider=identity.provider,
                        expected_protocol=identity.protocol,
                    )
                    for item in raw_receipts
                ]
        except DraftGenerationError as exc:
            raise DraftGenerationError(
                "existing remote draft configured-model identity is invalid: %s" % exc
            ) from exc
    else:
        receipts = [_validate_remote_receipt(item) for item in raw_receipts]

    normalized = dict(generation)
    normalized["response_sha256"] = list(response_hashes)
    normalized["remote_receipts"] = receipts
    if mode == "remote_llm":
        if generation.get("origin_action") != "fresh_remote":
            raise DraftGenerationError(
                "existing remote draft origin action is invalid"
            )
        if (
            successful_http_count != len(receipts)
            or content_count > successful_http_count
            or not receipts
        ):
            raise DraftGenerationError(
                "existing remote draft receipt count is inconsistent"
            )
        if generic_schema:
            validate_configured_model_receipt(
                receipts[-1],
                expected_model=identity.model,
                expected_provider=identity.provider,
                expected_protocol=identity.protocol,
                response_model_aliases=identity.response_model_aliases,
            )
            if receipts[-1]["content_sha256"] != response_hashes[-1]:
                raise DraftGenerationError(
                    "existing remote draft final receipt is not the accepted response"
                )
        receipt_content_hashes = [
            item["content_sha256"]
            for item in receipts
            if item["content_sha256"] is not None
        ]
        if receipt_content_hashes != response_hashes:
            raise DraftGenerationError(
                "existing remote draft receipt/content hashes are inconsistent"
            )
    else:
        if (
            generation.get("origin_action") != "injected_test_double"
            or generation.get("configuration_source") != "injected"
            or receipts
            or successful_http_count != 0
        ):
            raise DraftGenerationError(
                "existing injected draft origin evidence is invalid"
            )
    return normalized


def _require_project_layout(
    names,
    *,
    file_contents: Optional[Dict[str, str]] = None,
    root: Optional[Path] = None,
) -> None:
    names = set(names)
    for required in ("CMakeLists.txt", "build.sh"):
        if required not in names:
            raise DraftGenerationError(
                "draft manifest is missing %s" % required
            )
    required_dirs = (
        "op_host",
        "op_kernel",
        "op_api",
        "tests",
        "asnumpy_adapter",
    )
    if all(
        any(PurePosixPath(name).parts[:1] == (required_dir,) for name in names)
        for required_dir in required_dirs
    ):
        return
    # A cannbot-skill ASC result is valid when its manifest proves the same
    # ACLNN two-stage boundary.  The helper is deliberately strict and cannot
    # be reached without concrete file contents or a sealed workspace.
    if _validate_asc_two_stage_layout(
        names, file_contents=file_contents, root=root
    ):
        return
    missing = next(
        (
            required_dir
            for required_dir in required_dirs
            if not any(
                PurePosixPath(name).parts[:1] == (required_dir,)
                for name in names
            )
        ),
        required_dirs[0],
    )
    raise DraftGenerationError("draft manifest is missing %s/" % missing)


def _draft_artifact_id(
    gap_id: str,
    files: Dict[str, str],
    generation: Dict[str, object],
) -> str:
    identity = json.dumps(
        {
            "gap_id": gap_id,
            "files": files,
            "generation": generation,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "draft-" + hashlib.sha256(identity).hexdigest()[:20]


def _load_existing_draft(draft_dir: Path, spec: GapSpec) -> OperatorDraft:
    manifest_path = draft_dir / "operator-artifact.json"
    if draft_dir.is_symlink() or not draft_dir.is_dir():
        raise DraftGenerationError("existing draft must not contain symlinks")
    manifest_text, _manifest_payload = _read_bounded_text(
        manifest_path,
        _MAX_METADATA_BYTES,
        "existing draft manifest",
    )
    try:
        data = json.loads(manifest_text)
    except json.JSONDecodeError as exc:
        raise DraftGenerationError(
            "existing draft has no valid operator-artifact.json"
        ) from exc
    if not isinstance(data, dict):
        raise DraftGenerationError("existing draft manifest must be an object")
    if frozenset(data) != _ARTIFACT_FIELDS:
        raise DraftGenerationError("existing draft manifest schema is invalid")
    schema_version = data.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != ARTIFACT_SCHEMA_VERSION
    ):
        raise DraftGenerationError("existing draft has an unsupported schema")
    if data.get("gap_id") != spec.gap_id:
        raise DraftGenerationError("existing draft belongs to another gap")
    if data.get("status") != "generated_unverified":
        raise DraftGenerationError(
            "existing draft status must remain generated_unverified"
        )
    if data.get("gates") != _EXPECTED_UNVERIFIED_GATES:
        raise DraftGenerationError(
            "existing draft gates do not match the unverified lifecycle"
        )
    if data.get("workspace") != str(draft_dir):
        raise DraftGenerationError("existing draft workspace identity mismatch")
    if (
        not isinstance(data.get("summary"), str)
        or len(data["summary"]) > _MAX_SUMMARY_CHARS
    ):
        raise DraftGenerationError("existing draft summary is invalid")
    normalized_generation = _validate_generation(data.get("generation"))
    files = data.get("files")
    if (
        not isinstance(files, dict)
        or not files
        or len(files) > _MAX_FILES
    ):
        raise DraftGenerationError("existing draft file manifest is invalid")
    normalized_files: Dict[str, str] = {}
    for relative, expected in files.items():
        path_key = str(_safe_relative_path(relative))
        if path_key != relative:
            raise DraftGenerationError(
                "existing draft file path is not normalized"
            )
        if not _is_sha256(expected):
            raise DraftGenerationError(
                "existing draft digest is invalid: %s" % path_key
            )
        normalized_files[path_key] = expected
    if len(normalized_files) != len(files):
        raise DraftGenerationError("existing draft has duplicate normalized paths")
    actual_files = _bounded_tree_hashes(
        draft_dir,
        excluded=("operator-artifact.json",),
        maximum_files=_MAX_FILES,
        maximum_entries=_MAX_OPERATOR_TREE_ENTRIES,
        maximum_file_bytes=_MAX_TOTAL_BYTES,
        maximum_total_bytes=_MAX_TOTAL_BYTES,
        label="existing draft",
    )
    if set(actual_files) != set(normalized_files):
        raise DraftGenerationError(
            "existing draft file set differs from its manifest"
        )
    _require_project_layout(normalized_files, root=draft_dir)

    for relative, expected in normalized_files.items():
        if actual_files[relative] != expected:
            raise DraftGenerationError(
                "existing draft hash mismatch: %s" % relative
            )
    if (
        normalized_generation["attempt_count"]
        > _MAX_FRESH_TRANSPORT_ATTEMPTS
        and "docs/numpilot-repair-provenance.json" not in normalized_files
    ):
        raise DraftGenerationError(
            "extended transport evidence requires sealed repair provenance"
        )
    artifact_id = _draft_artifact_id(
        spec.gap_id,
        normalized_files,
        normalized_generation,
    )
    if data.get("artifact_id") != artifact_id:
        raise DraftGenerationError("existing draft artifact_id mismatch")
    return OperatorDraft(
        schema_version=schema_version,
        artifact_id=artifact_id,
        gap_id=str(data["gap_id"]),
        workspace=str(draft_dir),
        summary=str(data.get("summary", "")),
        status="generated_unverified",
        files=normalized_files,
        generation=normalized_generation,
        gates=dict(_EXPECTED_UNVERIFIED_GATES),
    )


def _validate_operator_repair_skill_binding(
    report: Dict[str, object],
    skill_bundle,
) -> None:
    """Bind an operator repair report to the verified bundle in this run."""

    if skill_bundle is None:
        return
    if report["skill_bundle_sha256"] != getattr(
        skill_bundle, "bundle_sha256", None
    ):
        raise DraftGenerationError("repair base Skill bundle hash mismatch")
    files = getattr(skill_bundle, "files", None)
    if not isinstance(files, dict) or any(
        not isinstance(files.get(relative), str) or not files[relative]
        for relative in report["cannbot_repair_skill_files"]
    ):
        raise DraftGenerationError(
            "repair base Skill file binding is unavailable"
        )


def _load_repair_candidate(
    candidate_dir: Path,
    spec: GapSpec,
    signed_source: OperatorDraft,
    skill_bundle=None,
) -> Tuple[Dict[str, str], str, Dict[str, object]]:
    """Verify a repaired overlay and return its complete current file hashes.

    The signed ``operator-artifact.json`` remains immutable in every repair
    candidate.  The repair report therefore acts as a hash-bound overlay over
    that original artifact.  Reports without the complete candidate file map
    cannot prove this binding and are rejected with migration guidance.
    """

    candidate_dir = Path(candidate_dir)
    report_path = candidate_dir / "numpilot-repair-report.json"
    artifact_path = candidate_dir / "operator-artifact.json"
    if (
        candidate_dir.is_symlink()
        or not candidate_dir.is_dir()
    ):
        raise DraftGenerationError("repair base must not contain symlinks")
    report_text, report_payload = _read_bounded_text(
        report_path,
        _MAX_METADATA_BYTES,
        "repair base report",
    )
    artifact_payload = _read_bounded_bytes(
        artifact_path,
        _MAX_METADATA_BYTES,
        "repair base signed artifact",
    )
    signed_artifact_payload = _read_bounded_bytes(
        Path(signed_source.workspace) / "operator-artifact.json",
        _MAX_METADATA_BYTES,
        "signed source artifact",
    )
    try:
        report = json.loads(report_text)
    except json.JSONDecodeError as exc:
        raise DraftGenerationError(
            "repair base has no valid repair report or signed artifact"
        ) from exc
    if artifact_payload != signed_artifact_payload:
        raise DraftGenerationError("repair base changed the signed artifact manifest")
    if not isinstance(report, dict):
        raise DraftGenerationError("repair base report must be an object")
    report_schema = report.get("schema_version")
    expected_repair_fields = (
        _REPAIR_REPORT_FIELDS_V2
        if report_schema == 2
        else _REPAIR_REPORT_FIELDS_V1
    )
    if frozenset(report) != expected_repair_fields:
        if "candidate_files_sha256" not in report:
            raise DraftGenerationError(
                "legacy repair report cannot prove its complete audit binding; "
                "regenerate it (fail-closed)"
            )
        raise DraftGenerationError("repair base report schema is invalid")
    if (
        isinstance(report.get("schema_version"), bool)
        or not isinstance(report.get("schema_version"), int)
        or report.get("schema_version") not in (1, 2)
        or report.get("status") != "repaired_unverified"
        or report.get("source_artifact_id") != signed_source.artifact_id
        or report.get("gap_id") != spec.gap_id
        or report.get("gates") != _EXPECTED_UNVERIFIED_GATES
        or report.get("actual_remote_call") is not True
    ):
        raise DraftGenerationError("repair base report identity is invalid")

    source_workspace = report.get("source_workspace")
    if (
        not isinstance(source_workspace, str)
        or not source_workspace
        or len(source_workspace) > 4096
        or any(ord(character) < 32 for character in source_workspace)
        or not Path(source_workspace).is_absolute()
        or report.get("signed_source_workspace") != signed_source.workspace
    ):
        raise DraftGenerationError("repair base workspace binding is invalid")
    parent_report_hash = report.get("parent_repair_report_sha256")
    if parent_report_hash is not None and not _is_sha256(parent_report_hash):
        raise DraftGenerationError("repair base parent binding is invalid")
    for name in (
        "diagnostic_sha256",
        "generation_request_sha256",
        "skill_bundle_sha256",
    ):
        if not _is_sha256(report.get(name)):
            raise DraftGenerationError(
                "repair base audit hash is invalid: %s" % name
            )
    repair_skill_files = report.get("cannbot_repair_skill_files")
    if (
        not isinstance(repair_skill_files, list)
        or not repair_skill_files
        or len(repair_skill_files) > 64
        or len(set(repair_skill_files)) != len(repair_skill_files)
    ):
        raise DraftGenerationError("repair base Skill file binding is invalid")
    normalized_skill_files = []
    for raw_path in repair_skill_files:
        if not isinstance(raw_path, str):
            raise DraftGenerationError(
                "repair base Skill file binding is invalid"
            )
        path = PurePosixPath(raw_path)
        if (
            path.is_absolute()
            or not path.parts
            or "." in path.parts
            or ".." in path.parts
            or str(path) != raw_path
            or len(raw_path) > _MAX_PATH_CHARS
            or any(ord(character) < 32 for character in raw_path)
        ):
            raise DraftGenerationError(
                "repair base Skill file binding is invalid"
            )
        normalized_skill_files.append(raw_path)

    paths = report.get("paths")
    before = report.get("before_sha256")
    after = report.get("after_sha256")
    if (
        not isinstance(paths, list)
        or not paths
        or len(paths) > _MAX_REPAIR_PATHS
        or not isinstance(before, dict)
        or not isinstance(after, dict)
    ):
        raise DraftGenerationError("repair base hash overlay is invalid")
    if any(not isinstance(path, str) for path in paths):
        raise DraftGenerationError("repair base path hashes are invalid")
    normalized_paths = tuple(str(_safe_relative_path(path)) for path in paths)
    if (
        tuple(paths) != normalized_paths
        or len(set(normalized_paths)) != len(normalized_paths)
        or len(before) != len(normalized_paths)
        or len(after) != len(normalized_paths)
        or set(before) != set(normalized_paths)
        or set(after) != set(normalized_paths)
        or any(path not in signed_source.files for path in normalized_paths)
        or any(not _is_sha256(value) for value in before.values())
        or any(not _is_sha256(value) for value in after.values())
    ):
        raise DraftGenerationError("repair base path hashes are invalid")
    if _generation_identity_snapshot(report) != _generation_identity_snapshot(
        signed_source.generation
    ):
        raise DraftGenerationError(
            "repair base configured-model identity differs from signed generation"
        )
    report_attempts = _require_exact_int(
        report.get("attempt_count"),
        "repair base attempt count",
        minimum=1,
        maximum=len(normalized_paths) * 3,
    )
    response_hashes = report.get("response_sha256")
    raw_receipts = report.get("remote_receipts")
    if (
        not isinstance(response_hashes, list)
        or not response_hashes
        or len(response_hashes) > report_attempts
        or any(not _is_sha256(item) for item in response_hashes)
        or not isinstance(raw_receipts, list)
        or not raw_receipts
        or len(raw_receipts) > report_attempts
    ):
        raise DraftGenerationError("repair base transport evidence is invalid")
    identity = require_model_configuration(
        report.get("provider"),
        report.get("model"),
        protocol=report.get(
            "protocol", OPENAI_CHAT_COMPLETIONS_PROTOCOL
        ),
        response_model_aliases=report.get("response_model_aliases", ()),
    )
    receipt_validator = (
        validate_configured_model_attempt_receipt
        if report.get("schema_version") == 2
        else validate_configured_model_receipt
    )
    receipts = [
        receipt_validator(
            item,
            expected_model=identity.model,
            expected_provider=identity.provider,
            expected_protocol=identity.protocol,
            response_model_aliases=identity.response_model_aliases,
        )
        for item in raw_receipts
    ]
    content_hashes = [
        item["content_sha256"]
        for item in receipts
        if item["content_sha256"] is not None
    ]
    if (
        content_hashes != response_hashes
    ):
        raise DraftGenerationError(
            "repair base receipt/content hashes are inconsistent"
        )
    if report.get("schema_version") == 2:
        validate_configured_model_receipt(
            receipts[-1],
            expected_model=identity.model,
            expected_provider=identity.provider,
            expected_protocol=identity.protocol,
            response_model_aliases=identity.response_model_aliases,
        )
        if receipts[-1]["content_sha256"] != response_hashes[-1]:
            raise DraftGenerationError(
                "repair base final receipt is not the accepted response"
            )
    report = dict(report)
    report["cannbot_repair_skill_files"] = normalized_skill_files
    report["response_sha256"] = list(response_hashes)
    report["remote_receipts"] = receipts

    candidate_files = report.get("candidate_files_sha256")
    if (
        not isinstance(candidate_files, dict)
        or not candidate_files
        or len(candidate_files) > _MAX_FILES
    ):
        raise DraftGenerationError("repair base complete file map is invalid")
    normalized_files = {}
    for relative, digest in candidate_files.items():
        path = str(_safe_relative_path(relative))
        if relative != path or path in normalized_files:
            raise DraftGenerationError(
                "repair base complete file map has duplicate or non-normalized paths"
            )
        if not _is_sha256(digest):
            raise DraftGenerationError(
                "repair base digest is invalid: %s" % path
            )
        normalized_files[path] = digest
    if set(normalized_files) != set(signed_source.files):
        raise DraftGenerationError(
            "repair base complete file map differs from signed artifact"
        )
    if any(normalized_files[path] != after[path] for path in normalized_paths):
        raise DraftGenerationError(
            "repair base overlay differs from its complete file map"
        )

    _validate_operator_repair_skill_binding(report, skill_bundle)

    actual_files = _bounded_tree_hashes(
        candidate_dir,
        excluded=("operator-artifact.json", "numpilot-repair-report.json"),
        maximum_files=_MAX_FILES,
        maximum_entries=_MAX_OPERATOR_TREE_ENTRIES,
        maximum_file_bytes=_MAX_TOTAL_BYTES,
        maximum_total_bytes=_MAX_TOTAL_BYTES,
        label="repair base",
    )
    if set(actual_files) != set(normalized_files):
        raise DraftGenerationError("repair base file set differs from its report")
    for relative, expected in normalized_files.items():
        if actual_files[relative] != expected:
            raise DraftGenerationError(
                "repair base hash mismatch: %s" % relative
            )
    return normalized_files, _sha256_bytes(report_payload), dict(report)


def _load_repair_chain(
    candidate_dir: Path,
    spec: GapSpec,
    signed_source: OperatorDraft,
    skill_bundle=None,
) -> Tuple[Dict[str, str], Tuple[Dict[str, object], ...]]:
    """Validate a complete parent-linked repair chain, oldest report first."""

    current_dir = Path(candidate_dir)
    seen = set()
    reverse_reports = []
    final_files = None
    while True:
        if len(reverse_reports) >= _MAX_REPAIR_CHAIN_DEPTH:
            raise DraftGenerationError("repair chain exceeds the depth limit")
        identity = os.path.abspath(str(current_dir))
        if identity in seen:
            raise DraftGenerationError("repair chain contains a directory cycle")
        seen.add(identity)
        current_files, current_hash, report = _load_repair_candidate(
            current_dir,
            spec,
            signed_source,
            skill_bundle,
        )
        if final_files is None:
            final_files = current_files
        reverse_reports.append(dict(report, report_sha256=current_hash))

        parent_hash = report.get("parent_repair_report_sha256")
        source_workspace = report.get("source_workspace")
        if not isinstance(source_workspace, str) or not source_workspace:
            raise DraftGenerationError("repair chain source workspace is invalid")
        if parent_hash is None:
            if source_workspace != signed_source.workspace:
                raise DraftGenerationError(
                    "first repair does not descend from the signed artifact"
                )
            for path, before_hash in report["before_sha256"].items():
                if signed_source.files[path] != before_hash:
                    raise DraftGenerationError(
                        "first repair before hash differs from signed artifact"
                    )
            break
        if not _is_sha256(parent_hash):
            raise DraftGenerationError("repair chain parent hash is invalid")

        parent_dir = Path(source_workspace)
        parent_files, actual_parent_hash, _parent_report = _load_repair_candidate(
            parent_dir,
            spec,
            signed_source,
            skill_bundle,
        )
        if actual_parent_hash != parent_hash:
            raise DraftGenerationError("repair chain parent report hash mismatch")
        for path, before_hash in report["before_sha256"].items():
            if parent_files[path] != before_hash:
                raise DraftGenerationError(
                    "repair chain before hash differs from its parent"
                )
        expected_current = dict(parent_files)
        expected_current.update(report["after_sha256"])
        if expected_current != current_files:
            raise DraftGenerationError(
                "repair chain current files do not derive from its parent"
            )
        current_dir = parent_dir

    return dict(final_files or {}), tuple(reversed(reverse_reports))


def _direct_generation_stages(name: str):
    """Return the fixed, auditable stage/file/reference plan."""

    add_root = _SKILL_PREFIX + "/references/add_example/"
    groups = (
        (
            "build",
            (
                "CMakeLists.txt",
                "build.sh",
                "op_host/CMakeLists.txt",
                "op_kernel/CMakeLists.txt",
            ),
            (
                _SKILL_PREFIX + "/SKILL.md",
                _SKILL_PREFIX + "/references/build-deploy-guide.md",
                add_root + "CMakeLists.txt",
                add_root + "build.sh",
                add_root + "op_host/CMakeLists.txt",
                add_root + "op_kernel/CMakeLists.txt",
            ),
        ),
        (
            "host",
            (
                "op_host/%s_def.cpp" % name,
                "op_host/%s_infershape.cpp" % name,
                "op_host/arch22/%s_tiling.cpp" % name,
            ),
            (
                _SKILL_PREFIX + "/SKILL.md",
                _SKILL_PREFIX + "/references/basic-guide.md",
                add_root + "op_host/add_example_def.cpp",
                add_root + "op_host/add_example_infershape.cpp",
                add_root + "op_host/arch22/add_example_tiling.cpp",
            ),
        ),
        (
            "kernel",
            (
                "op_kernel/%s_arch22.cpp" % name,
                "op_kernel/arch22/%s.h" % name,
                "op_kernel/arch22/%s_tiling_data.h" % name,
                "op_kernel/arch22/%s_tiling_key.h" % name,
            ),
            (
                _SKILL_PREFIX + "/SKILL.md",
                _SKILL_PREFIX + "/references/basic-guide.md",
                add_root + "op_kernel/add_example_arch22.cpp",
                add_root + "op_kernel/arch22/add_example.h",
                add_root + "op_kernel/arch22/add_example_tiling_data.h",
                add_root + "op_kernel/arch22/add_example_tiling_key.h",
            ),
        ),
        (
            "api",
            (
                "op_api/aclnn_%s.h" % name,
                "op_api/aclnn_%s.cpp" % name,
                "op_api/%s.h" % name,
                "op_api/%s.cpp" % name,
            ),
            (
                _SKILL_PREFIX + "/SKILL.md",
                add_root + "op_api/aclnn_add_example.h",
                add_root + "op_api/aclnn_add_example.cpp",
                add_root + "op_api/add_example.h",
                add_root + "op_api/add_example.cpp",
            ),
        ),
        (
            "test_adapter",
            (
                "tests/test_%s.py" % name,
                "asnumpy_adapter/%s.json" % name,
            ),
            (
                _SKILL_PREFIX + "/SKILL.md",
            ),
        ),
    )
    return tuple(
        (
            "%s_%02d" % (group_name, file_index),
            (path,),
            tuple(dict.fromkeys(
                (*CANNBOT_GENERATION_SKILL_FILES, *references)
            )),
        )
        for group_name, paths, references in groups
        for file_index, path in enumerate(paths, start=1)
    )


def _should_use_staged_direct(spec: GapSpec) -> bool:
    """Use bounded per-file generation for every direct Ascend C candidate.

    Direct NumPy gaps and explicit OperatorSpec gaps share the same fixed
    17-file project contract.  Restricting staged generation to OperatorSpec
    made a native gap such as ``numpy.cbrt`` fall back to one very large model
    response, which is both less auditable and prone to gateway timeouts.
    """

    return spec.route == "ascendc_registry_candidate"


def _compact_generation_contract(request: str) -> str:
    marker = "\n## Embedded official Huawei CANNBot skill\n"
    return request.split(marker, 1)[0].strip()


def _stage_invariants(spec: GapSpec, path: str) -> str:
    """Render deterministic cross-file constraints for every model stage."""

    signatures = sorted({
        (
            int(item.get("positional_args", 0)),
            tuple(str(name) for name in item.get("keyword_args", ())),
        )
        for item in spec.signature_variants
    })
    signature_text = ", ".join(
        "%d positional; keywords=%s" % (
            positional,
            list(keywords),
        )
        for positional, keywords in signatures
    )
    rules = [
        "Project-wide invariants (mandatory):",
        "- C++ OpType is `%s`; file/API stem is `%s`; public API is `%s`." % (
            "".join(part.capitalize() for part in spec.api_name.split("_")),
            spec.api_name,
            spec.asnumpy_api,
        ),
        "- Source call signatures are exactly: %s." % signature_text,
        "- Admitted dtypes are exactly: %s; do not add convenient extra dtypes." % (
            ", ".join(spec.dtype_rules),
        ),
        "- Admitted shape rules are exactly: %s." % (
            ", ".join(spec.shape_rules),
        ),
        "- Preserve the NumPy oracle parameter order. Host definition, infer-shape, "
        "tiling, kernel entry, L0/L2 ACLNN declarations/definitions, tests, and adapter "
        "must agree on input/output count, order, names, dtypes, and shapes.",
    ]
    if path.startswith("tests/"):
        rules.extend((
            "- This test file must contain pure NumPy oracle tests that collect and pass "
            "when `asnumpy` and the custom operator are absent.",
            "- Do not import `asnumpy` at module scope. Runtime/public-API tests must be "
            "separately guarded or skipped and must never disguise a missing runtime as "
            "a CPU golden-test pass.",
        ))
    if path.endswith("_tiling_key.h"):
        rules.append(
            "- Declare every template parameter used by both ASCENDC_TPL_SEL_PARAM in "
            "host tiling and the kernel entry template, using the official arch22 pattern."
        )
    return "\n".join(rules)


class LlmDraftProvider:
    """Generate source files but deliberately never compile or execute them."""

    def __init__(
        self,
        chat: Optional[Callable[[str, str], str]] = None,
        *,
        max_attempts: int = 3,
        provider_label: str = "injected",
        model_name: str = "test-double",
        configuration_status=None,
        configuration_loader: Optional[Callable] = None,
        skill_bundle=None,
        staged_direct: bool = False,
        repair_skill_files: Optional[Tuple[str, ...]] = None,
        operator_goal: str = "accuracy",
    ):
        if max_attempts not in (1, 2, 3):
            raise ValueError("max_attempts must be 1, 2, or 3")
        self._chat = chat
        self.max_attempts = max_attempts
        self.provider_label = provider_label
        self.model_name = model_name
        self._configuration_status = configuration_status
        self._configuration_loader = configuration_loader
        self._skill_bundle = skill_bundle
        self._staged_direct = bool(staged_direct)
        self._repair_skill_files = tuple(
            repair_skill_files or CANNBOT_REPAIR_SKILL_FILES
        )
        if operator_goal != "accuracy":
            raise ValueError(
                "operator_goal must be accuracy; performance generation is "
                "not implemented"
            )
        self._operator_goal = operator_goal
        if (
            not self._repair_skill_files
            or len(set(self._repair_skill_files)) != len(self._repair_skill_files)
            or any(not isinstance(path, str) or not path for path in self._repair_skill_files)
        ):
            raise ValueError("repair_skill_files must be unique non-empty paths")
        if self._staged_direct and self._skill_bundle is None:
            raise ValueError("staged direct generation requires a verified skill bundle")
        self._run_actions: Dict[str, str] = {}

    def action_for(self, draft: OperatorDraft) -> str:
        """Return what this provider instance did during the current run."""

        return self._run_actions.get(draft.workspace, "unknown")

    def _resolve_chat(self):
        if self._chat is not None:
            return self._chat, {
                "mode": "injected_test_double",
                "provider": self.provider_label,
                "model": self.model_name,
                "protocol": OPENAI_CHAT_COMPLETIONS_PROTOCOL,
                "response_model_aliases": [],
                "configuration_source": "injected",
                "origin_action": "injected_test_double",
                "remote_receipts": [],
            }
        from numpilot.llm_config import load_llm_config
        from numpilot.model_gateway import ModelGateway

        status = self._configuration_status
        if status is None:
            loader = self._configuration_loader or load_llm_config
            status = loader()
            if status is None:
                raise DraftGenerationError(
                    "LLM is not configured; run configure-llm --check"
                )
            self._configuration_status = status
        if not status.configured:
            raise DraftGenerationError(
                "LLM is not configured; run configure-llm --check"
            )
        identity = require_model_configuration(
            status.provider,
            status.model,
            protocol=getattr(
                status, "protocol", OPENAI_CHAT_COMPLETIONS_PROTOCOL
            ),
            response_model_aliases=getattr(
                status, "response_model_aliases", ()
            ),
        )
        # All production gap-generation traffic goes through the typed gateway.
        # The gateway still uses LlmEngine as its transport, while keeping task,
        # role, request-hash and receipt validation at one boundary. Injected
        # test-doubles continue to return from the early branch above.
        gateway = ModelGateway(configuration_loader=lambda: status)
        gateway_chat = gateway.as_chat(
            task="gap_generation", role="operator-generator"
        )

        def chat_with_legacy_generation_errors(system, user, **options):
            try:
                return gateway_chat(system, user, **options)
            except Exception as exc:
                # Keep the provider's existing retry/receipt behavior while the
                # production request itself is owned by ModelGateway.
                cause = getattr(exc, "__cause__", None)
                if cause is not None:
                    from numpilot.engines.llm import LlmResponseError
                    from numpilot.model_contract import ModelContractError
                    if isinstance(cause, LlmResponseError):
                        raise cause
                    if isinstance(cause, ModelContractError):
                        raise DraftGenerationError(str(cause)) from None
                raise

        return chat_with_legacy_generation_errors, {
            "mode": "remote_llm",
            "provider": identity.provider,
            "protocol": identity.protocol,
            "model": identity.model,
            "response_model_aliases": list(identity.response_model_aliases),
            "configuration_source": status.source,
            "origin_action": "fresh_remote",
            "remote_receipts": [],
        }

    def _generate_staged_direct(
        self,
        chat,
        generation_base,
        spec: GapSpec,
        generation_request: str,
    ):
        """Generate a direct operator as bounded per-file receipt-bound stages."""

        contract = _compact_generation_contract(generation_request)
        summaries = []
        collected = []
        response_hashes = []
        remote_receipts = []
        total_attempts = 0
        stages = _direct_generation_stages(spec.api_name)
        for stage_index, (stage, paths, references) in enumerate(stages, start=1):
            failure = ""
            failures = []
            for stage_attempt in range(1, self.max_attempts + 1):
                total_attempts += 1
                prompt = "".join((
                    "Direct local generation stage %d/%d: %s.\n" % (
                        stage_index,
                        len(stages),
                        stage,
                    ),
                    "Return exactly these paths and no others:\n",
                    "\n".join("- `%s`" % path for path in paths),
                    "\n\nEvery file must be complete and mutually consistent with files "
                    "from the other deterministic stages.\n\n",
                    "Operator goal: accuracy-first. Complete semantic, build, "
                    "runtime, and numerical correctness; do not spend this "
                    "generation stage on performance optimization.\n\n",
                    _stage_invariants(spec, paths[0]),
                    "\n\n",
                    contract,
                    "\n\n",
                    self._skill_bundle.prompt_context(references),
                ))
                if failure:
                    prompt += (
                        "\nThe previous response for this stage was rejected: "
                        + failure[:500]
                        + "\nReturn a corrected NUMPILOT_BUNDLE_V1 bundle."
                    )
                try:
                    result = chat(_STAGED_SYSTEM_PROMPT, prompt)
                    if not isinstance(result, tuple) or len(result) != 2:
                        raise DraftGenerationError(
                            "remote model call returned no auditable receipt"
                        )
                    response, receipt = result
                    _record_generation_attempt(
                        remote_receipts,
                        response_hashes,
                        receipt,
                        response,
                        generation_base,
                    )
                    summary, files = _validate_stage_bundle(response, paths)
                    summaries.append("%s: %s" % (stage, summary))
                    collected.extend(files)
                    break
                except DraftGenerationError as exc:
                    failure = str(exc)
                    failures.append(
                        "%s attempt %d: %s"
                        % (stage, stage_attempt, failure[:500])
                    )
                except Exception as exc:
                    from numpilot.engines.llm import LlmResponseError

                    if isinstance(exc, LlmResponseError):
                        failed_receipt = _validate_generation_attempt_receipt(
                            exc.receipt, None, generation_base
                        )
                        remote_receipts.append(failed_receipt)
                    status_code = getattr(exc, "status_code", None)
                    if isinstance(status_code, int) and not isinstance(
                        status_code, bool
                    ):
                        failure = (
                            "remote model call failed: HTTP status %d" % status_code
                        )
                    else:
                        failure = (
                            "remote model call failed: %s" % type(exc).__name__
                        )
                    failures.append(
                        "%s attempt %d: %s"
                        % (stage, stage_attempt, failure[:500])
                    )
            else:
                error = DraftGenerationError(
                    "staged model generation failed after %d total request(s): %s"
                    % (total_attempts, "; ".join(failures))
                )
                # Keep transport accounting available to the caller without
                # putting response text into a durable error/report.  The
                # pipeline consumes these bounded attributes when the caller
                # explicitly requests multi-gap continuation.
                error.request_count = total_attempts
                error.receipt_count = len(remote_receipts)
                error.actual_remote_call = True
                raise error

        names = [str(path) for path, _content in collected]
        if len(names) != len(set(names)):
            raise DraftGenerationError("staged generation produced duplicate files")
        total_bytes = sum(
            len(content.encode("utf-8")) for _path, content in collected
        )
        if total_bytes > _MAX_TOTAL_BYTES:
            raise DraftGenerationError("staged generation exceeds the size limit")
        project_files = {str(path): content for path, content in collected}
        _require_project_layout(project_files, file_contents=project_files)
        return (
            "; ".join(summaries)[:_MAX_SUMMARY_CHARS],
            tuple(collected),
            response_hashes,
            remote_receipts,
            total_attempts,
        )

    def repair_to_candidate(
        self,
        spec: GapSpec,
        draft_dir: Path,
        candidate_dir: Path,
        *,
        paths: Tuple[str, ...],
        diagnostic: str,
        generation_request: str,
        base_candidate_dir: Optional[Path] = None,
    ) -> Dict[str, object]:
        """Repair selected files into a new candidate without mutating a draft."""

        if self._skill_bundle is None:
            raise DraftGenerationError("LLM repair requires a verified skill bundle")
        draft_dir = Path(draft_dir).absolute()
        source = _load_existing_draft(draft_dir, spec)
        repair_source_dir = draft_dir
        current_files = dict(source.files)
        parent_repair_sha256 = None
        repair_reports: Tuple[Dict[str, object], ...] = ()
        if base_candidate_dir is not None:
            repair_source_dir = Path(base_candidate_dir).absolute()
            current_files, repair_reports = _load_repair_chain(
                repair_source_dir,
                spec,
                source,
                self._skill_bundle,
            )
            parent_repair_sha256 = repair_reports[-1]["report_sha256"]
        candidate_dir = Path(candidate_dir).absolute()
        if os.path.lexists(candidate_dir):
            raise DraftGenerationError("repair candidate path already exists")
        if not isinstance(diagnostic, str) or not diagnostic.strip():
            raise DraftGenerationError("repair diagnostic must be non-empty")
        diagnostic_bytes = diagnostic.encode("utf-8")
        if len(diagnostic_bytes) > 64 * 1024 or "\x00" in diagnostic:
            raise DraftGenerationError("repair diagnostic is invalid or too large")
        if (
            not isinstance(paths, tuple)
            or not paths
            or len(paths) > _MAX_REPAIR_PATHS
            or any(not isinstance(path, str) for path in paths)
            or len(set(paths)) != len(paths)
        ):
            raise DraftGenerationError("repair path set is empty, duplicated, or too large")

        stage_by_path = {}
        for _stage, stage_paths, references in _direct_generation_stages(
            spec.api_name
        ):
            for stage_path in stage_paths:
                stage_by_path[stage_path] = (stage_paths, references)
        normalized_paths = tuple(str(_safe_relative_path(path)) for path in paths)
        if normalized_paths != paths or len(set(normalized_paths)) != len(
            normalized_paths
        ):
            raise DraftGenerationError(
                "repair paths must be unique normalized relative paths"
            )
        for path in normalized_paths:
            if path not in current_files or path not in stage_by_path:
                raise DraftGenerationError(
                    "repair path is not part of the signed stage plan: %s" % path
                )

        if len(repair_reports) >= _MAX_REPAIR_CHAIN_DEPTH:
            raise DraftGenerationError("repair chain has no remaining depth budget")
        reserved_attempts = (
            source.generation["attempt_count"]
            + sum(report["attempt_count"] for report in repair_reports)
            + len(normalized_paths) * self.max_attempts
        )
        if reserved_attempts > _MAX_TRANSPORT_ATTEMPTS:
            raise DraftGenerationError(
                "repair request exceeds the remaining sealable attempt budget"
            )

        chat, generation_base = self._resolve_chat()
        if _generation_identity_snapshot(
            generation_base
        ) != _generation_identity_snapshot(source.generation):
            raise DraftGenerationError(
                "repair LLM identity differs from signed generation"
            )
        contract = _compact_generation_contract(generation_request)
        collected = {}
        receipts = []
        response_hashes = []
        total_attempts = 0

        for repair_index, path in enumerate(normalized_paths, start=1):
            related_paths, references = stage_by_path[path]
            repair_references = tuple(dict.fromkeys(
                (*self._repair_skill_files, *references)
            ))
            related_sections = []
            for related in related_paths:
                related_file = repair_source_dir / _safe_relative_path(related)
                content, payload = _read_bounded_text(
                    related_file,
                    _MAX_TOTAL_BYTES,
                    "repair context file %s" % related,
                )
                if _sha256_bytes(payload) != current_files[related]:
                    raise DraftGenerationError(
                        "repair context file changed after validation: %s" % related
                    )
                related_sections.append(
                    "CURRENT FILE `%s`:\n%s" % (related, content)
                )

            failure = ""
            failures = []
            for repair_attempt in range(1, self.max_attempts + 1):
                total_attempts += 1
                prompt = "\n\n".join((
                    "Repair file %d/%d. Return exactly `%s` and no other path."
                    % (repair_index, len(normalized_paths), path),
                    _stage_invariants(spec, path),
                    "Local CANN compiler/test diagnostic (authoritative):\n"
                    + diagnostic.strip(),
                    contract,
                    "Related current project files (context only; return only the "
                    "requested path):\n\n" + "\n\n".join(related_sections),
                    self._skill_bundle.prompt_context(repair_references),
                ))
                if failure:
                    prompt += (
                        "\n\nThe previous repair response was rejected: "
                        + failure[:500]
                    )
                try:
                    result = chat(_REPAIR_SYSTEM_PROMPT, prompt)
                    if not isinstance(result, tuple) or len(result) != 2:
                        raise DraftGenerationError(
                            "remote repair call returned no auditable receipt"
                        )
                    response, receipt = result
                    _record_generation_attempt(
                        receipts,
                        response_hashes,
                        receipt,
                        response,
                        generation_base,
                    )
                    _summary, files = _validate_stage_bundle(response, (path,))
                    collected[path] = files[0][1]
                    break
                except DraftGenerationError as exc:
                    failure = str(exc)
                    failures.append(
                        "%s attempt %d: %s"
                        % (path, repair_attempt, failure[:500])
                    )
                except Exception as exc:
                    from numpilot.engines.llm import LlmResponseError

                    if isinstance(exc, LlmResponseError):
                        failed_receipt = _validate_generation_attempt_receipt(
                            exc.receipt, None, generation_base
                        )
                        receipts.append(failed_receipt)
                    status_code = getattr(exc, "status_code", None)
                    if isinstance(status_code, int) and not isinstance(
                        status_code, bool
                    ):
                        failure = "remote repair call failed: HTTP status %d" % status_code
                    else:
                        failure = "remote repair call failed: %s" % type(exc).__name__
                    failures.append(
                        "%s attempt %d: %s"
                        % (path, repair_attempt, failure[:500])
                    )
            else:
                raise DraftRepairError(
                    "model repair failed after %d total request(s): %s"
                    % (total_attempts, "; ".join(failures)),
                    receipt_count=len(receipts),
                    request_count=total_attempts,
                )

        before_hashes = {path: current_files[path] for path in normalized_paths}
        after_hashes = {
            path: _sha256_bytes(collected[path].encode("utf-8"))
            for path in normalized_paths
        }
        candidate_files = dict(current_files)
        candidate_files.update(after_hashes)
        report = {
            "schema_version": 2,
            "status": "repaired_unverified",
            "source_artifact_id": source.artifact_id,
            "source_workspace": str(repair_source_dir),
            "signed_source_workspace": source.workspace,
            "parent_repair_report_sha256": parent_repair_sha256,
            "gap_id": spec.gap_id,
            "model": generation_base["model"],
            "provider": generation_base["provider"],
            "protocol": generation_base["protocol"],
            "response_model_aliases": list(
                generation_base["response_model_aliases"]
            ),
            "configuration_source": generation_base["configuration_source"],
            "attempt_count": total_attempts,
            "paths": list(normalized_paths),
            "before_sha256": before_hashes,
            "after_sha256": after_hashes,
            "candidate_files_sha256": candidate_files,
            "diagnostic_sha256": _sha256_bytes(diagnostic_bytes),
            "generation_request_sha256": _sha256_bytes(
                generation_request.encode("utf-8")
            ),
            "skill_bundle_sha256": self._skill_bundle.bundle_sha256,
            "cannbot_repair_skill_files": list(self._repair_skill_files),
            "response_sha256": response_hashes,
            "remote_receipts": receipts,
            "actual_remote_call": True,
            "gates": dict(_EXPECTED_UNVERIFIED_GATES),
        }

        candidate_dir.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temp_dir = Path(tempfile.mkdtemp(
            prefix=".repair-", dir=str(candidate_dir.parent)
        ))
        try:
            shutil.copytree(repair_source_dir, temp_dir, dirs_exist_ok=True)
            for path in normalized_paths:
                target = temp_dir / _safe_relative_path(path)
                payload = collected[path].encode("utf-8")
                temp_file = target.with_name(".%s.repair" % target.name)
                with open(temp_file, "xb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.chmod(temp_file, 0o600)
                os.replace(temp_file, target)
            report_payload = (
                json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n"
            ).encode("utf-8")
            if len(report_payload) > _MAX_METADATA_BYTES:
                raise DraftGenerationError(
                    "repair report exceeds the size limit"
                )
            report_path = temp_dir / "numpilot-repair-report.json"
            report_temp = temp_dir / ".numpilot-repair-report.json.repair"
            with open(report_temp, "xb") as handle:
                handle.write(report_payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(report_temp, 0o600)
            os.replace(report_temp, report_path)
            validated_files = _bounded_tree_hashes(
                temp_dir,
                excluded=(
                    "operator-artifact.json",
                    "numpilot-repair-report.json",
                ),
                maximum_files=_MAX_FILES,
                maximum_entries=_MAX_OPERATOR_TREE_ENTRIES,
                maximum_file_bytes=_MAX_TOTAL_BYTES,
                maximum_total_bytes=_MAX_TOTAL_BYTES,
                label="repair candidate",
            )
            if validated_files != candidate_files:
                raise DraftGenerationError(
                    "repair candidate differs from its report before publication"
                )
            os.replace(temp_dir, candidate_dir)
            temp_dir = None
        finally:
            if temp_dir is not None:
                shutil.rmtree(temp_dir, ignore_errors=True)
        return report

    def generate_asnumpy_adapter_candidate(
        self,
        output_root: Path,
        *,
        operator_artifact: Dict[str, object],
        adapter_contract: str,
        asnumpy_context: Dict[str, str],
        expected_paths: Tuple[str, ...],
        adapter_context_plan: Dict[str, object],
        skill_files: Tuple[str, ...] = ASNUMPY_INTEGRATION_SKILL_FILES,
    ) -> Dict[str, object]:
        """Ask the model for an isolated AsNumpy registry/test overlay."""

        if self._skill_bundle is None:
            raise DraftGenerationError(
                "AsNumpy adapter generation requires a verified skill bundle"
            )
        if (
            not isinstance(operator_artifact, dict)
            or not isinstance(operator_artifact.get("artifact_id"), str)
            or not isinstance(adapter_contract, str)
            or not adapter_contract.strip()
        ):
            raise DraftGenerationError("AsNumpy adapter inputs are invalid")
        if (
            not expected_paths
            or len(expected_paths) > _MAX_ADAPTER_FILES
            or len(set(expected_paths)) != len(expected_paths)
        ):
            raise DraftGenerationError("AsNumpy adapter path set is invalid")
        normalized_paths = tuple(
            str(_safe_asnumpy_adapter_relative_path(path))
            for path in expected_paths
        )
        if normalized_paths != expected_paths:
            raise DraftGenerationError("AsNumpy adapter paths are not normalized")
        normalized_plan = _validate_adapter_context_plan(adapter_context_plan)
        if tuple(normalized_plan["expected_overlay_paths"]) != normalized_paths:
            raise DraftGenerationError(
                "AsNumpy adapter paths differ from the inspected context plan"
            )
        if not isinstance(asnumpy_context, dict) or not asnumpy_context:
            raise DraftGenerationError("AsNumpy reference context is empty")
        context_bytes = 0
        for label, content in asnumpy_context.items():
            if (
                not isinstance(label, str)
                or not label
                or len(label) > _MAX_PATH_CHARS
                or any(ord(character) < 32 for character in label)
                or not isinstance(content, str)
                or not content
                or "\x00" in content
            ):
                raise DraftGenerationError("AsNumpy reference context is invalid")
            context_bytes += len(content.encode("utf-8"))
        if context_bytes > 256 * 1024:
            raise DraftGenerationError("AsNumpy reference context exceeds size limit")
        context_hashes = {
            label: _sha256_bytes(content.encode("utf-8"))
            for label, content in sorted(asnumpy_context.items())
        }
        if any(
            context_hashes.get(label) != digest
            for label, digest in normalized_plan["context_sha256"].items()
        ):
            raise DraftGenerationError(
                "AsNumpy reference content differs from the inspected context plan"
            )
        planned_context_bytes = sum(
            len(asnumpy_context[label].encode("utf-8"))
            for label in normalized_plan["context_sha256"]
        )
        if planned_context_bytes != normalized_plan["total_bytes"]:
            raise DraftGenerationError(
                "AsNumpy reference byte count differs from the inspected context plan"
            )
        official_exact_edit = (
            normalized_plan["architecture"]
            == "official_aclnn_existing_module"
        )
        artifact_declares_official = any((
            operator_artifact.get("artifact_type")
            == "official_binary_adapter_input",
            operator_artifact.get("strategy")
            == "official_aclnn_existing_module",
            operator_artifact.get("origin_kind") == "official_binary",
        ))
        try:
            contract_hint = json.loads(adapter_contract)
        except (json.JSONDecodeError, RecursionError):
            contract_hint = None
        contract_declares_official = isinstance(contract_hint, dict) and any((
            contract_hint.get("contract_type")
            == "official_binary_asnumpy_adapter",
            contract_hint.get("strategy")
            == "official_aclnn_existing_module",
            isinstance(contract_hint.get("source_candidate"), dict)
            and contract_hint["source_candidate"].get("origin_kind")
            == "official_binary",
        ))
        if (
            (artifact_declares_official or contract_declares_official)
            and not official_exact_edit
        ):
            raise DraftGenerationError(
                "official adapter declarations require an official "
                "existing-module context plan"
            )
        if official_exact_edit:
            _validate_official_exact_edit_inputs(
                operator_artifact=operator_artifact,
                adapter_contract=adapter_contract,
                asnumpy_context=asnumpy_context,
                normalized_paths=normalized_paths,
                normalized_plan=normalized_plan,
                skill_files=skill_files,
            )

        root = Path(output_root)
        candidate_dir = root / "candidate"
        if root.is_symlink() or not root.is_dir():
            raise DraftGenerationError("AsNumpy adapter output root is invalid")
        if os.path.lexists(candidate_dir):
            raise DraftGenerationError("AsNumpy adapter candidate already exists")

        chat, generation_base = self._resolve_chat()
        if generation_base["mode"] != "remote_llm":
            raise DraftGenerationError(
                "AsNumpy adapter publication requires a real remote model"
            )
        artifact_json = json.dumps(
            operator_artifact, ensure_ascii=False, indent=2, sort_keys=True
        )
        skill_context = self._skill_bundle.prompt_context(skill_files)
        collected: Dict[str, str] = {}
        collected_provenance: Dict[str, Dict[str, str]] = {}
        collected_bytes = 0
        receipts = []
        response_hashes = []
        failures = []
        total_attempts = 0

        for file_index, path in enumerate(normalized_paths, start=1):
            reference_sections = (
                ""
                if official_exact_edit
                else "\n\n".join(
                    "ASNUMPY REFERENCE `%s`:\n%s" % (label, content)
                    for label, content in sorted(asnumpy_context.items())
                    if label != path
                )
            )
            failure = ""
            for file_attempt in range(1, self.max_attempts + 1):
                prior = (
                    ""
                    if official_exact_edit
                    else "\n\n".join(
                        "ALREADY GENERATED `%s`:\n%s" % item
                        for item in collected.items()
                    )
                )
                if official_exact_edit:
                    base_content = asnumpy_context[path]
                    base_sha256 = _sha256_bytes(
                        base_content.encode("utf-8")
                    )
                    target_window_context, target_window_intervals = (
                        _render_official_exact_edit_target_windows(
                            target_path=path,
                            base_content=base_content,
                        )
                    )
                    dependency_projection = (
                        _render_official_exact_edit_projection(
                            target_path=path,
                            asnumpy_context=asnumpy_context,
                            normalized_plan=normalized_plan,
                            collected=collected,
                            collected_provenance=collected_provenance,
                        )
                    )
                    prompt = "\n\n".join(filter(None, (
                        "Official existing-file edit %d/%d. Propose exact edits "
                        "for `%s` and no other path."
                        % (file_index, len(normalized_paths), path),
                        "Return NUMPILOT_EXACT_EDIT_V1 only. Do not return the "
                        "complete file, NUMPILOT_BUNDLE_V1, a diff, Markdown, "
                        "or commentary.",
                        "TARGET PATH: `%s`\nTARGET BASE SHA-256: %s\n"
                        "Every OLD block must be copied from and remain inside "
                        "one supplied target edit window."
                        % (path, base_sha256),
                        target_window_context,
                        "Expected overlay paths:\n" + "\n".join(
                            "- `%s`" % item for item in normalized_paths
                        ),
                        "SIGNED OPERATOR ARTIFACT MANIFEST:\n" + artifact_json,
                        "DETERMINISTIC HASH-BOUND OFFICIAL ADAPTER CONTRACT:\n"
                        + adapter_contract,
                        dependency_projection,
                        skill_context,
                    )))
                    system_prompt = _ASNUMPY_OFFICIAL_EXACT_EDIT_SYSTEM_PROMPT
                else:
                    prompt = "\n\n".join(filter(None, (
                        "AsNumpy integration file %d/%d. Return exactly `%s` and no "
                        "other path." % (file_index, len(normalized_paths), path),
                        "Your response first line must be exactly "
                        "NUMPILOT_BUNDLE_V1; do not use Markdown fences or commentary.",
                        "Expected overlay paths:\n" + "\n".join(
                            "- `%s`" % item for item in normalized_paths
                        ),
                        "SIGNED OPERATOR ARTIFACT MANIFEST:\n" + artifact_json,
                        "MODEL-GENERATED ADAPTER CONTRACT:\n" + adapter_contract,
                        reference_sections,
                        prior,
                        skill_context,
                    )))
                    system_prompt = _ASNUMPY_ADAPTER_SYSTEM_PROMPT
                if failure:
                    prompt += (
                        "\n\nThe previous response was rejected by local validation: "
                        + failure[:500]
                    )
                if official_exact_edit:
                    _enforce_official_exact_edit_prompt_budget(
                        system_prompt=system_prompt,
                        user_prompt=prompt,
                        model=generation_base["model"],
                        max_tokens=None,
                        thinking_mode=None,
                        reasoning_effort=None,
                        stream=True,
                    )
                total_attempts += 1
                try:
                    if official_exact_edit:
                        result = chat(
                            system_prompt,
                            prompt,
                            stream=True,
                        )
                    else:
                        # Preserve the historical two-argument callable
                        # contract for injected test doubles and every
                        # complete-file/custom adapter route.
                        result = chat(system_prompt, prompt)
                    if not isinstance(result, tuple) or len(result) != 2:
                        raise DraftGenerationError(
                            "remote adapter call returned no auditable receipt"
                        )
                    response, receipt = result
                    accepted_receipt = _record_generation_attempt(
                        receipts,
                        response_hashes,
                        receipt,
                        response,
                        generation_base,
                    )
                    if official_exact_edit:
                        content = _apply_official_exact_edit_response(
                            response,
                            expected_path=path,
                            base_content=base_content,
                            allowed_intervals=target_window_intervals,
                        )
                    else:
                        _summary, files = _validate_stage_bundle(
                            response,
                            (path,),
                            path_validator=_safe_asnumpy_adapter_relative_path,
                        )
                        content = files[0][1]
                    content_size = len(content.encode("utf-8"))
                    if content_size > _MAX_ADAPTER_FILE_BYTES:
                        raise DraftGenerationError(
                            "generated AsNumpy adapter file exceeds the size limit"
                        )
                    if collected_bytes + content_size > _MAX_ADAPTER_TOTAL_BYTES:
                        raise DraftGenerationError(
                            "generated AsNumpy adapter exceeds the total-size limit"
                        )
                    collected[path] = content
                    if official_exact_edit:
                        collected_provenance[path] = {
                            "base_sha256": base_sha256,
                            "rebuilt_sha256": _sha256_bytes(
                                content.encode("utf-8")
                            ),
                            "response_sha256": _sha256_bytes(
                                response.encode("utf-8")
                            ),
                            "request_body_sha256": accepted_receipt[
                                "request_body_sha256"
                            ],
                            "response_body_sha256": accepted_receipt[
                                "response_body_sha256"
                            ],
                            "receipt_content_sha256": accepted_receipt[
                                "content_sha256"
                            ],
                        }
                    collected_bytes += content_size
                    break
                except DraftGenerationError as exc:
                    failure = str(exc)
                    failures.append(
                        "%s attempt %d: %s"
                        % (path, file_attempt, failure[:500])
                    )
                except Exception as exc:
                    from numpilot.engines.llm import LlmResponseError

                    if isinstance(exc, LlmResponseError):
                        failed_receipt = _validate_generation_attempt_receipt(
                            exc.receipt, None, generation_base
                        )
                        receipts.append(failed_receipt)
                    status_code = getattr(exc, "status_code", None)
                    failure = (
                        "remote adapter call failed: HTTP status %d" % status_code
                        if isinstance(status_code, int)
                        and not isinstance(status_code, bool)
                        else "remote adapter call failed: %s" % type(exc).__name__
                    )
                    failures.append(
                        "%s attempt %d: %s"
                        % (path, file_attempt, failure[:500])
                    )
            else:
                raise DraftRepairError(
                    "model AsNumpy adapter generation failed after %d request(s): %s"
                    % (total_attempts, "; ".join(failures)),
                    receipt_count=len(receipts),
                    request_count=total_attempts,
                )

        file_hashes = {
            path: _sha256_bytes(content.encode("utf-8"))
            for path, content in collected.items()
        }
        report = {
            "schema_version": 4,
            "status": "generated_unverified",
            "source_artifact_id": operator_artifact["artifact_id"],
            "model": generation_base["model"],
            "provider": generation_base["provider"],
            "protocol": generation_base["protocol"],
            "response_model_aliases": list(
                generation_base["response_model_aliases"]
            ),
            "configuration_source": generation_base["configuration_source"],
            "attempt_count": total_attempts,
            "files_sha256": file_hashes,
            "adapter_contract_sha256": _sha256_bytes(
                adapter_contract.encode("utf-8")
            ),
            "asnumpy_context_sha256": context_hashes,
            "asnumpy_adapter_context_plan": normalized_plan,
            "skill_bundle_sha256": self._skill_bundle.bundle_sha256,
            "integration_skill_files": list(skill_files),
            "response_sha256": response_hashes,
            "remote_receipts": receipts,
            "actual_remote_call": True,
            "gates": dict(_EXPECTED_UNVERIFIED_GATES),
        }
        report["adapter_artifact_id"] = _adapter_artifact_id(report)
        report = _validate_adapter_report(
            report,
            candidate_hashes=file_hashes,
        )
        _validate_adapter_skill_binding(report, self._skill_bundle)
        report_payload = _bounded_json_payload(
            report, "AsNumpy adapter report"
        )

        temp_dir = Path(tempfile.mkdtemp(prefix=".adapter-", dir=str(root)))
        report_temp = root / ".asnumpy-adapter-report.json.tmp"
        try:
            for relative, content in collected.items():
                target = temp_dir / _safe_asnumpy_adapter_relative_path(relative)
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                target.write_text(content, encoding="utf-8")
                os.chmod(target, 0o600)
            validated_files = _bounded_tree_hashes(
                temp_dir,
                excluded=(),
                maximum_files=_MAX_ADAPTER_FILES,
                maximum_entries=_MAX_ADAPTER_TREE_ENTRIES,
                maximum_file_bytes=_MAX_ADAPTER_FILE_BYTES,
                maximum_total_bytes=_MAX_ADAPTER_TOTAL_BYTES,
                label="generated AsNumpy adapter candidate",
                path_validator=_safe_asnumpy_adapter_relative_path,
            )
            if validated_files != file_hashes:
                raise DraftGenerationError(
                    "generated AsNumpy adapter differs before publication"
                )
            with open(report_temp, "xb") as handle:
                handle.write(report_payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(report_temp, 0o600)
            os.replace(temp_dir, candidate_dir)
            temp_dir = None
            report_path = root / "asnumpy-adapter-report.json"
            os.replace(report_temp, report_path)
        finally:
            if temp_dir is not None:
                shutil.rmtree(temp_dir, ignore_errors=True)
            if os.path.lexists(report_temp):
                os.unlink(report_temp)
        return report

    def repair_asnumpy_adapter_candidate(
        self,
        output_root: Path,
        *,
        base_adapter_root: Path,
        diagnostic: str,
        path: str,
        asnumpy_context: Dict[str, str],
        adapter_context_plan: Dict[str, object],
        skill_files: Tuple[str, ...] = ASNUMPY_INTEGRATION_SKILL_FILES,
        repair_profile: str = "compile",
    ) -> Dict[str, object]:
        """Let the remote model repair one hash-bound AsNumpy overlay file."""

        if self._skill_bundle is None:
            raise DraftGenerationError(
                "AsNumpy adapter repair requires a verified skill bundle"
            )
        if (
            not isinstance(repair_profile, str)
            or repair_profile not in {"compile", "precision", "runtime"}
        ):
            raise DraftGenerationError(
                "AsNumpy adapter repair profile must be compile, precision, or runtime"
            )
        if not isinstance(diagnostic, str) or not diagnostic.strip():
            raise DraftGenerationError("AsNumpy adapter diagnostic must be non-empty")
        diagnostic_bytes = diagnostic.encode("utf-8")
        if len(diagnostic_bytes) > 64 * 1024 or "\x00" in diagnostic:
            raise DraftGenerationError(
                "AsNumpy adapter diagnostic is invalid or too large"
            )
        normalized_path = str(_safe_asnumpy_adapter_relative_path(path))
        if normalized_path != path:
            raise DraftGenerationError("AsNumpy adapter repair path is not normalized")
        if not isinstance(asnumpy_context, dict) or not asnumpy_context:
            raise DraftGenerationError("AsNumpy repair context is empty")
        context_bytes = 0
        for label, content in asnumpy_context.items():
            if (
                not isinstance(label, str)
                or not label
                or len(label) > _MAX_PATH_CHARS
                or any(ord(character) < 32 for character in label)
                or not isinstance(content, str)
                or not content
                or "\x00" in content
            ):
                raise DraftGenerationError("AsNumpy repair context is invalid")
            context_bytes += len(content.encode("utf-8"))
        if context_bytes > 256 * 1024:
            raise DraftGenerationError("AsNumpy repair context exceeds size limit")
        normalized_plan = _validate_adapter_context_plan(adapter_context_plan)
        context_hashes = {
            label: _sha256_bytes(content.encode("utf-8"))
            for label, content in sorted(asnumpy_context.items())
        }
        if set(context_hashes) != set(normalized_plan["context_sha256"]) or any(
            context_hashes.get(label) != digest
            for label, digest in normalized_plan["context_sha256"].items()
        ):
            raise DraftGenerationError(
                "AsNumpy repair context differs from the inspected context plan"
            )
        if context_bytes != normalized_plan["total_bytes"]:
            raise DraftGenerationError(
                "AsNumpy repair byte count differs from the inspected context plan"
            )

        base_root = Path(base_adapter_root).absolute()
        base_candidate = base_root / "candidate"
        base_report, report_payload, normalized_hashes = (
            _load_asnumpy_adapter_report(base_root)
        )
        if base_report.get("schema_version") != 4:
            raise DraftGenerationError(
                "AsNumpy adapter base predates context-plan binding; regenerate it"
            )
        if base_report.get("asnumpy_adapter_context_plan") != normalized_plan:
            raise DraftGenerationError(
                "AsNumpy adapter reference layout or content changed; regenerate it"
            )
        if set(normalized_hashes) != set(normalized_plan["expected_overlay_paths"]):
            raise DraftGenerationError(
                "AsNumpy adapter base paths differ from the inspected context plan"
            )
        _validate_adapter_skill_binding(base_report, self._skill_bundle)
        if normalized_path not in normalized_hashes:
            raise DraftGenerationError(
                "repair path is not part of the model-generated AsNumpy overlay"
            )
        current_contents = {}
        current_total_bytes = 0
        for relative, expected_hash in normalized_hashes.items():
            content, content_bytes, actual_hash = _read_bounded_adapter_text(
                base_candidate / _safe_asnumpy_adapter_relative_path(relative),
                relative,
            )
            current_total_bytes += content_bytes
            if current_total_bytes > _MAX_ADAPTER_TOTAL_BYTES:
                raise DraftGenerationError(
                    "AsNumpy adapter base exceeds the total-size limit"
                )
            if actual_hash != expected_hash:
                raise DraftGenerationError(
                    "AsNumpy adapter base changed after validation: %s" % relative
                )
            current_contents[relative] = content

        root = Path(output_root)
        candidate_dir = root / "candidate"
        if root.is_symlink() or not root.is_dir() or os.path.lexists(candidate_dir):
            raise DraftGenerationError("AsNumpy adapter repair output is invalid")
        base_context_hashes = base_report.get("asnumpy_context_sha256")
        if not isinstance(base_context_hashes, dict):
            raise DraftGenerationError(
                "AsNumpy adapter base context manifest is invalid"
            )
        chat, generation_base = self._resolve_chat()
        if generation_base["mode"] != "remote_llm":
            raise DraftGenerationError(
                "AsNumpy adapter repair requires a real remote model"
            )
        if _generation_identity_snapshot(
            base_report
        ) != _generation_identity_snapshot(generation_base):
            raise DraftGenerationError(
                "AsNumpy adapter repair model identity differs from its base"
            )

        context_sections = "\n\n".join(
            "ASNUMPY BUILD REFERENCE `%s`:\n%s" % (label, content)
            for label, content in sorted(asnumpy_context.items())
        )
        current_sections = "\n\n".join(
            "CURRENT MODEL OVERLAY `%s`:\n%s" % (label, content)
            for label, content in sorted(current_contents.items())
        )
        skill_context = self._skill_bundle.prompt_context(skill_files)
        official_repair = (
            normalized_plan["architecture"]
            == "official_aclnn_existing_module"
        )
        receipts = []
        response_hashes = []
        failures = []
        replacement = None
        total_attempts = 0
        failure = ""
        for repair_attempt in range(1, self.max_attempts + 1):
            if official_repair:
                base_content = current_contents[normalized_path]
                base_sha256 = normalized_hashes[normalized_path]
                target_window_context, target_window_intervals = (
                    _render_official_exact_edit_target_windows(
                        target_path=normalized_path,
                        base_content=base_content,
                    )
                )
                dependency_projection = (
                    _render_official_exact_edit_repair_dependencies(
                        target_path=normalized_path,
                        asnumpy_context=asnumpy_context,
                        base_context_hashes=base_context_hashes,
                        normalized_plan=normalized_plan,
                        current_contents=current_contents,
                        current_hashes=normalized_hashes,
                    )
                )
                prompt = "\n\n".join(filter(None, (
                    "Official existing-file repair 1/1. Propose exact edits "
                    "for `%s` and no other path." % normalized_path,
                    "Return NUMPILOT_EXACT_EDIT_V1 only. Do not return the "
                    "complete file, NUMPILOT_BUNDLE_V1, a diff, Markdown, or "
                    "commentary.",
                    "TARGET PATH: `%s`\nTARGET BASE SHA-256: %s\n"
                    "Every OLD block must be copied from and remain inside "
                    "one supplied target edit window."
                    % (normalized_path, base_sha256),
                    "Repair qualification profile (audit binding): `%s`."
                    % repair_profile,
                    "Raw local AsNumpy build/test diagnostic "
                    "(authoritative):\n" + diagnostic.strip(),
                    target_window_context,
                    dependency_projection,
                    skill_context,
                )))
                system_prompt = (
                    _ASNUMPY_OFFICIAL_EXACT_EDIT_REPAIR_SYSTEM_PROMPT
                )
            else:
                prompt = "\n\n".join((
                    "Repair exactly `%s` and no other path." % normalized_path,
                    "Repair qualification profile (audit binding): `%s`."
                    % repair_profile,
                    "Your response first line must be exactly NUMPILOT_BUNDLE_V1 "
                    "and your last line must be exactly END_BUNDLE. Do not use "
                    "Markdown fences or commentary.",
                    "Raw local AsNumpy build/test diagnostic (authoritative):\n"
                    + diagnostic.strip(),
                    current_sections,
                    context_sections,
                    skill_context,
                ))
                system_prompt = _ASNUMPY_ADAPTER_REPAIR_SYSTEM_PROMPT
            if failure:
                prompt += (
                    "\n\nThe previous repair response was rejected by local "
                    "validation: " + failure[:500]
                )
            if official_repair:
                _enforce_official_exact_edit_prompt_budget(
                    system_prompt=system_prompt,
                    user_prompt=prompt,
                    model=generation_base["model"],
                    max_tokens=None,
                    thinking_mode=None,
                    reasoning_effort=None,
                    stream=True,
                )
            total_attempts += 1
            try:
                if official_repair:
                    result = chat(
                        system_prompt,
                        prompt,
                        stream=True,
                    )
                else:
                    result = chat(system_prompt, prompt)
                if not isinstance(result, tuple) or len(result) != 2:
                    raise DraftGenerationError(
                        "remote AsNumpy repair call returned no auditable receipt"
                    )
                response, receipt = result
                _record_generation_attempt(
                    receipts,
                    response_hashes,
                    receipt,
                    response,
                    generation_base,
                )
                if official_repair:
                    proposed = _apply_official_exact_edit_response(
                        response,
                        expected_path=normalized_path,
                        base_content=base_content,
                        allowed_intervals=target_window_intervals,
                    )
                else:
                    _summary, files = _validate_stage_bundle(
                        response,
                        (normalized_path,),
                        path_validator=_safe_asnumpy_adapter_relative_path,
                    )
                    proposed = files[0][1]
                proposed_size = len(proposed.encode("utf-8"))
                replaced_size = len(
                    current_contents[normalized_path].encode("utf-8")
                )
                if proposed_size > _MAX_ADAPTER_FILE_BYTES:
                    raise DraftGenerationError(
                        "repaired AsNumpy adapter file exceeds the size limit"
                    )
                if (
                    current_total_bytes - replaced_size + proposed_size
                    > _MAX_ADAPTER_TOTAL_BYTES
                ):
                    raise DraftGenerationError(
                        "repaired AsNumpy adapter exceeds the total-size limit"
                    )
                replacement = proposed
                break
            except DraftGenerationError as exc:
                failure = str(exc)
                failures.append(
                    "%s attempt %d: %s"
                    % (normalized_path, repair_attempt, failure[:500])
                )
            except Exception as exc:
                from numpilot.engines.llm import (
                    LlmResponseError,
                    LlmTransportError,
                )

                if isinstance(exc, LlmResponseError):
                    failed_receipt = _validate_generation_attempt_receipt(
                        exc.receipt, None, generation_base
                    )
                    receipts.append(failed_receipt)
                status_code = getattr(exc, "status_code", None)
                if isinstance(exc, LlmTransportError):
                    failure = "remote AsNumpy repair call failed: %s" % exc
                else:
                    failure = (
                        "remote AsNumpy repair call failed: HTTP status %d" % status_code
                        if isinstance(status_code, int) and not isinstance(status_code, bool)
                        else "remote AsNumpy repair call failed: %s" % type(exc).__name__
                    )
                failures.append(
                    "%s attempt %d: %s"
                    % (normalized_path, repair_attempt, failure[:500])
                )
        if replacement is None:
            raise DraftRepairError(
                "model AsNumpy adapter repair failed after %d request(s): %s"
                % (total_attempts, "; ".join(failures)),
                receipt_count=len(receipts),
                request_count=total_attempts,
            )

        before_hash = normalized_hashes[normalized_path]
        after_hash = _sha256_bytes(replacement.encode("utf-8"))
        updated_hashes = dict(normalized_hashes)
        updated_hashes[normalized_path] = after_hash
        report = {
            "schema_version": 4,
            "status": "repaired_unverified",
            "source_artifact_id": base_report["source_artifact_id"],
            "source_adapter_root": str(base_root),
            "parent_adapter_artifact_id": base_report["adapter_artifact_id"],
            "parent_adapter_report_sha256": _sha256_bytes(report_payload),
            "model": generation_base["model"],
            "provider": generation_base["provider"],
            "protocol": generation_base["protocol"],
            "response_model_aliases": list(
                generation_base["response_model_aliases"]
            ),
            "configuration_source": generation_base["configuration_source"],
            "repair_profile": repair_profile,
            "attempt_count": total_attempts,
            "paths": [normalized_path],
            "before_sha256": {normalized_path: before_hash},
            "after_sha256": {normalized_path: after_hash},
            "files_sha256": updated_hashes,
            "diagnostic_sha256": _sha256_bytes(diagnostic_bytes),
            "asnumpy_context_sha256": context_hashes,
            "asnumpy_adapter_context_plan": normalized_plan,
            "skill_bundle_sha256": self._skill_bundle.bundle_sha256,
            "integration_skill_files": list(skill_files),
            "response_sha256": response_hashes,
            "remote_receipts": receipts,
            "actual_remote_call": True,
            "gates": dict(_EXPECTED_UNVERIFIED_GATES),
        }
        report["adapter_artifact_id"] = _adapter_artifact_id(report)
        report = _validate_adapter_report(
            report,
            candidate_hashes=updated_hashes,
            parent_report=base_report,
            parent_report_sha256=_sha256_bytes(report_payload),
        )
        _validate_adapter_skill_binding(report, self._skill_bundle)
        serialized_report = _bounded_json_payload(
            report, "AsNumpy adapter repair report"
        )
        temp_dir = Path(tempfile.mkdtemp(prefix=".adapter-repair-", dir=str(root)))
        report_temp = root / ".asnumpy-adapter-report.json.tmp"
        try:
            shutil.copytree(base_candidate, temp_dir, dirs_exist_ok=True)
            target = temp_dir / _safe_asnumpy_adapter_relative_path(
                normalized_path
            )
            temporary = target.with_name(".%s.repair" % target.name)
            with open(temporary, "xb") as handle:
                handle.write(replacement.encode("utf-8"))
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, 0o600)
            os.replace(temporary, target)
            validated_files = _bounded_tree_hashes(
                temp_dir,
                excluded=(),
                maximum_files=_MAX_ADAPTER_FILES,
                maximum_entries=_MAX_ADAPTER_TREE_ENTRIES,
                maximum_file_bytes=_MAX_ADAPTER_FILE_BYTES,
                maximum_total_bytes=_MAX_ADAPTER_TOTAL_BYTES,
                label="repaired AsNumpy adapter candidate",
                path_validator=_safe_asnumpy_adapter_relative_path,
            )
            if validated_files != updated_hashes:
                raise DraftGenerationError(
                    "repaired AsNumpy adapter differs before publication"
                )
            with open(report_temp, "xb") as handle:
                handle.write(serialized_report)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(report_temp, 0o600)
            os.replace(temp_dir, candidate_dir)
            temp_dir = None
            report_path = root / "asnumpy-adapter-report.json"
            os.replace(report_temp, report_path)
        finally:
            if temp_dir is not None:
                shutil.rmtree(temp_dir, ignore_errors=True)
            if os.path.lexists(report_temp):
                os.unlink(report_temp)
        return report

    def seal_repair_candidate(
        self,
        spec: GapSpec,
        draft_dir: Path,
        candidate_dir: Path,
        artifact_dir: Path,
    ) -> OperatorDraft:
        """Publish a verified repair chain as a new immutable unverified artifact."""

        draft_dir = Path(draft_dir)
        candidate_dir = Path(candidate_dir)
        artifact_dir = Path(artifact_dir)
        if self._skill_bundle is None:
            raise DraftGenerationError(
                "sealing a repair requires the verified Skill bundle"
            )
        source = _load_existing_draft(draft_dir, spec)
        if os.path.lexists(artifact_dir):
            raise DraftGenerationError("sealed artifact path already exists")
        candidate_files, reports = _load_repair_chain(
            candidate_dir,
            spec,
            source,
            self._skill_bundle,
        )
        if not reports:
            raise DraftGenerationError("repair chain is empty")

        generation = dict(source.generation)
        response_hashes = list(generation["response_sha256"])
        receipts = list(generation["remote_receipts"])
        attempt_count = generation["attempt_count"]
        for report in reports:
            if _generation_identity_snapshot(
                report
            ) != _generation_identity_snapshot(generation):
                raise DraftGenerationError(
                    "repair chain LLM identity differs from signed generation"
                )
            report_attempts = report.get("attempt_count")
            report_hashes = report.get("response_sha256")
            report_receipts = report.get("remote_receipts")
            if (
                isinstance(report_attempts, bool)
                or not isinstance(report_attempts, int)
                or report_attempts < 1
                or not isinstance(report_hashes, list)
                or not report_hashes
                or not isinstance(report_receipts, list)
                or not report_receipts
            ):
                raise DraftGenerationError("repair chain transport evidence is invalid")
            receipt_validator = (
                validate_configured_model_attempt_receipt
                if report.get("schema_version") == 2
                else validate_configured_model_receipt
            )
            normalized_receipts = [
                receipt_validator(
                    receipt,
                    expected_model=generation["model"],
                    expected_provider=generation["provider"],
                    expected_protocol=generation.get(
                        "protocol", OPENAI_CHAT_COMPLETIONS_PROTOCOL
                    ),
                    response_model_aliases=generation.get(
                        "response_model_aliases", ()
                    ),
                )
                for receipt in report_receipts
            ]
            if report.get("schema_version") == 2:
                validate_configured_model_receipt(
                    normalized_receipts[-1],
                    expected_model=generation["model"],
                    expected_provider=generation["provider"],
                    expected_protocol=generation.get(
                        "protocol", OPENAI_CHAT_COMPLETIONS_PROTOCOL
                    ),
                    response_model_aliases=generation.get(
                        "response_model_aliases", ()
                    ),
                )
                if (
                    normalized_receipts[-1]["content_sha256"]
                    != report_hashes[-1]
                ):
                    raise DraftGenerationError(
                        "repair chain final receipt is not the accepted response"
                    )
            content_hashes = [
                receipt["content_sha256"]
                for receipt in normalized_receipts
                if receipt["content_sha256"] is not None
            ]
            if (
                any(not _is_sha256(digest) for digest in report_hashes)
                or content_hashes != report_hashes
            ):
                raise DraftGenerationError(
                    "repair chain receipt/content hashes are inconsistent"
                )
            attempt_count += report_attempts
            response_hashes.extend(report_hashes)
            receipts.extend(normalized_receipts)
        if attempt_count > _MAX_TRANSPORT_ATTEMPTS:
            raise DraftGenerationError("sealed repair chain exceeds attempt limit")
        generation.update({
            "attempt_count": attempt_count,
            "transport_attempt_count": attempt_count,
            "successful_http_response_count": len(receipts),
            "content_response_count": len(response_hashes),
            "response_sha256": response_hashes,
            "remote_receipts": receipts,
            "local_validation": "passed",
            "fallback": "disabled",
        })
        generation = _validate_generation(generation)

        provenance_relative = "docs/numpilot-repair-provenance.json"
        if any(
            provenance_relative in report.get("paths", ())
            for report in reports
        ):
            raise DraftGenerationError(
                "repair chain cannot modify reserved provenance metadata"
            )

        provenance = {
            "schema_version": 1,
            "source_artifact_id": source.artifact_id,
            "source_workspace": source.workspace,
            "final_candidate_workspace": str(candidate_dir),
            "final_candidate_files_sha256": candidate_files,
            "repair_reports": list(reports),
        }
        provenance_text = (
            json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        )
        if len(provenance_text.encode("utf-8")) > _MAX_TOTAL_BYTES:
            raise DraftGenerationError("sealed repair provenance exceeds size limit")

        files = []
        sealed_source_bytes = 0
        for relative in sorted(candidate_files):
            # A repair may start from an artifact that was sealed previously.
            # Its candidate inherits the old provenance file, which this seal
            # replaces with provenance for the new repair chain.
            if relative == provenance_relative:
                continue
            path = candidate_dir / _safe_relative_path(relative)
            content, payload = _read_bounded_text(
                path,
                _MAX_TOTAL_BYTES,
                "repaired artifact file %s" % relative,
            )
            if _sha256_bytes(payload) != candidate_files[relative]:
                raise DraftGenerationError(
                    "repaired artifact file changed after validation: %s"
                    % relative
                )
            sealed_source_bytes += len(payload)
            if sealed_source_bytes > _MAX_TOTAL_BYTES:
                raise DraftGenerationError(
                    "sealed repair artifact exceeds size limits"
                )
            files.append((_safe_relative_path(relative), content))
        provenance_path = _safe_relative_path(provenance_relative)
        files.append((provenance_path, provenance_text))
        total_bytes = sum(
            len(content.encode("utf-8")) for _path, content in files
        )
        if total_bytes > _MAX_TOTAL_BYTES or len(files) > _MAX_FILES:
            raise DraftGenerationError("sealed repair artifact exceeds size limits")
        project_files = {str(path): content for path, content in files}
        _require_project_layout(project_files, file_contents=project_files)

        draft = self._publish_draft(
            spec,
            artifact_dir.parent,
            artifact_dir,
            "Sealed compiler-directed repair chain for %s" % spec.qualified_api,
            tuple(files),
            generation,
            maximum_attempts=_MAX_TRANSPORT_ATTEMPTS,
        )
        self._run_actions[draft.workspace] = "sealed_repair_chain"
        return draft

    def _publish_draft(
        self,
        spec: GapSpec,
        workspace: Path,
        draft_dir: Path,
        summary: str,
        files,
        generation,
        *,
        maximum_attempts: int = _MAX_FRESH_TRANSPORT_ATTEMPTS,
    ) -> OperatorDraft:
        # Publication is the trust boundary: use the same strict evidence
        # validator used when reloading an existing artifact.
        generation = _validate_generation(
            generation,
            maximum_attempts=maximum_attempts,
        )
        workspace.mkdir(parents=True, exist_ok=True, mode=0o700)
        temp_dir = Path(tempfile.mkdtemp(prefix=".draft-", dir=str(workspace)))
        try:
            hashes: Dict[str, str] = {}
            total_bytes = 0
            for relative, content in files:
                if len(hashes) >= _MAX_FILES:
                    raise DraftGenerationError(
                        "draft publication exceeds the file-count limit"
                    )
                if not isinstance(relative, PurePosixPath) or not isinstance(
                    content, str
                ):
                    raise DraftGenerationError(
                        "draft publication contains an invalid file"
                    )
                if str(_safe_relative_path(str(relative))) != str(relative):
                    raise DraftGenerationError(
                        "draft publication path is not normalized"
                    )
                target = temp_dir.joinpath(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                payload = content.encode("utf-8")
                if len(payload) > _MAX_TOTAL_BYTES:
                    raise DraftGenerationError(
                        "draft publication file exceeds the size limit"
                    )
                total_bytes += len(payload)
                if total_bytes > _MAX_TOTAL_BYTES:
                    raise DraftGenerationError(
                        "draft publication exceeds the total-size limit"
                    )
                if str(relative) in hashes:
                    raise DraftGenerationError(
                        "draft publication contains a duplicate path"
                    )
                with open(target, "xb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.chmod(target, 0o700 if relative.name == "build.sh" else 0o600)
                hashes[str(relative)] = _sha256_bytes(payload)

            draft = OperatorDraft(
                artifact_id=_draft_artifact_id(
                    spec.gap_id,
                    hashes,
                    generation,
                ),
                gap_id=spec.gap_id,
                workspace=str(draft_dir),
                summary=summary,
                files=hashes,
                generation=generation,
            )
            manifest = json.dumps(
                draft.to_dict(), ensure_ascii=False, indent=2, sort_keys=True
            ) + "\n"
            if len(manifest.encode("utf-8")) > _MAX_METADATA_BYTES:
                raise DraftGenerationError(
                    "draft artifact manifest exceeds the size limit"
                )
            manifest_path = temp_dir / "operator-artifact.json"
            with open(manifest_path, "x", encoding="utf-8") as handle:
                handle.write(manifest)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(manifest_path, 0o600)
            os.replace(temp_dir, draft_dir)
        except Exception:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise
        action = (
            "fresh_remote"
            if generation["mode"] == "remote_llm"
            else "injected_test_double"
        )
        self._run_actions[draft.workspace] = action
        return draft

    def generate(
        self,
        spec: GapSpec,
        workspace: Path,
        *,
        generation_request: str = "",
    ) -> OperatorDraft:
        if spec.cann_version.strip().lower() in {"", "unknown", "unset", "none"}:
            raise DraftGenerationError(
                "a concrete CANN version is required before LLM draft generation"
            )
        draft_dir = workspace / "draft"
        if os.path.lexists(draft_dir):
            if draft_dir.is_dir() and not draft_dir.is_symlink():
                draft = _load_existing_draft(draft_dir, spec)
                if (
                    self._chat is None
                    and draft.generation.get("mode") != "remote_llm"
                ):
                    raise DraftGenerationError(
                        "production reuse requires a receipt-bound configured-model draft"
                    )
                self._run_actions[draft.workspace] = "reused_existing"
                return draft
            raise DraftGenerationError("draft target exists and is not a directory")

        request = (
            "Generate the operator project for this GapSpec. "
            "The JSON contract is the only semantic source of truth:\n"
            + json.dumps(spec.to_dict(), ensure_ascii=False, sort_keys=True)
        )
        if generation_request:
            request += (
                "\n\nGeneration workflow contract (already extracted from the "
                "local cannbot skill and AsNumpy reference; follow it directly):\n"
                + generation_request
            )
        chat, generation_base = self._resolve_chat()
        if (
            self._staged_direct
            and generation_base["mode"] == "remote_llm"
            and _should_use_staged_direct(spec)
        ):
            (
                summary,
                files,
                response_hashes,
                remote_receipts,
                attempt_count,
            ) = self._generate_staged_direct(
                chat,
                generation_base,
                spec,
                generation_request,
            )
            generation = dict(generation_base)
            generation.update({
                "attempt_count": attempt_count,
                "transport_attempt_count": attempt_count,
                "successful_http_response_count": len(remote_receipts),
                "content_response_count": len(response_hashes),
                "response_sha256": response_hashes,
                "remote_receipts": remote_receipts,
                "local_validation": "passed",
                "fallback": "disabled",
            })
            try:
                return self._publish_draft(
                    spec,
                    workspace,
                    draft_dir,
                    summary,
                    files,
                    generation,
                )
            except DraftGenerationError as exc:
                # A model may have been contacted successfully while local
                # publication failed.  Preserve that fact in-memory for a
                # sanitized continuation report; never persist the exception
                # message or response body.
                exc.request_count = attempt_count
                exc.receipt_count = len(remote_receipts)
                exc.actual_remote_call = generation["mode"] == "remote_llm"
                raise
        failure = ""
        summary = ""
        files: Tuple[Tuple[PurePosixPath, str], ...] = ()
        response_hashes = []
        remote_receipts = []
        attempt_failures = []
        attempt_count = 0
        for attempt_count in range(1, self.max_attempts + 1):
            prompt = request
            if failure:
                prompt += (
                    "\nPrevious response was rejected by the local validator: "
                    + failure[:500]
                    + "\nReturn a corrected JSON object."
                )
            try:
                result = chat(_SYSTEM_PROMPT, prompt)
                if generation_base["mode"] == "remote_llm":
                    if (
                        not isinstance(result, tuple)
                        or len(result) != 2
                    ):
                        raise DraftGenerationError(
                            "remote model call returned no auditable receipt"
                        )
                    response, receipt = result
                    _record_generation_attempt(
                        remote_receipts,
                        response_hashes,
                        receipt,
                        response,
                        generation_base,
                    )
                else:
                    response = result
                if (
                    generation_base["mode"] != "remote_llm"
                    and isinstance(response, str)
                ):
                    response_hashes.append(
                        _sha256_bytes(response.encode("utf-8"))
                    )
                summary, files = _validate_bundle(response)
                break
            except DraftGenerationError as exc:
                failure = str(exc)
                attempt_failures.append(
                    "attempt %d: %s" % (attempt_count, failure[:500])
                )
            except Exception as exc:
                from numpilot.engines.llm import LlmResponseError

                if (
                    generation_base["mode"] == "remote_llm"
                    and isinstance(exc, LlmResponseError)
                ):
                    failed_receipt = _validate_generation_attempt_receipt(
                        exc.receipt, None, generation_base
                    )
                    remote_receipts.append(failed_receipt)
                status_code = getattr(exc, "status_code", None)
                if isinstance(status_code, int) and not isinstance(status_code, bool):
                    failure = "remote model call failed: HTTP status %d" % status_code
                else:
                    failure = "remote model call failed: %s" % type(exc).__name__
                attempt_failures.append(
                    "attempt %d: %s" % (attempt_count, failure[:500])
                )
        else:
            error = DraftGenerationError(
                "model failed to produce a safe draft after %d attempt(s): %s"
                % (self.max_attempts, "; ".join(attempt_failures))
            )
            error.request_count = attempt_count
            error.receipt_count = len(remote_receipts)
            error.actual_remote_call = generation_base["mode"] == "remote_llm"
            raise error
        generation = dict(generation_base)
        generation.update({
            "attempt_count": attempt_count,
            "transport_attempt_count": attempt_count,
            "successful_http_response_count": len(remote_receipts),
            "content_response_count": len(response_hashes),
            "response_sha256": response_hashes,
            "remote_receipts": remote_receipts,
            "local_validation": "passed",
            "fallback": "disabled",
        })
        try:
            return self._publish_draft(
                spec,
                workspace,
                draft_dir,
                summary,
                files,
                generation,
            )
        except DraftGenerationError as exc:
            exc.request_count = attempt_count
            exc.receipt_count = len(remote_receipts)
            exc.actual_remote_call = generation["mode"] == "remote_llm"
            raise
