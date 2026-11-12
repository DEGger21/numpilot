"""Plan a bounded, read-only AsNumpy adapter-generation context.

The pinned AsNumpy source used by NumPilot predates the manifest generator,
while newer checkouts can expose ``scripts/gen_custom_ops.py`` plus a registry.
This module detects those two layouts without importing AsNumpy or modifying
the source tree.  Every prompt context file is read from the target checkout,
kept inside its real root, rejected when symlinked, and bound to a SHA-256.

The returned overlay paths are only an allow-list for the configured model.
They do not authorize applying the overlay or activating an operator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from types import MappingProxyType
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union


MANIFEST_ARCHITECTURE = "manifest_generator"
LEGACY_ARCHITECTURE = "legacy_manual"
OFFICIAL_ACLNN_ARCHITECTURE = "official_aclnn_existing_module"
MANIFEST_MODE = "manifest"
MANUAL_MODE = "manual"
EXISTING_MODULE_MODE = "existing_module"

_SAFE_API_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ADAPTER_API_PATHS = (
    re.compile(r"^csrc/custom_ops/registry/([a-z][a-z0-9_]{0,63})/op\.yaml$"),
    re.compile(r"^include/asnumpy/custom_ops/([a-z][a-z0-9_]{0,63})\.hpp$"),
    re.compile(r"^csrc/custom_ops/([a-z][a-z0-9_]{0,63})\.cpp$"),
    re.compile(
        r"^tests/asnumpy_tests/custom_ops_tests/"
        r"test_([a-z][a-z0-9_]{0,63})\.py$"
    ),
)
_MAX_CONTEXT_FILES = 8
_MAX_OFFICIAL_CONTEXT_FILES = 12
_MAX_CONTEXT_FILE_BYTES = 96 * 1024
_MAX_CONTEXT_TOTAL_BYTES = 256 * 1024
_MAX_PATH_CHARS = 512

_MANIFEST_MARKER_SCRIPT = "scripts/gen_custom_ops.py"
_MANIFEST_MARKER_REGISTRY = "csrc/custom_ops/registry"
_MANIFEST_CONTEXT_PATHS: Tuple[str, ...] = (
    # Keep this set compatible with the existing CLI's manifest workflow.
    "scripts/gen_custom_ops.py",
    "csrc/custom_ops/registry/angle_v2/op.yaml",
    "tests/asnumpy_tests/custom_ops_tests/test_angle_v2.py",
    "CMakeLists.txt",
    "cmake/CustomOps.cmake",
    "csrc/custom_ops/CMakeLists.txt",
    "include/asnumpy/custom_ops/loader.hpp",
    "csrc/custom_ops/loader.cpp",
)

_LEGACY_CONTEXT_PATHS: Tuple[str, ...] = (
    # The enabled ``softmax`` call chain exists in the pinned 1929a3 source
    # and proves each manual integration layer without relying on numpyexample
    # file names added after that commit.
    "include/asnumpy/nn/activation.hpp",
    "csrc/nn/activation.cpp",
    "bindings/python/bind_nn.cpp",
    "bindings/python/module.cpp",
    "src/asnumpy/nn.py",
    "src/asnumpy/__init__.py",
    "bindings/python/CMakeLists.txt",
    "tests/asnumpy_tests/nn_tests/test_activation.py",
)

# An installed ACLNN candidate is not a custom OPP.  It must be wired into an
# existing AsNumPy module using an API-specific, locally reviewed path plan;
# allowing the model to invent a module or an overlay path would turn a
# semantic integration task into an unrestricted source-tree edit.  Each
# reviewed plan covers exactly one public proof case and names its signed
# ACLNN header; adding an API requires a new reviewed entry here.
_OFFICIAL_ACLNN_EXISTING_MODULE_PLANS = {
    "matmul": {
        "context_paths": (
            "include/asnumpy/linalg/product.hpp",
            "csrc/linalg/product.cpp",
            "csrc/linalg/CMakeLists.txt",
            "bindings/python/bind_linalg.cpp",
            "bindings/python/CMakeLists.txt",
            "src/asnumpy/linalg/direct.py",
            "src/asnumpy/linalg/__init__.py",
            "tests/asnumpy_tests/linalg_tests/test_product.py",
            # The Python dispatch edit must prove the exact ndarray property
            # surface and that ``strides`` are expressed in bytes.  Without
            # these two files a model could invent NumPy's ``.flags`` API or
            # apply an unsound C-contiguity check.
            "src/asnumpy/utils.py",
            "bindings/python/bind_utils.cpp",
        ),
        "expected_overlay_paths": (
            "csrc/linalg/product.cpp",
            "src/asnumpy/linalg/direct.py",
            "tests/asnumpy_tests/linalg_tests/test_product.py",
        ),
        # Schema-2 prompt dependencies are deliberately per target.  The
        # complete context above remains locally inspected and hash-bound, but
        # unrelated/downstream files are not repeated in every model request.
        "prompt_dependencies": {
            "csrc/linalg/product.cpp": (
                ("reference", "include/asnumpy/linalg/product.hpp"),
                (
                    "signed_official_header",
                    "SIGNED_OFFICIAL_HEADER/aclnn_matmul.h",
                ),
            ),
            "src/asnumpy/linalg/direct.py": (
                ("reference", "bindings/python/bind_linalg.cpp"),
            ),
            "tests/asnumpy_tests/linalg_tests/test_product.py": (
                ("generated_overlay", "src/asnumpy/linalg/direct.py"),
            ),
        },
        "signed_official_header": "SIGNED_OFFICIAL_HEADER/aclnn_matmul.h",
        # The selected installed DSO must be the exact library already named
        # by the reviewed AsNumPy link graph.  Merely proving that another DSO
        # exports the same ACLNN symbols is insufficient: ELF interposition
        # could otherwise make a successful NPU test exercise an unbound
        # implementation.  This exact CMake block is deliberately versioned
        # with the reviewed source context rather than inferred by the model.
        "official_library_binding": {
            "cmake_path": "bindings/python/CMakeLists.txt",
            "target": "_core",
            "link_item": "opapi",
            "selected_library_basename": "libopapi.so",
            "cmake_link_block": (
                "target_link_libraries(_core\n"
                "    PUBLIC\n"
                "    ascendcl\n"
                "    runtime\n"
                "    nnopbase\n"
                "    opapi\n"
                "    Python::NumPy\n"
                ")"
            ),
        },
    },
    "concatenate": {
        "context_paths": (
            "include/asnumpy/array/basic.hpp",
            "csrc/array/basic.cpp",
            "bindings/python/bind_array.cpp",
            "bindings/python/CMakeLists.txt",
            "include/asnumpy/utils/acl_executor.hpp",
            "include/asnumpy/utils/acl_resource.hpp",
            "include/asnumpy/utils/npu_array.hpp",
            "include/asnumpy/utils/status_handler.hpp",
            "src/asnumpy/array.py",
            "src/asnumpy/__init__.py",
            "src/asnumpy/utils.py",
            "tests/asnumpy_tests/creation_tests/test_basic.py",
        ),
        "expected_overlay_paths": (
            "include/asnumpy/array/basic.hpp",
            "csrc/array/basic.cpp",
            "bindings/python/bind_array.cpp",
            "src/asnumpy/array.py",
            "src/asnumpy/__init__.py",
            "tests/asnumpy_tests/creation_tests/test_basic.py",
        ),
        # The signed header declares only aclnnCat/aclnnCatGetWorkspaceSize;
        # the C++ implementation must build and destroy its aclTensorList with
        # aclCreateTensorList/aclDestroyTensorList from the signed ACL meta
        # header already included by npu_array.hpp.  Every target depends on
        # an earlier reviewed artifact so the binding, Python wrapper, export,
        # and test cannot disagree about the public signature.
        "prompt_dependencies": {
            "include/asnumpy/array/basic.hpp": (
                ("reference", "include/asnumpy/utils/npu_array.hpp"),
            ),
            "csrc/array/basic.cpp": (
                (
                    "generated_overlay",
                    "include/asnumpy/array/basic.hpp",
                ),
                (
                    "reference",
                    "include/asnumpy/utils/acl_resource.hpp",
                ),
                (
                    "signed_official_header",
                    "SIGNED_OFFICIAL_HEADER/aclnn_cat.h",
                ),
            ),
            "bindings/python/bind_array.cpp": (
                (
                    "generated_overlay",
                    "include/asnumpy/array/basic.hpp",
                ),
                ("generated_overlay", "csrc/array/basic.cpp"),
            ),
            "src/asnumpy/array.py": (
                (
                    "generated_overlay",
                    "bindings/python/bind_array.cpp",
                ),
            ),
            "src/asnumpy/__init__.py": (
                ("generated_overlay", "src/asnumpy/array.py"),
            ),
            "tests/asnumpy_tests/creation_tests/test_basic.py": (
                ("generated_overlay", "src/asnumpy/array.py"),
            ),
        },
        "signed_official_header": "SIGNED_OFFICIAL_HEADER/aclnn_cat.h",
        "official_library_binding": {
            "cmake_path": "bindings/python/CMakeLists.txt",
            "target": "_core",
            "link_item": "opapi",
            "selected_library_basename": "libopapi.so",
            "cmake_link_block": (
                "target_link_libraries(_core\n"
                "    PUBLIC\n"
                "    ascendcl\n"
                "    runtime\n"
                "    nnopbase\n"
                "    opapi\n"
                "    Python::NumPy\n"
                ")"
            ),
        },
    },
    # ``numpy.cbrt`` is not a plain fractional power: negative inputs need a
    # sign-preserving composition, and the signed ACLNN surface only declares
    # the two-stage ``aclnnPowTensorScalar`` pair.  The reviewed overlay stays
    # inside the existing math module and must prove signed zero, NaN,
    # infinities, integer conversion and float16/32/64 promotion before the
    # operator can be qualified.  No custom-ops path is permitted here.
    "cbrt": {
        "context_paths": (
            "include/asnumpy/math/exponents_and_logarithms.hpp",
            "csrc/math/exponents_and_logarithms.cpp",
            "bindings/python/bind_math.cpp",
            "bindings/python/CMakeLists.txt",
            "include/asnumpy/utils/acl_executor.hpp",
            "include/asnumpy/utils/acl_resource.hpp",
            "include/asnumpy/utils/npu_array.hpp",
            "include/asnumpy/utils/status_handler.hpp",
            "include/asnumpy/utils/dtype_promotion.hpp",
            "src/asnumpy/math.py",
            "src/asnumpy/__init__.py",
            "tests/asnumpy_tests/math_tests/test_exponents_and_logarithms.py",
        ),
        "expected_overlay_paths": (
            "include/asnumpy/math/exponents_and_logarithms.hpp",
            "csrc/math/exponents_and_logarithms.cpp",
            "bindings/python/bind_math.cpp",
            "src/asnumpy/math.py",
            "src/asnumpy/__init__.py",
            "tests/asnumpy_tests/math_tests/test_exponents_and_logarithms.py",
        ),
        "prompt_dependencies": {
            "include/asnumpy/math/exponents_and_logarithms.hpp": (
                ("reference", "include/asnumpy/utils/npu_array.hpp"),
            ),
            "csrc/math/exponents_and_logarithms.cpp": (
                (
                    "generated_overlay",
                    "include/asnumpy/math/exponents_and_logarithms.hpp",
                ),
                ("reference", "include/asnumpy/utils/acl_executor.hpp"),
                ("reference", "include/asnumpy/utils/dtype_promotion.hpp"),
                (
                    "signed_official_header",
                    "SIGNED_OFFICIAL_HEADER/aclnn_pow.h",
                ),
            ),
            "bindings/python/bind_math.cpp": (
                (
                    "generated_overlay",
                    "include/asnumpy/math/exponents_and_logarithms.hpp",
                ),
                (
                    "generated_overlay",
                    "csrc/math/exponents_and_logarithms.cpp",
                ),
            ),
            "src/asnumpy/math.py": (
                (
                    "generated_overlay",
                    "bindings/python/bind_math.cpp",
                ),
            ),
            "src/asnumpy/__init__.py": (
                ("generated_overlay", "src/asnumpy/math.py"),
            ),
            "tests/asnumpy_tests/math_tests/test_exponents_and_logarithms.py": (
                ("generated_overlay", "src/asnumpy/math.py"),
            ),
        },
        "signed_official_header": "SIGNED_OFFICIAL_HEADER/aclnn_pow.h",
        "official_library_binding": {
            "cmake_path": "bindings/python/CMakeLists.txt",
            "target": "_core",
            "link_item": "opapi",
            "selected_library_basename": "libopapi.so",
            "cmake_link_block": (
                "target_link_libraries(_core\n"
                "    PUBLIC\n"
                "    ascendcl\n"
                "    runtime\n"
                "    nnopbase\n"
                "    opapi\n"
                "    Python::NumPy\n"
                ")"
            ),
        },
    },
    # ``numpy.invert`` is dispatched from the existing AsNumPy ``logic``
    # module.  The signed ACLNN surface exposes only the elementwise
    # ``aclnnBitwiseNot`` pair (int8/16/32/64, uint8, bool); float inputs
    # must be refused, and bool must stay logical NOT while integers keep
    # two's-complement bit patterns.  No custom-ops path is permitted.
    "invert": {
        "context_paths": (
            "include/asnumpy/logic/logic.hpp",
            "csrc/logic/logic.cpp",
            "bindings/python/bind_logic.cpp",
            "bindings/python/CMakeLists.txt",
            "include/asnumpy/utils/acl_executor.hpp",
            "include/asnumpy/utils/acl_resource.hpp",
            "include/asnumpy/utils/npu_array.hpp",
            "include/asnumpy/utils/status_handler.hpp",
            "src/asnumpy/logic.py",
            "src/asnumpy/__init__.py",
            "tests/asnumpy_tests/logic_tests/test_logic.py",
        ),
        "expected_overlay_paths": (
            "include/asnumpy/logic/logic.hpp",
            "csrc/logic/logic.cpp",
            "bindings/python/bind_logic.cpp",
            "src/asnumpy/logic.py",
            "src/asnumpy/__init__.py",
        ),
        "prompt_dependencies": {
            "include/asnumpy/logic/logic.hpp": (
                ("reference", "include/asnumpy/utils/npu_array.hpp"),
            ),
            "csrc/logic/logic.cpp": (
                ("generated_overlay", "include/asnumpy/logic/logic.hpp"),
                ("reference", "include/asnumpy/utils/acl_executor.hpp"),
                ("reference", "include/asnumpy/utils/status_handler.hpp"),
                ("signed_official_header", "SIGNED_OFFICIAL_HEADER/aclnn_bitwise_not.h"),
            ),
            "bindings/python/bind_logic.cpp": (
                ("generated_overlay", "include/asnumpy/logic/logic.hpp"),
                ("generated_overlay", "csrc/logic/logic.cpp"),
            ),
            "src/asnumpy/logic.py": (
                ("generated_overlay", "bindings/python/bind_logic.cpp"),
            ),
            "src/asnumpy/__init__.py": (
                ("generated_overlay", "src/asnumpy/logic.py"),
            ),
        },
        "signed_official_header": "SIGNED_OFFICIAL_HEADER/aclnn_bitwise_not.h",
        "official_library_binding": {
            "cmake_path": "bindings/python/CMakeLists.txt",
            "target": "_core",
            "link_item": "opapi",
            "selected_library_basename": "libopapi.so",
            "cmake_link_block": (
                "target_link_libraries(_core\n"
                "    PUBLIC\n"
                "    ascendcl\n"
                "    runtime\n"
                "    nnopbase\n"
                "    opapi\n"
                "    Python::NumPy\n"
                ")"
            ),
        },
    },
    # ``numpy.argsort`` is dispatched from the existing AsNumPy ``sorting``
    # module, whose reviewed target link graph already names ``opapi``.
    # The ACLNN pair returns int64 order indices for one dim; the Python
    # surface must normalize axis, default to the last axis, and reject
    # kind/order arguments it cannot honor instead of ignoring them.
    "argsort": {
        "context_paths": (
            "include/asnumpy/sorting/sorting.hpp",
            "csrc/sorting/sorting.cpp",
            "bindings/python/bind_sorting.cpp",
            "bindings/python/CMakeLists.txt",
            "include/asnumpy/utils/acl_executor.hpp",
            "include/asnumpy/utils/acl_resource.hpp",
            "include/asnumpy/utils/npu_array.hpp",
            "include/asnumpy/utils/status_handler.hpp",
            "src/asnumpy/sorting.py",
            "src/asnumpy/__init__.py",
            "tests/asnumpy_tests/sort_tests/test_sorting.py",
        ),
        "expected_overlay_paths": (
            "include/asnumpy/sorting/sorting.hpp",
            "csrc/sorting/sorting.cpp",
            "bindings/python/bind_sorting.cpp",
            "src/asnumpy/sorting.py",
            "src/asnumpy/__init__.py",
        ),
        "prompt_dependencies": {
            "include/asnumpy/sorting/sorting.hpp": (
                ("reference", "include/asnumpy/utils/npu_array.hpp"),
            ),
            "csrc/sorting/sorting.cpp": (
                ("generated_overlay", "include/asnumpy/sorting/sorting.hpp"),
                ("reference", "include/asnumpy/utils/acl_executor.hpp"),
                ("reference", "include/asnumpy/utils/status_handler.hpp"),
                ("signed_official_header", "SIGNED_OFFICIAL_HEADER/aclnn_argsort.h"),
            ),
            "bindings/python/bind_sorting.cpp": (
                ("generated_overlay", "include/asnumpy/sorting/sorting.hpp"),
                ("generated_overlay", "csrc/sorting/sorting.cpp"),
            ),
            "src/asnumpy/sorting.py": (
                ("generated_overlay", "bindings/python/bind_sorting.cpp"),
            ),
            "src/asnumpy/__init__.py": (
                ("generated_overlay", "src/asnumpy/sorting.py"),
            ),
        },
        "signed_official_header": "SIGNED_OFFICIAL_HEADER/aclnn_argsort.h",
        "official_library_binding": {
            "cmake_path": "bindings/python/CMakeLists.txt",
            "target": "_core",
            "link_item": "opapi",
            "selected_library_basename": "libopapi.so",
            "cmake_link_block": (
                "target_link_libraries(_core\n"
                "    PUBLIC\n"
                "    ascendcl\n"
                "    runtime\n"
                "    nnopbase\n"
                "    opapi\n"
                "    Python::NumPy\n"
                ")"
            ),
        },
    },
    # ``numpy.searchsorted`` is dispatched from the existing AsNumPy
    # ``sorting`` module.  The ACLNN pair takes an explicit right flag and
    # a sorter operand; NumPy's default side='left' must be mapped to
    # right=False, and the int64 index dtype plus the empty-sequence
    # boundary must be proven by the API gate.
    "searchsorted": {
        "context_paths": (
            "include/asnumpy/sorting/sorting.hpp",
            "csrc/sorting/sorting.cpp",
            "bindings/python/bind_sorting.cpp",
            "bindings/python/CMakeLists.txt",
            "include/asnumpy/utils/acl_executor.hpp",
            "include/asnumpy/utils/acl_resource.hpp",
            "include/asnumpy/utils/npu_array.hpp",
            "include/asnumpy/utils/status_handler.hpp",
            "src/asnumpy/sorting.py",
            "src/asnumpy/__init__.py",
            "tests/asnumpy_tests/sort_tests/test_sorting.py",
        ),
        "expected_overlay_paths": (
            "include/asnumpy/sorting/sorting.hpp",
            "csrc/sorting/sorting.cpp",
            "bindings/python/bind_sorting.cpp",
            "src/asnumpy/sorting.py",
            "src/asnumpy/__init__.py",
        ),
        "prompt_dependencies": {
            "include/asnumpy/sorting/sorting.hpp": (
                ("reference", "include/asnumpy/utils/npu_array.hpp"),
            ),
            "csrc/sorting/sorting.cpp": (
                ("generated_overlay", "include/asnumpy/sorting/sorting.hpp"),
                ("reference", "include/asnumpy/utils/acl_executor.hpp"),
                ("reference", "include/asnumpy/utils/status_handler.hpp"),
                ("signed_official_header", "SIGNED_OFFICIAL_HEADER/aclnn_searchsorted.h"),
            ),
            "bindings/python/bind_sorting.cpp": (
                ("generated_overlay", "include/asnumpy/sorting/sorting.hpp"),
                ("generated_overlay", "csrc/sorting/sorting.cpp"),
            ),
            "src/asnumpy/sorting.py": (
                ("generated_overlay", "bindings/python/bind_sorting.cpp"),
            ),
            "src/asnumpy/__init__.py": (
                ("generated_overlay", "src/asnumpy/sorting.py"),
            ),
        },
        "signed_official_header": "SIGNED_OFFICIAL_HEADER/aclnn_searchsorted.h",
        "official_library_binding": {
            "cmake_path": "bindings/python/CMakeLists.txt",
            "target": "_core",
            "link_item": "opapi",
            "selected_library_basename": "libopapi.so",
            "cmake_link_block": (
                "target_link_libraries(_core\n"
                "    PUBLIC\n"
                "    ascendcl\n"
                "    runtime\n"
                "    nnopbase\n"
                "    opapi\n"
                "    Python::NumPy\n"
                ")"
            ),
        },
    },
    # ``numpy.median`` is dispatched from the existing AsNumPy
    # ``statistics`` module.  ``aclnnMedian`` reduces the flattened
    # tensor to a single value and is exported only from ``libopapi.so``;
    # axis/out/keepdims must be refused rather than approximated, and
    # even-length midpoint averaging must match NumPy exactly.
    "median": {
        "context_paths": (
            "include/asnumpy/statistics/averages_and_variances.hpp",
            "csrc/statistics/averages_and_variances.cpp",
            "bindings/python/bind_statistics.cpp",
            "bindings/python/CMakeLists.txt",
            "include/asnumpy/utils/acl_executor.hpp",
            "include/asnumpy/utils/acl_resource.hpp",
            "include/asnumpy/utils/npu_array.hpp",
            "include/asnumpy/utils/status_handler.hpp",
            "src/asnumpy/statistics.py",
            "src/asnumpy/__init__.py",
            "tests/asnumpy_tests/statistics_tests/test_averages_and_variances.py",
        ),
        "expected_overlay_paths": (
            "include/asnumpy/statistics/averages_and_variances.hpp",
            "csrc/statistics/averages_and_variances.cpp",
            "bindings/python/bind_statistics.cpp",
            "src/asnumpy/statistics.py",
            "src/asnumpy/__init__.py",
        ),
        "prompt_dependencies": {
            "include/asnumpy/statistics/averages_and_variances.hpp": (
                ("reference", "include/asnumpy/utils/npu_array.hpp"),
            ),
            "csrc/statistics/averages_and_variances.cpp": (
                ("generated_overlay", "include/asnumpy/statistics/averages_and_variances.hpp"),
                ("reference", "include/asnumpy/utils/acl_executor.hpp"),
                ("reference", "include/asnumpy/utils/status_handler.hpp"),
                ("signed_official_header", "SIGNED_OFFICIAL_HEADER/aclnn_median.h"),
            ),
            "bindings/python/bind_statistics.cpp": (
                ("generated_overlay", "include/asnumpy/statistics/averages_and_variances.hpp"),
                ("generated_overlay", "csrc/statistics/averages_and_variances.cpp"),
            ),
            "src/asnumpy/statistics.py": (
                ("generated_overlay", "bindings/python/bind_statistics.cpp"),
            ),
            "src/asnumpy/__init__.py": (
                ("generated_overlay", "src/asnumpy/statistics.py"),
            ),
        },
        "signed_official_header": "SIGNED_OFFICIAL_HEADER/aclnn_median.h",
        "official_library_binding": {
            "cmake_path": "bindings/python/CMakeLists.txt",
            "target": "_core",
            "link_item": "opapi",
            "selected_library_basename": "libopapi.so",
            "cmake_link_block": (
                "target_link_libraries(_core\n"
                "    PUBLIC\n"
                "    ascendcl\n"
                "    runtime\n"
                "    nnopbase\n"
                "    opapi\n"
                "    Python::NumPy\n"
                ")"
            ),
        },
    },
}


class AsnumpyAdapterContextError(ValueError):
    """The target AsNumpy tree cannot yield a safe adapter context."""


@dataclass(frozen=True)
class AsnumpyAdapterContextPlan:
    """Immutable inputs for one configured-model adapter generation call."""

    architecture: str
    mode: str
    reference_root: Path
    context: Mapping[str, str]
    context_sha256: Mapping[str, str]
    expected_overlay_paths: Tuple[str, ...]
    total_bytes: int
    plan_sha256: str
    schema_version: int = 1
    prompt_dependencies: Mapping[
        str, Tuple[Tuple[str, str], ...]
    ] = field(default_factory=lambda: MappingProxyType({}))
    official_library_binding: Mapping[str, object] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def manifest(self) -> Dict[str, object]:
        """Return the hash-bound, content-free plan receipt."""

        manifest = {
            "schema_version": self.schema_version,
            "architecture": self.architecture,
            "mode": self.mode,
            "reference_root": str(self.reference_root),
            "context_sha256": dict(self.context_sha256),
            "expected_overlay_paths": list(self.expected_overlay_paths),
            "total_bytes": self.total_bytes,
            "plan_sha256": self.plan_sha256,
        }
        if self.schema_version in (2, 3):
            manifest["prompt_dependencies"] = {
                target: [
                    {"source": source, "path": path}
                    for source, path in dependencies
                ]
                for target, dependencies in self.prompt_dependencies.items()
            }
        if self.schema_version == 3:
            manifest["official_library_binding"] = dict(
                self.official_library_binding
            )
        return manifest


def infer_asnumpy_adapter_api(paths: Sequence[str]) -> str:
    """Recover one public API identity from a validated overlay path set.

    Both supported layouts contain at least one API-bearing path.  Requiring
    every such path to agree prevents a repair from borrowing context for a
    different operator.  This helper inspects names only; the caller remains
    responsible for validating the report and candidate file hashes first.
    """

    if (
        isinstance(paths, (str, bytes))
        or not isinstance(paths, Sequence)
        or not paths
        or len(paths) > _MAX_CONTEXT_FILES
    ):
        raise AsnumpyAdapterContextError(
            "AsNumpy adapter overlay path set is invalid"
        )
    identities = set()
    normalized = set()
    for value in paths:
        path = str(_safe_relative_path(value))
        if path != value or path in normalized:
            raise AsnumpyAdapterContextError(
                "AsNumpy adapter overlay paths are invalid or duplicated"
            )
        normalized.add(path)
        for pattern in _ADAPTER_API_PATHS:
            match = pattern.fullmatch(path)
            if match is not None:
                identities.add(match.group(1))
    if len(identities) != 1:
        raise AsnumpyAdapterContextError(
            "AsNumpy adapter overlay does not bind exactly one API"
        )
    return next(iter(identities))


def infer_official_adapter_api(paths: Sequence[str]) -> str:
    """Recover the reviewed API identity of an official-module overlay.

    Official overlays edit pre-existing AsNumPy modules, so their file names
    do not encode the operator.  The reviewed existing-module plan table is
    the only authority: the candidate path set must equal exactly one
    reviewed ``expected_overlay_paths`` tuple.  The caller remains
    responsible for validating the adapter report and file hashes first.
    """

    if (
        isinstance(paths, (str, bytes))
        or not isinstance(paths, Sequence)
        or not paths
        or len(paths) > _MAX_CONTEXT_FILES
    ):
        raise AsnumpyAdapterContextError(
            "official ACLNN adapter overlay path set is invalid"
        )
    normalized = set()
    for value in paths:
        path = str(_safe_relative_path(value))
        if path != value or path in normalized:
            raise AsnumpyAdapterContextError(
                "official ACLNN adapter overlay paths are invalid or duplicated"
            )
        normalized.add(path)
    matches = [
        api_name
        for api_name, reviewed in _OFFICIAL_ACLNN_EXISTING_MODULE_PLANS.items()
        if set(reviewed["expected_overlay_paths"]) == normalized
    ]
    if len(matches) != 1:
        raise AsnumpyAdapterContextError(
            "official ACLNN adapter overlay does not match one reviewed plan"
        )
    return matches[0]


def _safe_relative_path(value: str) -> PurePosixPath:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > _MAX_PATH_CHARS
        or "\\" in value
        or any(ord(character) < 32 for character in value)
    ):
        raise AsnumpyAdapterContextError("AsNumpy context path is invalid")
    parts = value.split("/")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in parts):
        raise AsnumpyAdapterContextError("AsNumpy context path is unsafe")
    return path


def _reference_root(value: Union[str, os.PathLike[str]]) -> Path:
    try:
        candidate = Path(value).absolute()
    except (TypeError, ValueError, OSError) as exc:
        raise AsnumpyAdapterContextError(
            "AsNumpy reference root is invalid"
        ) from exc
    if candidate.is_symlink() or not candidate.is_dir():
        raise AsnumpyAdapterContextError(
            "AsNumpy reference root must be a real directory"
        )
    try:
        return candidate.resolve(strict=True)
    except OSError as exc:
        raise AsnumpyAdapterContextError(
            "cannot resolve AsNumpy reference root"
        ) from exc


def _inside_root(root: Path, target: Path, label: str) -> Path:
    try:
        resolved = target.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise AsnumpyAdapterContextError(
            "%s escapes the AsNumpy reference root" % label
        ) from exc
    return resolved


def _reject_symlink_components(root: Path, relative: PurePosixPath) -> Path:
    target = root
    for part in relative.parts:
        target = target / part
        if target.is_symlink():
            raise AsnumpyAdapterContextError(
                "AsNumpy reference path contains a symlink: %s" % relative
            )
    return target


def _read_context_file(root: Path, relative_value: str) -> Tuple[str, int, str]:
    relative = _safe_relative_path(relative_value)
    target = _reject_symlink_components(root, relative)
    resolved = _inside_root(root, target, "AsNumpy reference %s" % relative)
    if resolved != target:
        # ``root`` is canonical and every component was checked, so a mismatch
        # here indicates an unsafe late path substitution.
        raise AsnumpyAdapterContextError(
            "AsNumpy reference path is not canonical: %s" % relative
        )

    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(target, flags)
    except OSError as exc:
        raise AsnumpyAdapterContextError(
            "cannot open AsNumpy reference file: %s" % relative
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise AsnumpyAdapterContextError(
                "AsNumpy reference must be a regular file: %s" % relative
            )
        if before.st_size <= 0 or before.st_size > _MAX_CONTEXT_FILE_BYTES:
            raise AsnumpyAdapterContextError(
                "AsNumpy reference file is empty or too large: %s" % relative
            )
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read(_MAX_CONTEXT_FILE_BYTES + 1)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)

    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after or len(payload) != before.st_size:
        raise AsnumpyAdapterContextError(
            "AsNumpy reference changed while being read: %s" % relative
        )
    if target.is_symlink() or _inside_root(root, target, str(relative)) != resolved:
        raise AsnumpyAdapterContextError(
            "AsNumpy reference changed after validation: %s" % relative
        )
    try:
        content = payload.decode("utf-8")
    except UnicodeError as exc:
        raise AsnumpyAdapterContextError(
            "AsNumpy reference is not UTF-8: %s" % relative
        ) from exc
    if not content.strip() or "\x00" in content:
        raise AsnumpyAdapterContextError(
            "AsNumpy reference content is invalid: %s" % relative
        )
    digest = hashlib.sha256(payload).hexdigest()
    # Bind the exact bytes passed to the model, not a normalized text form.
    if hashlib.sha256(content.encode("utf-8")).hexdigest() != digest:
        raise AsnumpyAdapterContextError(
            "AsNumpy reference hash verification failed: %s" % relative
        )
    return content, len(payload), digest


def _manifest_architecture(root: Path) -> bool:
    script = root / _MANIFEST_MARKER_SCRIPT
    registry = root / _MANIFEST_MARKER_REGISTRY
    script_exists = os.path.lexists(script)
    registry_exists = os.path.lexists(registry)
    if script_exists != registry_exists:
        raise AsnumpyAdapterContextError(
            "AsNumpy manifest architecture is incomplete"
        )
    if not script_exists:
        return False
    script_relative = _safe_relative_path(_MANIFEST_MARKER_SCRIPT)
    registry_relative = _safe_relative_path(_MANIFEST_MARKER_REGISTRY)
    script = _reject_symlink_components(root, script_relative)
    registry = _reject_symlink_components(root, registry_relative)
    if not script.is_file() or not registry.is_dir():
        raise AsnumpyAdapterContextError(
            "AsNumpy manifest markers are not regular file/directory entries"
        )
    _inside_root(root, script, "AsNumpy manifest generator")
    _inside_root(root, registry, "AsNumpy manifest registry")
    return True


def _assert_overlay_targets_safe(root: Path, paths: Tuple[str, ...]) -> None:
    for value in paths:
        relative = _safe_relative_path(value)
        target = root
        for index, part in enumerate(relative.parts):
            target = target / part
            if target.is_symlink():
                raise AsnumpyAdapterContextError(
                    "AsNumpy overlay path contains a symlink: %s" % relative
                )
            if not os.path.lexists(target):
                break
            is_last = index == len(relative.parts) - 1
            if (is_last and not target.is_file()) or (
                not is_last and not target.is_dir()
            ):
                raise AsnumpyAdapterContextError(
                    "AsNumpy overlay path contains an unsafe entry: %s" % relative
                )
            try:
                target.resolve(strict=True).relative_to(root)
            except (OSError, ValueError) as exc:
                raise AsnumpyAdapterContextError(
                    "AsNumpy overlay path escapes the reference root: %s" % relative
                ) from exc


def plan_asnumpy_adapter_context(
    reference_root: Union[str, os.PathLike[str]],
    api_name: str,
) -> AsnumpyAdapterContextPlan:
    """Detect the target layout and return a hash-bound model context plan.

    No source file is created or modified.  ``api_name`` is deliberately
    restricted to the same public-name grammar as the current CLI.
    """

    if not isinstance(api_name, str) or not _SAFE_API_NAME.fullmatch(api_name):
        raise AsnumpyAdapterContextError("AsNumpy adapter API name is invalid")
    root = _reference_root(reference_root)
    if _manifest_architecture(root):
        architecture = MANIFEST_ARCHITECTURE
        mode = MANIFEST_MODE
        context_paths = _MANIFEST_CONTEXT_PATHS
        expected_paths = (
            "csrc/custom_ops/registry/%s/op.yaml" % api_name,
            "tests/asnumpy_tests/custom_ops_tests/test_%s.py" % api_name,
        )
    else:
        architecture = LEGACY_ARCHITECTURE
        mode = MANUAL_MODE
        context_paths = _LEGACY_CONTEXT_PATHS
        # Eight files are sufficient to express the entire legacy call chain.
        # The wrapper implementation is compiled directly into the pybind
        # target by the single CMake overlay, avoiding extra shared build files.
        expected_paths = (
            "include/asnumpy/custom_ops/%s.hpp" % api_name,
            "csrc/custom_ops/%s.cpp" % api_name,
            "bindings/python/bind_custom_ops.cpp",
            "bindings/python/module.cpp",
            "src/asnumpy/custom_ops.py",
            "src/asnumpy/__init__.py",
            "bindings/python/CMakeLists.txt",
            "tests/asnumpy_tests/custom_ops_tests/test_%s.py" % api_name,
        )

    if (
        not context_paths
        or len(context_paths) > _MAX_CONTEXT_FILES
        or len(set(context_paths)) != len(context_paths)
        or not expected_paths
        or len(expected_paths) > _MAX_CONTEXT_FILES
        or len(set(expected_paths)) != len(expected_paths)
    ):
        raise AsnumpyAdapterContextError(
            "AsNumpy adapter context plan exceeds its path limits"
        )
    _assert_overlay_targets_safe(root, expected_paths)

    context: Dict[str, str] = {}
    hashes: Dict[str, str] = {}
    total_bytes = 0
    for relative in context_paths:
        content, size, digest = _read_context_file(root, relative)
        total_bytes += size
        if total_bytes > _MAX_CONTEXT_TOTAL_BYTES:
            raise AsnumpyAdapterContextError(
                "AsNumpy reference context exceeds the total-size limit"
            )
        context[relative] = content
        hashes[relative] = digest

    identity = json.dumps(
        {
            "schema_version": 1,
            "architecture": architecture,
            "mode": mode,
            "context_sha256": hashes,
            "expected_overlay_paths": expected_paths,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return AsnumpyAdapterContextPlan(
        architecture=architecture,
        mode=mode,
        reference_root=root,
        context=MappingProxyType(context),
        context_sha256=MappingProxyType(hashes),
        expected_overlay_paths=expected_paths,
        total_bytes=total_bytes,
        plan_sha256=hashlib.sha256(identity).hexdigest(),
    )


def _prove_official_library_binding(
    *,
    reviewed: Mapping[str, object],
    context: Mapping[str, str],
    context_sha256: Mapping[str, str],
    selected_library_relative_path: str,
    selected_library_sha256: str,
) -> Mapping[str, object]:
    """Bind one installed DSO to the exact reviewed AsNumPy link item.

    This is intentionally not a general CMake interpreter.  The official
    existing-module route is allowed only for a locally reviewed link block;
    a different target layout needs a new reviewed plan.  The selected DSO
    digest remains bound to the signed operator candidate/adapter contract.
    """

    raw_binding = reviewed.get("official_library_binding")
    if not isinstance(raw_binding, Mapping) or set(raw_binding) != {
        "cmake_path",
        "target",
        "link_item",
        "selected_library_basename",
        "cmake_link_block",
    }:
        raise AsnumpyAdapterContextError(
            "official ACLNN library-binding plan is invalid"
        )
    try:
        library_path = _safe_relative_path(selected_library_relative_path)
    except AsnumpyAdapterContextError as exc:
        raise AsnumpyAdapterContextError(
            "official ACLNN selected library path is invalid"
        ) from exc
    if (
        not isinstance(selected_library_sha256, str)
        or _SHA256.fullmatch(selected_library_sha256) is None
        or selected_library_sha256 == "0" * 64
    ):
        raise AsnumpyAdapterContextError(
            "official ACLNN selected library digest is invalid"
        )
    cmake_path = raw_binding["cmake_path"]
    target = raw_binding["target"]
    link_item = raw_binding["link_item"]
    expected_basename = raw_binding["selected_library_basename"]
    link_block = raw_binding["cmake_link_block"]
    if (
        not all(
            isinstance(value, str) and value
            for value in (
                cmake_path,
                target,
                link_item,
                expected_basename,
                link_block,
            )
        )
        or str(library_path) != selected_library_relative_path
        or library_path.name != expected_basename
        or expected_basename != "lib%s.so" % link_item
        or cmake_path not in context
        or cmake_path not in context_sha256
        or context[cmake_path].count(link_block) != 1
        or context[cmake_path].count("target_link_libraries(_core") != 2
        or re.search(
            r"\b(?:opapi_nn|opapi_math|cust_opapi)\b",
            context[cmake_path],
        )
        is not None
    ):
        raise AsnumpyAdapterContextError(
            "selected official ACLNN library does not match the reviewed "
            "AsNumPy link graph"
        )
    return MappingProxyType({
        "schema_version": 1,
        "proof_kind": "reviewed_exact_cmake_target_link_item",
        "cmake_path": cmake_path,
        "cmake_sha256": context_sha256[cmake_path],
        "target": target,
        "link_item": link_item,
        "selected_library_relative_path": selected_library_relative_path,
        "selected_library_basename": expected_basename,
        "selected_library_sha256": selected_library_sha256,
    })


def plan_official_aclnn_adapter_context(
    reference_root: Union[str, os.PathLike[str]],
    api_name: str,
    *,
    selected_library_relative_path: str,
    selected_library_sha256: str,
) -> AsnumpyAdapterContextPlan:
    """Plan a reviewed existing-module overlay for an installed ACLNN API.

    This entry point is intentionally separate from
    :func:`plan_asnumpy_adapter_context`: the latter is a custom-OPP planner,
    while an official binary candidate already has a kernel and needs only an
    architecture-aware public API bridge.  Unsupported APIs fail closed until
    their exact AsNumPy module surface has been reviewed and added here.
    """

    if not isinstance(api_name, str) or not _SAFE_API_NAME.fullmatch(api_name):
        raise AsnumpyAdapterContextError("AsNumpy adapter API name is invalid")
    reviewed = _OFFICIAL_ACLNN_EXISTING_MODULE_PLANS.get(api_name)
    if reviewed is None:
        raise AsnumpyAdapterContextError(
            "official ACLNN existing-module plan is not reviewed for %s"
            % api_name
        )
    root = _reference_root(reference_root)
    context_paths = tuple(reviewed["context_paths"])
    expected_paths = tuple(reviewed["expected_overlay_paths"])
    prompt_dependencies = {
        target: tuple(dependencies)
        for target, dependencies in reviewed["prompt_dependencies"].items()
    }
    if (
        not context_paths
        or len(context_paths) > _MAX_OFFICIAL_CONTEXT_FILES
        or len(set(context_paths)) != len(context_paths)
        or not expected_paths
        or len(expected_paths) > _MAX_CONTEXT_FILES
        or len(set(expected_paths)) != len(expected_paths)
        or tuple(prompt_dependencies) != expected_paths
    ):
        raise AsnumpyAdapterContextError(
            "official ACLNN adapter plan exceeds its path limits"
        )
    seen_generated = set()
    for target in expected_paths:
        dependencies = prompt_dependencies[target]
        if not dependencies or len(dependencies) > _MAX_CONTEXT_FILES:
            raise AsnumpyAdapterContextError(
                "official ACLNN prompt dependency plan is invalid"
            )
        seen = set()
        for source, relative in dependencies:
            identity = (source, relative)
            if identity in seen:
                raise AsnumpyAdapterContextError(
                    "official ACLNN prompt dependency is duplicated"
                )
            seen.add(identity)
            if source == "reference":
                valid = relative in context_paths and relative != target
            elif source == "signed_official_header":
                valid = relative == reviewed["signed_official_header"]
            elif source == "generated_overlay":
                valid = relative in seen_generated
            else:
                valid = False
            if not valid:
                raise AsnumpyAdapterContextError(
                    "official ACLNN prompt dependency plan is invalid"
                )
        seen_generated.add(target)
    _assert_overlay_targets_safe(root, expected_paths)

    context: Dict[str, str] = {}
    hashes: Dict[str, str] = {}
    total_bytes = 0
    for relative in context_paths:
        content, size, digest = _read_context_file(root, relative)
        total_bytes += size
        if total_bytes > _MAX_CONTEXT_TOTAL_BYTES:
            raise AsnumpyAdapterContextError(
                "AsNumpy reference context exceeds the total-size limit"
            )
        context[relative] = content
        hashes[relative] = digest

    official_library_binding = _prove_official_library_binding(
        reviewed=reviewed,
        context=context,
        context_sha256=hashes,
        selected_library_relative_path=selected_library_relative_path,
        selected_library_sha256=selected_library_sha256,
    )

    dependency_manifest = {
        target: [
            {"source": source, "path": path}
            for source, path in dependencies
        ]
        for target, dependencies in prompt_dependencies.items()
    }
    identity = json.dumps(
        {
            "schema_version": 3,
            "architecture": OFFICIAL_ACLNN_ARCHITECTURE,
            "mode": EXISTING_MODULE_MODE,
            "context_sha256": hashes,
            "expected_overlay_paths": expected_paths,
            "prompt_dependencies": dependency_manifest,
            "official_library_binding": dict(official_library_binding),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return AsnumpyAdapterContextPlan(
        architecture=OFFICIAL_ACLNN_ARCHITECTURE,
        mode=EXISTING_MODULE_MODE,
        reference_root=root,
        context=MappingProxyType(context),
        context_sha256=MappingProxyType(hashes),
        expected_overlay_paths=expected_paths,
        total_bytes=total_bytes,
        plan_sha256=hashlib.sha256(identity).hexdigest(),
        schema_version=3,
        prompt_dependencies=MappingProxyType(prompt_dependencies),
        official_library_binding=official_library_binding,
    )


def reviewed_official_adapter_prompt_dependencies(
    expected_overlay_paths: Sequence[str],
    *,
    api_name: Optional[str] = None,
) -> Dict[str, List[Dict[str, str]]]:
    """Return the reviewed prompt-dependency manifest for an overlay shape.

    This is a read-only lookup used by the provider gate.  It fails closed for
    any path tuple that is not one of the reviewed existing-module plans, so a
    caller cannot manufacture a dependency set for an unreviewed overlay.

    ``argsort`` and ``searchsorted`` share the same sorting-module overlay
    shape but bind different signed ACLNN headers, so a shape-only lookup
    cannot disambiguate them.  When ``api_name`` is supplied the lookup is
    restricted to that reviewed plan; shape-only callers keep the historical
    first-match behavior.
    """

    paths = tuple(expected_overlay_paths)
    if api_name is None:
        candidates = tuple(_OFFICIAL_ACLNN_EXISTING_MODULE_PLANS.values())
    else:
        reviewed_plan = _OFFICIAL_ACLNN_EXISTING_MODULE_PLANS.get(api_name)
        candidates = (reviewed_plan,) if reviewed_plan is not None else ()
    for reviewed in candidates:
        if tuple(reviewed["expected_overlay_paths"]) != paths:
            continue
        return {
            target: [
                {"source": source, "path": relative}
                for source, relative in reviewed["prompt_dependencies"][target]
            ]
            for target in paths
        }
    raise AsnumpyAdapterContextError(
        "official ACLNN prompt dependency plan is not reviewed"
    )
