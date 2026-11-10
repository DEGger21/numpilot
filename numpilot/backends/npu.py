# This file is part of the NumPilot project at Harbin Institute of Technology (HIT).
#
# Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
# SPDX-License-Identifier: Apache-2.0
#
# Authors (accounts):
# - Su Tonghua <@sutonghua>
# - Li Xing <@li-xingHIT>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""真实 asnumpy(NPU)后端适配器。需已 build/install asnumpy。"""
from dataclasses import dataclass
import hashlib
import importlib.util as importlib_util
from importlib import metadata as importlib_metadata
import os
from pathlib import Path
import platform
import re
import stat
import sys
from types import SimpleNamespace
from typing import Tuple

from numpilot.capabilities import (
    CapabilityMatch,
    CapabilityManifest,
    ExecutionProvenance,
    OperatorCallProfile,
    OperatorCapability,
    OperatorSignature,
    RuntimeProfileGuard,
    RuntimeIdentity,
    TensorProfile,
    UNKNOWN_ENVIRONMENT_VALUE,
    _read_cann_version,
    detect_runtime_environment,
    match_capability_profile,
    tensor_profile_from_value,
    _strict_json_loads,
)
from numpilot.backends.base import ExecutionBackend
from numpilot.capability_catalog import (
    ASNUMPY_CAPABILITY_CATALOG,
    CapabilityCatalogError,
    validate_canonical_key,
)
from numpilot.coverage import SUPPORTED_OPS
from numpilot.capability_profile import OPS_NN_MATMUL_ASNUMPY_AUDIT_ID


# This is deliberately an exact version match.  A different wheel may change a
# Python wrapper without changing whether its public name exists, so version
# drift must be re-audited before it can make a strict-NPU performance claim.
AUDITED_ASNUMPY_VERSION = '0.3.1.dev70+g60d139c54.d20261108'
AUDITED_CANN_HOME = "/usr/local/Ascend/cann-9.2.0-beta.2"
AUDITED_CANN_VERSION = "9.2.0-beta.2"
AUDITED_SOC_MODEL = "Ascend910B3"
AUDITED_PYTHON_VERSION = "3.12.14"
ASNUMPY_AUDIT_ID = (
    "asnumpy-0.3.1-dev70-g60d139c54-runtime-"
    "init295976c5-cannac907b3a-linalg4e7d1c7e-"
    "random6c9c7891-math32a64a3f-logice03f3b0f-"
    "core44c5b349-v6"
)
ASNUMPY_SOURCE_BUILD_AUDIT_ID = (
    "asnumpy-0.3.1-dev70-g60d139c54-"
    "init295976c5-cannac907b3a-linalg4e7d1c7e-"
    "random6c9c7891-pinned-source-receipt-v4"
)
ASNUMPY_INTERNAL_COMPONENT_AUDIT_ID = (
    "asnumpy-0.3.1-dev70-g60d139c54-internal-component-"
    "manifestff21a984-wheel-d3ea03b2-core44c5b349-v3"
)
_ASNUMPY_INTERNAL_COMPONENT_ID = 'asnumpy-0.3.1.dev70+g60d139c54.d20261108-cp312-cp312-linux_aarch64'
_ASNUMPY_INTERNAL_COMPONENT_MANIFEST_SHA256 = 'ff21a9842bb18897a760a5088e46f7710b3f3f0ed10f4b215079355417a7694b'
_AUDITED_RUNTIME_FILE_SHA256 = {
    "__init__.py": "295976c53a4fccc2807f9a3ef81821c063f839e6b32e4150d19288ee7af9224c",
    "_core": "44c5b349b3f8ce16b4d45157a3c39b19ec4a3edfba3aa1687d43081077b9e511",
    "_types.py": "cae6a7d7c1180dde6329b7fff0a840eee57bbd15e5cc6acafaae2e79294eb3b7",
    "_version.py": "9b068dc0b9540bfdab873e630dcffc344b70bc9830aa048e395befbf9dbe72ab",
    "array.py": "63cc6edc57c5ff904b4f6fc33ac43c6982430b6d915635d5046cca7c44a887e3",
    "cann.py": "ac907b3a294e597e5429629387b9615abed963026a879ef411ba4a4ddecbcc3e",
    "linalg/__init__.py": "bae3efa81d9840d622c3f29ed677dfd834e7e2c2769b9a35daea0b9b55e52a83",
    "linalg/_linalg.py": "c5ddcc4f890588690882b927cb50854999dfabd95ea8082a68274500cbc71240",
    "linalg/direct.py": "4e7d1c7e24a2955a7926127fe5a6f259dc067a03d55e23fbf7b94782335e41a8",
    "logic.py": "e03f3b0f77458c40e94f4c8785e5ac3ecfb1b4c7b2a744e463c8b5da6ea6cf4a",
    "math.py": "32a64a3f246d8e44f8b6abe9dc23fda447e6835f494df52f68f638e3ff1bb71e",
    "nn.py": "c0b2ed95d70fa15e92e2939d76cc93da116b64b2b94579aeb684b03e1cf07748",
    "random/__init__.py": "3324a11a240100c612d157d17b4d653446281a4ebd8dcfabdec731ad61481b0a",
    "random/_random.py": "6c9c7891bd44892caad282c16bfd469f0ffee54f5616576777f8416d4a547b65",
    "sorting.py": "bc206227bbb2709cef9242db9b3de32e4eec15c58a9b83d568e9181729d9cf2c",
    "statistics.py": "175315eec27c8fa2875e236b3c88cb047d5e9d3cbc7735a3f1fae6231a14d3cd",
    "utils.py": "b3d6cb4e6c1d6cfec8a48f72d14ffd175c82867cc8ca8d8cfccac0eea91229b1"
}

# Narrow official-operator integration used by the ops-nn MatMul case.  The
# base checkout is recorded separately from the hash-bound working-tree patch
# and wheel in the private qualification receipt.
OPS_NN_MATMUL_ASNUMPY_VERSION = "0.1.dev118+g48de2d208.d20260827"
OPS_NN_MATMUL_ASNUMPY_REPOSITORY = (
    "https://gitcode.com/li-xingHIT/numpyexample.git"
)
OPS_NN_MATMUL_ASNUMPY_BASE_COMMIT = (
    "48de2d2082a0ea46c3ed23dd208b01f4e2a21c35"
)
OPS_NN_MATMUL_ASNUMPY_BASE_TREE = (
    "842ead54d7e15ee87b37ca6dc277e9ca38988680"
)
OPS_NN_MATMUL_AUDITED_NATIVE_OPS = frozenset({"full", "matmul"})
_OPS_NN_MATMUL_RUNTIME_FILE_SHA256 = {
    "__init__.py": "c89e0683a33300b9b9a7bb3ec3f02d12e93142454a2761d2b7d4ce7e54ea9111",
    "_types.py": "3686cc4be153e88950357aacdc71e90d98a71f2d01e659c3376ff4e0b2d66fa7",
    "_version.py": "bc5586754dff637d88ebb3c757ec77879bcab9a812e9eba7c4859a8cba6db97b",
    "array.py": "16c901f5937027d2ce6ae6f9d58813d389d20252315333c954ec5ccb9a3b4cc3",
    "cann.py": "ac907b3a294e597e5429629387b9615abed963026a879ef411ba4a4ddecbcc3e",
    "linalg/__init__.py": "bae3efa81d9840d622c3f29ed677dfd834e7e2c2769b9a35daea0b9b55e52a83",
    "linalg/_linalg.py": "d2965c2e3f7a6019e20c8e526a9ef78730f3836cb900d7d9fec5234a7eb70e88",
    "linalg/direct.py": "c710c02718193f49f0c0e0e66d8e314e356868f77fa2f75ed1f391a3af1c698b",
    "logic.py": "58746575dfbfd873cb9db2763e6f3acf960c4767829dae785c376ad266c65ac9",
    "math.py": "4244a978ecd7ec6bdc56fdf3a63121a455e7341f5a6df7f2e236ff46a404b9cc",
    "nn.py": "c0b2ed95d70fa15e92e2939d76cc93da116b64b2b94579aeb684b03e1cf07748",
    "random/__init__.py": "3324a11a240100c612d157d17b4d653446281a4ebd8dcfabdec731ad61481b0a",
    "random/_random.py": "6c9c7891bd44892caad282c16bfd469f0ffee54f5616576777f8416d4a547b65",
    "sorting.py": "bc206227bbb2709cef9242db9b3de32e4eec15c58a9b83d568e9181729d9cf2c",
    "statistics.py": "175315eec27c8fa2875e236b3c88cb047d5e9d3cbc7735a3f1fae6231a14d3cd",
    "utils.py": "a0c09ce1c26d3cfa36df539d316fe37d40e7402d56da6785e18681b8508ca3c8",
    "_core": "ec559b94668176dc4104514a67866743778acd6ce4e67af0485ff0b7b51967bc",
}
_OPS_NN_MATMUL_AUDIT_RECEIPT_ENV = (
    "NUMPILOT_OPS_NN_MATMUL_AUDIT_RECEIPT"
)
_OPS_NN_MATMUL_AUDIT_RECEIPT_SHA256 = (
    "6d26144af1ef0aae8bfc6ac90e18d3fac1b415ac9c2bc9909487c471dfc9e6fb"
)
_OPS_NN_MATMUL_WHEEL_SHA256 = (
    "c08b8a3b0ca034926f42d1594aab42cd8c793faeebee5376d611f62d8638f953"
)

# Historical P0 qualification wheel. This is deliberately a narrow source/binary audit,
# not a declaration that every API in the locally built wheel is strict NPU.
P0_ASNUMPY_VERSION = "0.1.dev106+g21368abf9.d20260821"
P0_ASNUMPY_AUDIT_ID = (
    "asnumpy-0.1.dev106-g21368abf9-poisson-core-"
    "init4f4e8083-math62e0ba09-utilsfd94c025-core964c17f3-v2"
)
P0_AUDITED_NATIVE_OPS = frozenset({"add", "multiply", "clip"})
_P0_RUNTIME_FILE_SHA256 = {
    "__init__.py": "4f4e8083755dde87bd6f0dc8c765da71984cb1346c97e8b6dbb97d714be34c04",
    "math.py": "62e0ba09576ead5d3b44f6b0b33311e849efd47ab0bb11f7875c0a43f499085c",
    "utils.py": "fd94c0254d9b02dbd55510e88e345606e69a84903a4aceeb238b435bea14beed",
    "_core": "964c17f3c910f54992fa24f086493481dd471d060f523f2e124cb8e8d6c373f3",
}
P0_LINALG_ASNUMPY_AUDIT_ID = (
    "asnumpy-0.1.dev106-g21368abf9-poisson-linalg-"
    "init4f4e8083-array16c901f5-math62e0ba09-direct4479c588-"
    "utilsfd94c025-cored3ccb39f-v4"
)
P0_LINALG_AUDITED_NATIVE_OPS = frozenset(
    {
        "absolute",
        "add",
        "amax",
        "clip",
        "cos",
        "divide",
        "exp",
        "full",
        "linspace",
        "matmul",
        "multiply",
        "negative",
        "outer",
        "sin",
        "sqrt",
        "subtract",
        "sum",
    }
)
P0_MOVE_ASNUMPY_VERSION = "0.1.dev106+g21368abf9.d20260822"
P0_MOVE_ASNUMPY_AUDIT_ID = (
    "asnumpy-0.1.dev106-g21368abf9-poisson-move-"
    "init4f4e8083-array16c901f5-math62e0ba09-direct4479c588-"
    "utils1b298145-core4465ce6e-v3"
)
# The published Poisson qualification/profiler receipts predate the array.py
# extension of the runtime identity.  They may be activated only when the
# current runtime first passes the stricter v2 identity and their exact legacy
# base-manifest hash is reproduced; ordinary manifests always expose v2.
P0_MOVE_LEGACY_OVERLAY_AUDIT_ID = (
    "asnumpy-0.1.dev106-g21368abf9-poisson-move-"
    "init4f4e8083-math62e0ba09-direct4479c588-"
    "utils1b298145-core4465ce6e-v1"
)
_P0_CURRENT_LINALG_AUDIT_IDS = frozenset({
    P0_LINALG_ASNUMPY_AUDIT_ID,
    P0_MOVE_ASNUMPY_AUDIT_ID,
})
_P0_LINALG_AUDIT_IDS = frozenset({
    *_P0_CURRENT_LINALG_AUDIT_IDS,
    P0_MOVE_LEGACY_OVERLAY_AUDIT_ID,
})
# ``full`` was admitted only in the v4/v3 source audit.  The historical v1
# identity must retain its original capability semantics when reconstructing a
# published qualification-overlay base.
_P0_LEGACY_LINALG_AUDITED_NATIVE_OPS = (
    P0_LINALG_AUDITED_NATIVE_OPS - {"full"}
)
# The cbrt showcase wheel preserves the hash-audited P0 add/multiply wrappers
# while adding one generated custom operator.  Only the inherited base
# operators are admitted here; cbrt remains unknown until an explicit
# qualification overlay binds its artifact, private OPP and float32 profile.
CBRT_ASNUMPY_VERSION = "0.0.1"
CBRT_ASNUMPY_AUDIT_ID = (
    "asnumpy-0.0.1-cbrt-integration-"
    "init4f4e8083-math62e0ba09-custom78937935-"
    "registry01370707-utils1b298145-core338dd5ad-v2"
)
CBRT_ASNUMPY_AUDITED_BASE_OPS = frozenset({"add", "multiply"})
_CBRT_ASNUMPY_RUNTIME_FILE_SHA256 = {
    "__init__.py": "4f4e8083755dde87bd6f0dc8c765da71984cb1346c97e8b6dbb97d714be34c04",
    "math.py": "62e0ba09576ead5d3b44f6b0b33311e849efd47ab0bb11f7875c0a43f499085c",
    "custom_ops.py": "78937935889331e5f24f92abf805c93e3cad6f77f1fffd921edbc7961995d793",
    "_custom_ops_registry.py": "01370707cc5d7f5586d054029806021b9cac2cb0e2c013bcae3e6f4015259b04",
    "utils.py": "1b2981456b2e4258b2f3f9611a07c98dc5a79acb8d67d35e3f2595af13911435",
    "_core": "338dd5ad32008ee12337d64f882a81ce0334b083a60007d648df37995fd1f42b",
}
_P0_SCOPED_MATH_SIGNATURES = {
    "divide": 2,
    "exp": 1,
    "negative": 1,
    "sqrt": 1,
    "subtract": 2,
}
_P0_LINALG_RUNTIME_FILE_SHA256 = {
    "__init__.py": "4f4e8083755dde87bd6f0dc8c765da71984cb1346c97e8b6dbb97d714be34c04",
    "array.py": "16c901f5937027d2ce6ae6f9d58813d389d20252315333c954ec5ccb9a3b4cc3",
    "math.py": "62e0ba09576ead5d3b44f6b0b33311e849efd47ab0bb11f7875c0a43f499085c",
    "linalg/direct.py": "4479c588c57705e229611a44844b6c7497deb6d97a7f067155f645b1f2913dbf",
    "utils.py": "fd94c0254d9b02dbd55510e88e345606e69a84903a4aceeb238b435bea14beed",
    "_core": "d3ccb39f23a24611d4e5f0f34605fdf6825c74a11ef8a1aa4dac498a0ff38821",
}
_P0_MOVE_RUNTIME_FILE_SHA256 = {
    "__init__.py": "4f4e8083755dde87bd6f0dc8c765da71984cb1346c97e8b6dbb97d714be34c04",
    "array.py": "16c901f5937027d2ce6ae6f9d58813d389d20252315333c954ec5ccb9a3b4cc3",
    "math.py": "62e0ba09576ead5d3b44f6b0b33311e849efd47ab0bb11f7875c0a43f499085c",
    "linalg/direct.py": "4479c588c57705e229611a44844b6c7497deb6d97a7f067155f645b1f2913dbf",
    "utils.py": "1b2981456b2e4258b2f3f9611a07c98dc5a79acb8d67d35e3f2595af13911435",
    "_core": "4465ce6e0589eb000d450a49c15488a4d2d9f211fa7867646992c56e7c040747",
}
_MAX_AUDITED_RUNTIME_FILE_BYTES = 64 * 1024 * 1024
_MAX_BOOTSTRAP_RECEIPT_BYTES = 2 * 1024 * 1024
_BOOTSTRAP_RECEIPT_ENV = "NUMPILOT_BOOTSTRAP_RECEIPT"
_ASNUMPY_SOURCE_RECEIPT_ACTIONS = frozenset(
    {
        "environment_gates_passed",
        "pinned_asnumpy_source_verified",
        "asnumpy_wheel_built",
        "built_wheel_installed",
        "asnumpy_official_readme_smoke_passed",
    }
)
_ASNUMPY_COMPONENT_RECEIPT_ACTIONS = (
    "environment_gates_passed",
    "internal_component_verified",
    "verified_component_wheel_installed",
    "asnumpy_official_readme_smoke_passed",
)
_ASNUMPY_COMPONENT_RECEIPT_ACTIONS_WITH_CANNBOT = (
    "environment_gates_passed",
    "internal_component_verified",
    "cannbot_bundle_verified",
    "verified_component_wheel_installed",
    "asnumpy_official_readme_smoke_passed",
)
_ASNUMPY_SOURCE_KEY_FILES = frozenset(
    {
        "LICENSE",
        "README.md",
        "pyproject.toml",
        "CMakeLists.txt",
        "bindings/python/CMakeLists.txt",
        "csrc/CMakeLists.txt",
    }
)


_CPU_FALLBACKS = {
    "inner": (
        ExecutionProvenance.CPU_FALLBACK,
        "asnumpy@1e7177f1: linalg/direct.py calls numpy.inner on host arrays",
        ("all inputs execute on CPU",),
    ),
    "matmul": (
        ExecutionProvenance.CPU_FALLBACK,
        "asnumpy@1e7177f1 generic audit: matmul is not admitted without the static/source receipt profile",
        ("requires the author-owned FP32 2-D admission profile",),
    ),
    "outer": (
        ExecutionProvenance.CPU_FALLBACK,
        "asnumpy@1e7177f1: linalg/direct.py calls numpy.outer on host arrays",
        ("all inputs execute on CPU",),
    ),
    "dot": (
        ExecutionProvenance.CONDITIONAL_CPU_FALLBACK,
        "asnumpy@1e7177f1: linalg/direct.py inspects host data and uses numpy.dot for float64",
        (
            "device inputs are copied to host for dtype inspection",
            "float64 input executes on CPU",
        ),
    ),
    "vdot": (
        ExecutionProvenance.CONDITIONAL_CPU_FALLBACK,
        "asnumpy@1e7177f1: linalg/direct.py inspects host data and uses numpy.vdot for float64",
        (
            "device inputs are copied to host for dtype inspection",
            "float64 input executes on CPU",
        ),
    ),
    "floor": (
        ExecutionProvenance.CONDITIONAL_CPU_FALLBACK,
        "asnumpy@1e7177f1: math.py inspects host data and uses numpy.floor for integer/bool",
        (
            "dtype=None copies device input to host for dtype inspection",
            "integer or bool input with dtype=None executes on CPU",
        ),
    ),
    "signbit": (
        ExecutionProvenance.HOST_ASSISTED,
        "asnumpy@1e7177f1: math.py copies input to host to patch IEEE -0.0",
        ("always performs host-side negative-zero detection",),
    ),
}


_GENERIC_LINALG_PROVENANCE = {
    "linalg.det": (
        ExecutionProvenance.COMPOSITE_NPU,
        "asnumpy@1e7177f1: linalg.det composes audited NPU bindings",
        (),
    ),
    "linalg.inv": (
        ExecutionProvenance.NATIVE_NPU,
        "asnumpy@1e7177f1: linalg.inv delegates to an NPU binding",
        (),
    ),
    "linalg.matrix_power": (
        ExecutionProvenance.CPU_FALLBACK,
        "asnumpy@1e7177f1: linalg.matrix_power computes with NumPy on the host",
        ("all inputs execute on CPU",),
    ),
    "linalg.norm": (
        ExecutionProvenance.CPU_FALLBACK,
        "asnumpy@1e7177f1: linalg.norm computes with NumPy on the host",
        ("all inputs execute on CPU",),
    ),
    "linalg.qr": (
        ExecutionProvenance.NATIVE_NPU,
        "asnumpy@1e7177f1: linalg.qr delegates to an NPU binding",
        (),
    ),
    "linalg.slogdet": (
        ExecutionProvenance.CONDITIONAL_CPU_FALLBACK,
        "asnumpy@1e7177f1: linalg.slogdet includes a conditional host fallback",
        ("some inputs execute on CPU",),
    ),
}
_GENERIC_RANDOM_NATIVE = frozenset(
    {
        "random.normal",
        "random.standard_normal",
        "random.uniform",
    }
)
_GENERIC_RANDOM_KEYS = ASNUMPY_CAPABILITY_CATALOG.keys_for_namespace("random")
_GENERIC_RANDOM_PROVENANCE = {
    name: (
        (
            ExecutionProvenance.NATIVE_NPU
            if name in _GENERIC_RANDOM_NATIVE
            else ExecutionProvenance.COMPOSITE_NPU
        ),
        (
            "asnumpy@1e7177f1: %s delegates to an NPU binding" % name
            if name in _GENERIC_RANDOM_NATIVE
            else "asnumpy@1e7177f1: %s composes audited NPU bindings" % name
        ),
        (),
    )
    for name in _GENERIC_RANDOM_KEYS
}
_GENERIC_NESTED_PROVENANCE = {
    **_GENERIC_LINALG_PROVENANCE,
    **_GENERIC_RANDOM_PROVENANCE,
}

if set(_GENERIC_LINALG_PROVENANCE) != set(
    ASNUMPY_CAPABILITY_CATALOG.keys_for_namespace("linalg")
):
    raise RuntimeError("generic linalg provenance does not match the fixed catalog")


# ``coverage.py`` contains a demo snapshot that also lists ``angle``, but the
# audited 1e7177f1 public mapping/source does not contain that API.  If a wheel
# with the same claimed version unexpectedly exposes it, runtime availability is
# recorded while strict provenance remains fail-closed.
_UNAUDITED_OPS = {"angle"}


@dataclass(frozen=True)
class OverlayRuntimeConfig:
    """Explicit, fail-closed location of a private qualification overlay."""

    overlay_path: str
    evidence_root: str
    trusted_producers: Tuple[str, ...] = (
        "numpilot-qualification-harness-v1",
        "numpilot-official-aclnn-qualification-v1",
    )

    def __post_init__(self):
        if not isinstance(self.overlay_path, str) or not self.overlay_path:
            raise ValueError("overlay_path must be non-empty text")
        if not isinstance(self.evidence_root, str) or not self.evidence_root:
            raise ValueError("evidence_root must be non-empty text")
        if (
            not isinstance(self.trusted_producers, tuple)
            or not self.trusted_producers
            or any(
                not isinstance(name, str) or not name
                for name in self.trusted_producers
            )
        ):
            raise ValueError("trusted_producers must be a non-empty tuple")


@dataclass(frozen=True)
class BaselineApiRuntimeConfig:
    """Explicit certificate for unchanged baseline APIs in an integrated wheel.

    This is an API/profile admission, never a whole-runtime identity audit.
    It is composed only after the configured official overlay passes its own
    unchanged base-manifest and runtime checks.
    """

    certificate_path: str
    evidence_root: str

    def __post_init__(self):
        for name in ("certificate_path", "evidence_root"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(name + " must be non-empty text")


def _canonical_capability_key(value):
    """Return one safe internal key without a source namespace."""

    try:
        return validate_canonical_key(value)
    except CapabilityCatalogError as exc:
        raise ValueError("canonical capability key is invalid: %s" % exc) from exc


def _canonical_key_from_qualified_source_api(qualified_name):
    """Translate an exact public NumPy/operator path to one manifest key."""

    if not isinstance(qualified_name, str):
        raise ValueError("profile capability requires a canonical source API")
    source_namespace, separator, remainder = qualified_name.partition(".")
    if not separator or source_namespace not in {"numpy", "operator"}:
        raise ValueError("profile capability requires a canonical source API")
    key = _canonical_capability_key(remainder)
    if source_namespace == "operator" and "." in key:
        raise ValueError("operator profile capability must be top-level")
    return key


def _resolve_canonical_runtime_api(module, canonical_key):
    """Safely traverse a validated public dotted path on the runtime module."""

    key = _canonical_capability_key(canonical_key)
    value = module
    for component in key.split("."):
        value = getattr(value, component)
    return value


class _CanonicalRuntimeModuleView:
    """Expose dotted canonical names to RuntimeProfileGuard integrity checks."""

    def __init__(self, module):
        self._module = module

    def __getattr__(self, canonical_key):
        return _resolve_canonical_runtime_api(self._module, canonical_key)


@dataclass(frozen=True)
class _AnalysisManifestResolver:
    """Immutable capability snapshot used for every callsite in one analysis."""

    manifest: CapabilityManifest

    def resolve_profile_capability(
        self,
        qualified_name: str,
        profile: OperatorCallProfile,
    ) -> CapabilityMatch:
        name = _canonical_key_from_qualified_source_api(qualified_name)
        if not isinstance(profile, OperatorCallProfile):
            raise TypeError("profile must be OperatorCallProfile")
        capability = self.manifest.operators.get(name)
        if capability is None:
            capability = OperatorCapability(name=name, available=False)
        return match_capability_profile(capability, profile)


def _sha256_private_regular_file(path):
    """Hash the exact regular file opened without following a final symlink."""

    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(str(path), flags)
    except OSError:
        return None
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_mode & 0o022
            or metadata.st_size > _MAX_AUDITED_RUNTIME_FILE_BYTES
        ):
            return None
        digest = hashlib.sha256()
        total = 0
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            total += len(block)
            if total > _MAX_AUDITED_RUNTIME_FILE_BYTES:
                return None
            digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass


def _p0_runtime_identity_matches(module):
    raw_path = getattr(module, "__file__", None)
    if not isinstance(raw_path, str) or not raw_path:
        return False
    init_path = Path(raw_path)
    if init_path.is_symlink() or init_path.name != "__init__.py":
        return False
    parent = init_path.parent
    core_paths = tuple(parent.glob("_core*.so"))
    if len(core_paths) != 1 or core_paths[0].is_symlink():
        return False
    observed = {
        "__init__.py": _sha256_private_regular_file(init_path),
        "math.py": _sha256_private_regular_file(parent / "math.py"),
        "utils.py": _sha256_private_regular_file(parent / "utils.py"),
        "_core": _sha256_private_regular_file(core_paths[0]),
    }
    return observed == _P0_RUNTIME_FILE_SHA256


def _audited_runtime_observation(module):
    """Return the hash-bound generic runtime files, or ``None`` on drift."""

    raw_path = getattr(module, "__file__", None)
    if not isinstance(raw_path, str) or not raw_path:
        return None
    init_path = Path(raw_path)
    if (
        not init_path.is_absolute()
        or init_path.is_symlink()
        or init_path.name != "__init__.py"
    ):
        return None
    parent = init_path.parent
    nested_directories = {
        (parent / relative).parent
        for relative in _AUDITED_RUNTIME_FILE_SHA256
        if relative != "_core" and "/" in relative
    }
    if any(
        directory.is_symlink() or not directory.is_dir()
        for directory in nested_directories
    ):
        return None
    core_paths = tuple(parent.glob("_core*.so"))
    if len(core_paths) != 1 or core_paths[0].is_symlink():
        return None
    observed = {
        relative: _sha256_private_regular_file(parent / relative)
        for relative in _AUDITED_RUNTIME_FILE_SHA256
        if relative != "_core"
    }
    observed["_core"] = _sha256_private_regular_file(core_paths[0])
    if any(value is None for value in observed.values()):
        return None
    return init_path, core_paths[0], observed


def _fixed_audited_runtime_identity_matches(module):
    observation = _audited_runtime_observation(module)
    return (
        observation is not None
        and observation[2] == _AUDITED_RUNTIME_FILE_SHA256
    )


def _ops_nn_matmul_runtime_identity_matches(module):
    observation = _audited_runtime_observation(module)
    return (
        observation is not None
        and observation[2] == _OPS_NN_MATMUL_RUNTIME_FILE_SHA256
    )


def _bootstrap_receipt_path():
    explicit = os.getenv(_BOOTSTRAP_RECEIPT_ENV)
    if explicit:
        path = Path(explicit).expanduser()
    else:
        from numpilot.paths import default_bootstrap_receipt

        path = default_bootstrap_receipt()
    return path if path.is_absolute() else None


def _read_private_bootstrap_receipt(path):
    if not isinstance(path, Path):
        return None
    try:
        parent_stat = path.parent.stat()
    except OSError:
        return None
    if (
        path.parent.is_symlink()
        or not stat.S_ISDIR(parent_stat.st_mode)
        or parent_stat.st_mode & 0o022
        or (
            hasattr(os, "getuid")
            and parent_stat.st_uid != os.getuid()
        )
    ):
        return None
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(str(path), flags)
    except OSError:
        return None
    try:
        file_stat = os.fstat(descriptor)
        if (
            not stat.S_ISREG(file_stat.st_mode)
            or file_stat.st_mode & 0o077
            or file_stat.st_size <= 0
            or file_stat.st_size > _MAX_BOOTSTRAP_RECEIPT_BYTES
            or (
                hasattr(os, "getuid")
                and file_stat.st_uid != os.getuid()
            )
        ):
            return None
        payload = bytearray()
        while True:
            block = os.read(descriptor, min(1024 * 1024, _MAX_BOOTSTRAP_RECEIPT_BYTES + 1))
            if not block:
                break
            payload.extend(block)
            if len(payload) > _MAX_BOOTSTRAP_RECEIPT_BYTES:
                return None
        parsed = _strict_json_loads(payload.decode("utf-8"))
        return parsed if type(parsed) is dict else None
    except (OSError, UnicodeError, ValueError, TypeError):
        return None
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass


def _ops_nn_matmul_audit_receipt():
    raw_path = os.getenv(_OPS_NN_MATMUL_AUDIT_RECEIPT_ENV)
    if not isinstance(raw_path, str) or not raw_path:
        return None
    path = Path(raw_path)
    if not path.is_absolute():
        return None
    receipt = _read_private_bootstrap_receipt(path)
    digest = _sha256_private_regular_file(path)
    if (
        receipt is None
        or digest != _OPS_NN_MATMUL_AUDIT_RECEIPT_SHA256
        or receipt.get("schema_version") != 1
        or receipt.get("producer")
        != "numpilot-ops-nn-matmul-qualification-v1"
        or receipt.get("status") != "qualified"
        or receipt.get("audit_id") != OPS_NN_MATMUL_ASNUMPY_AUDIT_ID
    ):
        return None
    try:
        asnumpy = receipt["asnumpy"]
        qualification = receipt["qualification"]
    except (KeyError, TypeError):
        return None
    if (
        type(asnumpy) is not dict
        or asnumpy.get("repository") != OPS_NN_MATMUL_ASNUMPY_REPOSITORY
        or asnumpy.get("base_commit") != OPS_NN_MATMUL_ASNUMPY_BASE_COMMIT
        or asnumpy.get("base_tree") != OPS_NN_MATMUL_ASNUMPY_BASE_TREE
        or asnumpy.get("version") != OPS_NN_MATMUL_ASNUMPY_VERSION
        or asnumpy.get("wheel_sha256") != _OPS_NN_MATMUL_WHEEL_SHA256
        or asnumpy.get("runtime_file_sha256")
        != _OPS_NN_MATMUL_RUNTIME_FILE_SHA256
        or type(qualification) is not dict
        or qualification.get("all_gates_passed") is not True
    ):
        return None
    return path, digest, receipt


def _sha256_text(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _source_build_receipt_identity_matches(module):
    """Admit a non-reproducible native build only through its exact receipt."""

    observation = _audited_runtime_observation(module)
    receipt_path = _bootstrap_receipt_path()
    if observation is None or receipt_path is None:
        return False
    init_path, core_path, observed = observation

    # Every audited Python wrapper remains tied to the reviewed source.  Only
    # the native module hash is allowed to vary between deterministic source
    # inputs, and it must be bound by the bootstrap receipt below.
    expected_python = {
        name: digest
        for name, digest in _AUDITED_RUNTIME_FILE_SHA256.items()
        if name != "_core"
    }
    if {name: observed[name] for name in expected_python} != expected_python:
        return False

    receipt = _read_private_bootstrap_receipt(receipt_path)
    if receipt is None:
        return False
    try:
        from numpilot import __version__ as numpilot_version
        from numpilot.bootstrap import (
            ASNUMPY_COMMIT,
            ASNUMPY_REPOSITORY,
            ASNUMPY_TREE,
            ASNUMPY_VERSION,
        )

        repositories = receipt["repositories"]
        source = repositories["asnumpy"]
        wheel = receipt["wheel"]
        trust = wheel["trust"]
        installed = wheel["installed_files"]
        verification = receipt["verification"]["asnumpy"]
        verification_native = verification["native_modules"]
        installed_native = installed["native_modules"]
        source_key_hashes = source["files_sha256"]
        actions = receipt["actions"]
    except (KeyError, TypeError):
        return False

    if (
        receipt.get("schema_version") != 1
        or receipt.get("producer") != "numpilot-bootstrap-v1"
        or receipt.get("numpilot_version") != numpilot_version
        or receipt.get("status") != "ready"
        or receipt.get("ready") is not True
        or not _sha256_text(receipt.get("plan_sha256"))
        or not isinstance(actions, list)
        or any(not isinstance(action, str) for action in actions)
        or not _ASNUMPY_SOURCE_RECEIPT_ACTIONS.issubset(actions)
    ):
        return False

    if (
        type(source) is not dict
        or source.get("repository") != ASNUMPY_REPOSITORY
        or source.get("commit") != ASNUMPY_COMMIT
        or source.get("tree") != ASNUMPY_TREE
        or source.get("mode") != "pinned-source-build"
        or type(source_key_hashes) is not dict
        or set(source_key_hashes) != _ASNUMPY_SOURCE_KEY_FILES
        or any(not _sha256_text(value) for value in source_key_hashes.values())
    ):
        return False

    if (
        type(wheel) is not dict
        or wheel.get("kind") != "built-from-pinned-source"
        or wheel.get("distribution") != "asnumpy"
        or wheel.get("version") != ASNUMPY_VERSION
        or not _sha256_text(wheel.get("sha256"))
        or not _sha256_text(wheel.get("metadata_sha256"))
        or not _sha256_text(wheel.get("record_sha256"))
        or type(trust) is not dict
        or trust.get("kind") != "built-from-pinned-source"
        or trust.get("repository") != ASNUMPY_REPOSITORY
        or trust.get("commit") != ASNUMPY_COMMIT
        or trust.get("tree") != ASNUMPY_TREE
    ):
        return False

    try:
        module_path = str(init_path.resolve(strict=True))
        interpreter = str(Path(sys.executable).resolve(strict=True))
        receipt_interpreter = str(
            Path(verification["interpreter"]).resolve(strict=True)
        )
        installed_module = installed["module"]
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        return False

    native_hash = observed["_core"]
    module_hash = observed["__init__.py"]
    if (
        verification.get("status") != "passed"
        or verification.get("source")
        != "official AsNumpy README.md installation verification"
        or verification.get("asnumpy_version") != ASNUMPY_VERSION
        or verification.get("module_path") != module_path
        or verification.get("module_sha256") != module_hash
        or verification.get("wheel_file_identity_verified") is not True
        or verification.get("shape") != [1000, 1000]
        or verification.get("dtype") != "float32"
        or receipt_interpreter != interpreter
        or type(installed_module) is not dict
        or installed_module.get("path") != "asnumpy/__init__.py"
        or installed_module.get("sha256") != module_hash
        or not isinstance(verification_native, list)
        or not isinstance(installed_native, list)
        or len(verification_native) != 1
        or len(installed_native) != 1
        or type(verification_native[0]) is not dict
        or type(installed_native[0]) is not dict
        or verification_native[0].get("path") != core_path.name
        or verification_native[0].get("sha256") != native_hash
        or installed_native[0].get("path") != "asnumpy/" + core_path.name
        or installed_native[0].get("sha256") != native_hash
    ):
        return False

    try:
        installed_asnumpy_version = importlib_metadata.version("asnumpy")
        installed_numpy_version = importlib_metadata.version("numpy")
        installed_loguru_version = importlib_metadata.version("loguru")
    except importlib_metadata.PackageNotFoundError:
        return False
    return (
        installed_asnumpy_version == ASNUMPY_VERSION
        and verification.get("numpy_version") == installed_numpy_version
        and verification.get("loguru_version") == installed_loguru_version
    )


def _internal_component_receipt_identity_matches(module):
    """Admit only the exact release-bundled AsNumpy component and receipt.

    The component is deliberately a third identity class.  Although its
    source commit is pinned, the installed native object is the immutable
    release attachment rather than a native object built by this bootstrap
    run, so it must never inherit the ``pinned-source-build`` audit label.
    """

    observation = _audited_runtime_observation(module)
    receipt_path = _bootstrap_receipt_path()
    if observation is None or receipt_path is None:
        return False
    init_path, core_path, observed = observation

    expected_python = {
        name: digest
        for name, digest in _AUDITED_RUNTIME_FILE_SHA256.items()
        if name != "_core"
    }
    if {name: observed[name] for name in expected_python} != expected_python:
        return False

    receipt = _read_private_bootstrap_receipt(receipt_path)
    if receipt is None:
        return False
    try:
        from numpilot import __version__ as numpilot_version
        from numpilot.asnumpy_component import (
            component_receipt_identity,
            load_component_manifest,
        )
        from numpilot.bootstrap import (
            ASNUMPY_COMMIT,
            ASNUMPY_LICENSE,
            ASNUMPY_REPOSITORY,
            ASNUMPY_TREE,
            ASNUMPY_VERSION,
            _verify_internal_component,
        )

        component = load_component_manifest()
        expected_component = component_receipt_identity(component)
        expected_wheel = _verify_internal_component(component)
        source = receipt["repositories"]["asnumpy"]
        wheel = receipt["wheel"]
        verification = receipt["verification"]["asnumpy"]
        environment = receipt["environment"]
        receipt_python = environment["python"]
        receipt_machine = environment["machine"]
        receipt_cann = environment["cann"]
        actions = receipt["actions"]
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        return False
    except Exception:
        # Component parsing and wheel/attachment verification expose their own
        # exception types.  Identity discovery is a gate, so every malformed
        # or unavailable release attachment degrades to audit-required.
        return False

    if (
        expected_component.get("manifest_sha256")
        != _ASNUMPY_INTERNAL_COMPONENT_MANIFEST_SHA256
        or component.component_id != _ASNUMPY_INTERNAL_COMPONENT_ID
        or component.artifact_required is not True
        or component.artifact_status != "bundled-and-verified"
        or component.repository != ASNUMPY_REPOSITORY
        or component.commit != ASNUMPY_COMMIT
        or component.tree != ASNUMPY_TREE
        or component.version != ASNUMPY_VERSION
        or component.license != ASNUMPY_LICENSE
        or component.build.get("source_commit") != ASNUMPY_COMMIT
        or component.build.get("source_tree") != ASNUMPY_TREE
        or set(component.source_files_sha256) != _ASNUMPY_SOURCE_KEY_FILES
        or any(
            not _sha256_text(value)
            for value in component.source_files_sha256.values()
        )
    ):
        return False

    if (
        receipt.get("schema_version") != 1
        or receipt.get("producer") != "numpilot-bootstrap-v1"
        or receipt.get("numpilot_version") != numpilot_version
        or receipt.get("status") != "ready"
        or receipt.get("ready") is not True
        or not _sha256_text(receipt.get("plan_sha256"))
        or type(actions) is not list
        or tuple(actions)
        not in {
            _ASNUMPY_COMPONENT_RECEIPT_ACTIONS,
            _ASNUMPY_COMPONENT_RECEIPT_ACTIONS_WITH_CANNBOT,
        }
    ):
        return False

    expected_source = {
        "mode": "internal-component",
        "source_repository_claimed": True,
        "component": expected_component,
        "repository": ASNUMPY_REPOSITORY,
        "commit": ASNUMPY_COMMIT,
        "tree": ASNUMPY_TREE,
        "license": ASNUMPY_LICENSE,
    }
    expected_wheel_receipt = {"kind": "internal-component", **expected_wheel}
    if (
        type(source) is not dict
        or source != expected_source
        or type(wheel) is not dict
        or wheel != expected_wheel_receipt
    ):
        return False

    runtime = component.runtime
    required_runtime = {
        "cann_version",
        "machine",
        "loguru_version",
        "numpy_version",
        "python_implementation",
        "python_major_minor",
    }
    if type(runtime) is not dict or set(runtime) != required_runtime:
        return False
    try:
        interpreter = str(Path(sys.executable).resolve(strict=True))
        receipt_interpreter = str(
            Path(verification["interpreter"]).resolve(strict=True)
        )
        plan_interpreter = str(
            Path(receipt_python["executable"]).resolve(strict=True)
        )
        module_path = str(init_path.resolve(strict=True))
        installed_asnumpy_version = importlib_metadata.version("asnumpy")
        installed_numpy_version = importlib_metadata.version("numpy")
        installed_loguru_version = importlib_metadata.version("loguru")
        current_python_version = platform.python_version()
        current_python_implementation = platform.python_implementation()
        current_python_major_minor = "%d.%d" % sys.version_info[:2]
        current_machine = platform.machine()
        _, current_cann_version = _runtime_identity_cann_facts()
    except (
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        importlib_metadata.PackageNotFoundError,
    ):
        return False

    expected_installed = expected_wheel.get("installed_files")
    if type(expected_installed) is not dict:
        return False
    installed_module = expected_installed.get("module")
    installed_native = expected_installed.get("native_modules")
    verification_native = verification.get("native_modules")
    native_hash = observed["_core"]
    module_hash = observed["__init__.py"]
    if (
        type(verification) is not dict
        or verification.get("status") != "passed"
        or verification.get("source")
        != "official AsNumpy README.md installation verification"
        or verification.get("asnumpy_version") != ASNUMPY_VERSION
        or verification.get("module_path") != module_path
        or verification.get("module_sha256") != module_hash
        or verification.get("wheel_file_identity_verified") is not True
        or verification.get("shape") != [1000, 1000]
        or verification.get("dtype") != "float32"
        or receipt_interpreter != interpreter
        or plan_interpreter != interpreter
        or type(receipt_python) is not dict
        or receipt_python.get("version") != current_python_version
        or receipt_python.get("implementation") != current_python_implementation
        or receipt_python.get("major_minor") != current_python_major_minor
        or type(receipt_machine) is not dict
        or receipt_machine.get("machine") != current_machine
        or type(receipt_cann) is not dict
        or receipt_cann.get("version") != current_cann_version
        or receipt_cann.get("version_conflict") is not False
        or type(installed_module) is not dict
        or installed_module.get("path") != "asnumpy/__init__.py"
        or installed_module.get("sha256") != module_hash
        or type(installed_native) is not list
        or type(verification_native) is not list
        or len(installed_native) != 1
        or len(verification_native) != 1
        or type(installed_native[0]) is not dict
        or type(verification_native[0]) is not dict
        or installed_native[0].get("path") != "asnumpy/" + core_path.name
        or installed_native[0].get("sha256") != native_hash
        or verification_native[0].get("path") != core_path.name
        or verification_native[0].get("sha256") != native_hash
    ):
        return False

    return (
        installed_asnumpy_version == ASNUMPY_VERSION
        and runtime["cann_version"] == current_cann_version
        and runtime["machine"] == current_machine
        and runtime["numpy_version"] == installed_numpy_version
        and runtime["loguru_version"] == installed_loguru_version
        and runtime["python_implementation"] == current_python_implementation
        and runtime["python_major_minor"] == current_python_major_minor
        and verification.get("numpy_version") == installed_numpy_version
        and verification.get("loguru_version") == installed_loguru_version
    )


def _audited_runtime_identity_matches(module):
    """Bind a fixed wheel, source receipt, or internal component.

    Once a bootstrap receipt exists, the receipt-bound identity is authoritative.
    This prevents a component with identical file hashes from bypassing a
    modified or invalid receipt through the generic hash profile.
    """

    receipt_path = _bootstrap_receipt_path()
    if receipt_path is not None and os.path.lexists(str(receipt_path)):
        return (
            _internal_component_receipt_identity_matches(module)
            or _source_build_receipt_identity_matches(module)
        )
    return _fixed_audited_runtime_identity_matches(module)


_RUNTIME_IDENTITY_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+:-]{0,255}$")
_RUNTIME_IDENTITY_GIT_ID = re.compile(r"^[0-9a-f]{40}$")


def _identity_token(value):
    if isinstance(value, str) and _RUNTIME_IDENTITY_TOKEN.fullmatch(value):
        return value
    return UNKNOWN_ENVIRONMENT_VALUE


def _identity_repository(value):
    if (
        isinstance(value, str)
        and value == value.strip()
        and 0 < len(value) <= 4096
        and "://" in value
        and not any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        return value
    return UNKNOWN_ENVIRONMENT_VALUE


def _identity_git_id(value):
    if isinstance(value, str) and _RUNTIME_IDENTITY_GIT_ID.fullmatch(value):
        return value
    return UNKNOWN_ENVIRONMENT_VALUE


def _identity_path(value, *, require_existing=False, require_directory=False):
    """Return one bounded absolute path, or the fail-closed unknown marker."""

    if not isinstance(value, (str, os.PathLike)):
        return UNKNOWN_ENVIRONMENT_VALUE
    try:
        path = Path(value)
        if not path.is_absolute():
            return UNKNOWN_ENVIRONMENT_VALUE
        if require_existing:
            path = path.resolve(strict=True)
            if require_directory and not path.is_dir():
                return UNKNOWN_ENVIRONMENT_VALUE
        text = str(path)
    except (OSError, RuntimeError, TypeError, ValueError):
        return UNKNOWN_ENVIRONMENT_VALUE
    if (
        not text
        or len(text) > 4096
        or text != text.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in text)
    ):
        return UNKNOWN_ENVIRONMENT_VALUE
    return text


def _read_only_npu_smi_soc_model():
    """Return one sanitized hardware SoC observation, or ``unknown``."""

    try:
        from numpilot.env_check import _probe_npu_smi_soc_model

        return _identity_token(_probe_npu_smi_soc_model())
    except Exception:
        return UNKNOWN_ENVIRONMENT_VALUE


def _npu_runtime_environment(module):
    """Capture capability facts with one read-only SoC fallback.

    The generic capability helper intentionally leaves an absent SoC value as
    unknown.  NpuBackend can safely improve only that field with the same
    ``npu-smi info`` probe used by its strict runtime identity.  An unavailable,
    malformed, or failing probe remains ``unknown``.
    """

    unknown = UNKNOWN_ENVIRONMENT_VALUE
    try:
        environment = dict(detect_runtime_environment(module))
    except Exception:
        environment = {
            "module_path": unknown,
            "cann_home": unknown,
            "cann_version": unknown,
            "soc_version": unknown,
            "python_version": _identity_token(platform.python_version()),
        }
    soc_version = _identity_token(environment.get("soc_version"))
    if soc_version == unknown:
        soc_version = _read_only_npu_smi_soc_model()
    environment["soc_version"] = soc_version
    return environment


def _metadata_only_asnumpy_module():
    """Locate AsNumpy without importing it or initialising its native runtime."""

    try:
        specification = importlib_util.find_spec("asnumpy")
    except Exception:
        specification = None
    origin = getattr(specification, "origin", None)
    try:
        version = importlib_metadata.version("asnumpy")
    except Exception:
        version = None
    return SimpleNamespace(__file__=origin, __version__=version)


def _ops_nn_matmul_runtime_identity(module):
    """Build the exact local qualification identity for the MatMul case."""

    unknown = UNKNOWN_ENVIRONMENT_VALUE
    try:
        version = _identity_token(getattr(module, "__version__", None))
    except Exception:
        version = unknown
    file_facts = _runtime_identity_file_facts(module)
    cann_home, cann_version = _runtime_identity_cann_facts()
    # A strict identity must observe hardware, never trust module attributes or
    # environment variables that can be supplied by the caller.
    soc_model = _read_only_npu_smi_soc_model()
    python_executable = _identity_path(sys.executable, require_existing=True)
    try:
        python_version = _identity_token(platform.python_version())
    except Exception:
        python_version = unknown
    try:
        audit_id = _identity_token(_runtime_audit_id(module, version))
    except Exception:
        audit_id = unknown

    receipt_result = _ops_nn_matmul_audit_receipt()
    if receipt_result is None:
        receipt_path = unknown
        receipt_sha256 = unknown
        receipt = {}
    else:
        raw_receipt_path, receipt_sha256, receipt = receipt_result
        receipt_path = _identity_path(raw_receipt_path, require_existing=True)

    issues = set()
    if version != OPS_NN_MATMUL_ASNUMPY_VERSION:
        issues.add("AsNumpy version does not match the MatMul integration audit")
    if audit_id != OPS_NN_MATMUL_ASNUMPY_AUDIT_ID:
        issues.add("AsNumpy runtime hashes do not match the MatMul integration audit")

    expected_wrappers = {
        relative: digest
        for relative, digest in _OPS_NN_MATMUL_RUNTIME_FILE_SHA256.items()
        if relative != "_core"
    }
    if file_facts["wrapper_sha256"] != expected_wrappers:
        issues.add("AsNumpy wrapper hashes do not match the MatMul integration audit")
    if file_facts["module_path"] == unknown:
        issues.add("AsNumpy module path is unavailable")
    if file_facts["module_sha256"] != expected_wrappers["__init__.py"]:
        issues.add("AsNumpy module hash does not match the MatMul integration audit")
    if file_facts["native_module_path"] == unknown:
        issues.add("AsNumpy native module path is unavailable")
    if (
        file_facts["native_module_sha256"]
        != _OPS_NN_MATMUL_RUNTIME_FILE_SHA256["_core"]
    ):
        issues.add("AsNumpy native module hash does not match the MatMul integration audit")

    if receipt_result is None:
        issues.add("MatMul qualification receipt is missing, invalid, or drifted")
    else:
        try:
            receipt_runtime = receipt["runtime"]
        except (KeyError, TypeError):
            receipt_runtime = {}
        if (
            type(receipt_runtime) is not dict
            or receipt_runtime.get("python_executable") != python_executable
            or receipt_runtime.get("module_path") != file_facts["module_path"]
            or receipt_runtime.get("native_module_path")
            != file_facts["native_module_path"]
        ):
            issues.add("MatMul qualification receipt does not bind this runtime")

    expected_cann_home = _identity_path(AUDITED_CANN_HOME)
    if (
        cann_home == unknown
        or expected_cann_home == unknown
        or cann_home != expected_cann_home
    ):
        issues.add("CANN home does not match the MatMul integration audit")
    if cann_version != AUDITED_CANN_VERSION:
        issues.add("CANN version does not match the MatMul integration audit")
    if soc_model != AUDITED_SOC_MODEL:
        issues.add("SoC model does not match the MatMul integration audit")
    if python_version != AUDITED_PYTHON_VERSION:
        issues.add("Python version does not match the MatMul integration audit")
    if python_executable == unknown:
        issues.add("Python executable is unavailable")

    status = "matched" if not issues else "audit_required"
    return RuntimeIdentity(
        asnumpy_repository=OPS_NN_MATMUL_ASNUMPY_REPOSITORY,
        asnumpy_commit=OPS_NN_MATMUL_ASNUMPY_BASE_COMMIT,
        asnumpy_tree=OPS_NN_MATMUL_ASNUMPY_BASE_TREE,
        asnumpy_version=version,
        wrapper_sha256=file_facts["wrapper_sha256"],
        module_path=file_facts["module_path"],
        module_sha256=file_facts["module_sha256"],
        native_module_path=file_facts["native_module_path"],
        native_module_sha256=file_facts["native_module_sha256"],
        cann_home=cann_home,
        cann_version=cann_version,
        soc_model=soc_model,
        python_executable=python_executable,
        python_version=python_version,
        bootstrap_receipt_path=receipt_path,
        bootstrap_receipt_sha256=receipt_sha256,
        audit_id=audit_id,
        status=status,
        issues=tuple(sorted(issues)),
    )


def _runtime_identity_file_facts(module):
    """Hash audited files without importing AsNumpy or following final symlinks."""

    unknown = UNKNOWN_ENVIRONMENT_VALUE
    wrapper_hashes = {
        relative: unknown
        for relative in _AUDITED_RUNTIME_FILE_SHA256
        if relative != "_core"
    }
    facts = {
        "module_path": unknown,
        "module_sha256": unknown,
        "native_module_path": unknown,
        "native_module_sha256": unknown,
        "wrapper_sha256": wrapper_hashes,
    }
    try:
        raw_path = getattr(module, "__file__", None)
    except Exception:
        return facts
    if not isinstance(raw_path, str) or not raw_path:
        return facts
    init_path = Path(raw_path)
    if (
        not init_path.is_absolute()
        or init_path.is_symlink()
        or init_path.name != "__init__.py"
    ):
        return facts
    try:
        resolved_init = init_path.resolve(strict=True)
    except (OSError, RuntimeError):
        return facts
    facts["module_path"] = _identity_path(resolved_init, require_existing=True)
    module_hash = _sha256_private_regular_file(init_path)
    if module_hash is not None:
        facts["module_sha256"] = module_hash

    parent = init_path.parent
    for relative in wrapper_hashes:
        candidate = parent / relative
        relative_path = Path(relative)
        directory = parent
        safe = True
        for component in relative_path.parts[:-1]:
            directory = directory / component
            if directory.is_symlink() or not directory.is_dir():
                safe = False
                break
        if not safe:
            continue
        digest = _sha256_private_regular_file(candidate)
        if digest is not None:
            wrapper_hashes[relative] = digest

    try:
        core_paths = tuple(parent.glob("_core*.so"))
    except (OSError, ValueError):
        core_paths = ()
    if len(core_paths) == 1 and not core_paths[0].is_symlink():
        core_path = core_paths[0]
        try:
            resolved_core = core_path.resolve(strict=True)
        except (OSError, RuntimeError):
            resolved_core = None
        if resolved_core is not None:
            facts["native_module_path"] = _identity_path(
                resolved_core, require_existing=True
            )
            core_hash = _sha256_private_regular_file(core_path)
            if core_hash is not None:
                facts["native_module_sha256"] = core_hash
    return facts


def _runtime_identity_receipt_facts():
    unknown = UNKNOWN_ENVIRONMENT_VALUE
    try:
        receipt_path = _bootstrap_receipt_path()
    except (OSError, RuntimeError, TypeError, ValueError):
        receipt_path = None
    receipt_path_text = _identity_path(receipt_path)
    receipt = _read_private_bootstrap_receipt(receipt_path)
    receipt_hash = unknown
    if receipt is not None:
        observed_hash = _sha256_private_regular_file(receipt_path)
        if observed_hash is not None:
            receipt_hash = observed_hash

    try:
        source = receipt["repositories"]["asnumpy"]
    except (KeyError, TypeError):
        source = {}
    if type(source) is not dict:
        source = {}
    return {
        "receipt": receipt,
        "bootstrap_receipt_path": receipt_path_text,
        "bootstrap_receipt_sha256": receipt_hash,
        "asnumpy_repository": _identity_repository(source.get("repository")),
        "asnumpy_commit": _identity_git_id(source.get("commit")),
        "asnumpy_tree": _identity_git_id(source.get("tree")),
    }


def _runtime_identity_cann_facts():
    unknown = UNKNOWN_ENVIRONMENT_VALUE
    raw_home = (
        os.getenv("ASCEND_HOME_PATH")
        or os.getenv("ASCEND_TOOLKIT_HOME")
        or os.getenv("CANN_HOME")
    )
    cann_home = _identity_path(
        raw_home, require_existing=True, require_directory=True
    )
    version = None if cann_home == unknown else _read_cann_version(cann_home)
    return cann_home, _identity_token(version)


def _runtime_identity_receipt_interpreter(receipt):
    try:
        value = receipt["verification"]["asnumpy"]["interpreter"]
    except (KeyError, TypeError):
        return UNKNOWN_ENVIRONMENT_VALUE
    return _identity_path(value, require_existing=True)


def _p0_linalg_runtime_identity_matches(module):
    raw_path = getattr(module, "__file__", None)
    if not isinstance(raw_path, str) or not raw_path:
        return False
    init_path = Path(raw_path)
    if init_path.is_symlink() or init_path.name != "__init__.py":
        return False
    parent = init_path.parent
    linalg_dir = parent / "linalg"
    if linalg_dir.is_symlink() or not linalg_dir.is_dir():
        return False
    core_paths = tuple(parent.glob("_core*.so"))
    if len(core_paths) != 1 or core_paths[0].is_symlink():
        return False
    observed = {
        "__init__.py": _sha256_private_regular_file(init_path),
        "array.py": _sha256_private_regular_file(parent / "array.py"),
        "math.py": _sha256_private_regular_file(parent / "math.py"),
        "linalg/direct.py": _sha256_private_regular_file(linalg_dir / "direct.py"),
        "utils.py": _sha256_private_regular_file(parent / "utils.py"),
        "_core": _sha256_private_regular_file(core_paths[0]),
    }
    return observed == _P0_LINALG_RUNTIME_FILE_SHA256


def _p0_move_runtime_identity_matches(module):
    """Bind the copy-elision wheel to Python wrappers and its native module."""

    raw_path = getattr(module, "__file__", None)
    if not isinstance(raw_path, str) or not raw_path:
        return False
    init_path = Path(raw_path)
    if init_path.is_symlink() or init_path.name != "__init__.py":
        return False
    parent = init_path.parent
    linalg_dir = parent / "linalg"
    if linalg_dir.is_symlink() or not linalg_dir.is_dir():
        return False
    core_paths = tuple(parent.glob("_core*.so"))
    if len(core_paths) != 1 or core_paths[0].is_symlink():
        return False
    observed = {
        "__init__.py": _sha256_private_regular_file(init_path),
        "array.py": _sha256_private_regular_file(parent / "array.py"),
        "math.py": _sha256_private_regular_file(parent / "math.py"),
        "linalg/direct.py": _sha256_private_regular_file(linalg_dir / "direct.py"),
        "utils.py": _sha256_private_regular_file(parent / "utils.py"),
        "_core": _sha256_private_regular_file(core_paths[0]),
    }
    return observed == _P0_MOVE_RUNTIME_FILE_SHA256


def _cbrt_runtime_identity_matches(module):
    """Bind the showcase wheel without auto-admitting its custom operator."""

    raw_path = getattr(module, "__file__", None)
    if not isinstance(raw_path, str) or not raw_path:
        return False
    init_path = Path(raw_path)
    if init_path.is_symlink() or init_path.name != "__init__.py":
        return False
    parent = init_path.parent
    core_paths = tuple(parent.glob("_core*.so"))
    if len(core_paths) != 1 or core_paths[0].is_symlink():
        return False
    observed = {
        "__init__.py": _sha256_private_regular_file(init_path),
        "math.py": _sha256_private_regular_file(parent / "math.py"),
        "custom_ops.py": _sha256_private_regular_file(parent / "custom_ops.py"),
        "_custom_ops_registry.py": _sha256_private_regular_file(
            parent / "_custom_ops_registry.py"
        ),
        "utils.py": _sha256_private_regular_file(parent / "utils.py"),
        "_core": _sha256_private_regular_file(core_paths[0]),
    }
    return observed == _CBRT_ASNUMPY_RUNTIME_FILE_SHA256


def _runtime_audit_id(module, version):
    if (
        version == OPS_NN_MATMUL_ASNUMPY_VERSION
        and _ops_nn_matmul_runtime_identity_matches(module)
    ):
        return OPS_NN_MATMUL_ASNUMPY_AUDIT_ID
    if version == AUDITED_ASNUMPY_VERSION:
        # The bundled wheel has the same file-level runtime hashes as the
        # generic audit profile, so receipt-bound component identity must win
        # before the generic hash check.
        if _internal_component_receipt_identity_matches(module):
            return ASNUMPY_INTERNAL_COMPONENT_AUDIT_ID
        if _fixed_audited_runtime_identity_matches(module):
            return ASNUMPY_AUDIT_ID
        if _source_build_receipt_identity_matches(module):
            return ASNUMPY_SOURCE_BUILD_AUDIT_ID
        # Preserve the narrow injection seam used by unit/private backends.
        # Production modules reach one of the two concrete checks above.
        if _audited_runtime_identity_matches(module):
            return ASNUMPY_AUDIT_ID
    if version == P0_ASNUMPY_VERSION:
        if _p0_linalg_runtime_identity_matches(module):
            return P0_LINALG_ASNUMPY_AUDIT_ID
        if _p0_runtime_identity_matches(module):
            return P0_ASNUMPY_AUDIT_ID
    if (
        version == P0_MOVE_ASNUMPY_VERSION
        and _p0_move_runtime_identity_matches(module)
    ):
        return P0_MOVE_ASNUMPY_AUDIT_ID
    if (
        version == CBRT_ASNUMPY_VERSION
        and _cbrt_runtime_identity_matches(module)
    ):
        return CBRT_ASNUMPY_AUDIT_ID
    return "unaudited-asnumpy-version"


def _provenance_profile(name, version, audit_id):
    """Return source-audited provenance independently of runtime availability."""

    if audit_id == OPS_NN_MATMUL_ASNUMPY_AUDIT_ID:
        if name not in OPS_NN_MATMUL_AUDITED_NATIVE_OPS:
            return (
                ExecutionProvenance.UNKNOWN,
                (
                    "the ops-nn MatMul integration runtime is hash-bound, "
                    "but API %r is outside its two-operator case audit" % name
                ),
                ("requires an operator-specific source audit",),
            )
        if name == "matmul":
            return (
                ExecutionProvenance.NATIVE_NPU,
                (
                    "hash-bound AsNumPy public wrapper delegates qualified "
                    "2-D float32 inputs to installed CANN aclnnMatmul; pinned "
                    "ops-nn v9.0.0 is the reviewed source provenance"
                ),
                (
                    "dtype=float32",
                    "rank>=2",
                    "rank<=2",
                    "contiguous=true",
                    "numel>=1",
                ),
            )
        return (
            ExecutionProvenance.NATIVE_NPU,
            "hash-bound case audit: asnumpy.full delegates to an NPU binding",
            (),
        )

    if (
        audit_id == CBRT_ASNUMPY_AUDIT_ID
        and name not in CBRT_ASNUMPY_AUDITED_BASE_OPS
    ):
        return (
            ExecutionProvenance.UNKNOWN,
            (
                "asnumpy cbrt showcase wheel identity is bound, but API %r "
                "requires an operator-specific qualification overlay" % name
            ),
            ("requires an operator-specific qualification",),
        )
    if audit_id == P0_ASNUMPY_AUDIT_ID and name not in P0_AUDITED_NATIVE_OPS:
        return (
            ExecutionProvenance.UNKNOWN,
            (
                "asnumpy P0 wheel identity is bound, but API %r is outside "
                "the limited Poisson core audit" % name
            ),
            ("requires an operator-specific source audit",),
        )
    if (
        audit_id in _P0_CURRENT_LINALG_AUDIT_IDS
        and name not in P0_LINALG_AUDITED_NATIVE_OPS
    ) or (
        audit_id == P0_MOVE_LEGACY_OVERLAY_AUDIT_ID
        and name not in _P0_LEGACY_LINALG_AUDITED_NATIVE_OPS
    ):
        return (
            ExecutionProvenance.UNKNOWN,
            (
                "asnumpy P0 linalg wheel identity is bound, but API %r is "
                "outside the limited audited set" % name
            ),
            ("requires an operator-specific source audit",),
        )
    if audit_id not in {
        ASNUMPY_AUDIT_ID,
        ASNUMPY_INTERNAL_COMPONENT_AUDIT_ID,
        ASNUMPY_SOURCE_BUILD_AUDIT_ID,
        P0_ASNUMPY_AUDIT_ID,
        P0_LINALG_ASNUMPY_AUDIT_ID,
        P0_MOVE_ASNUMPY_AUDIT_ID,
        P0_MOVE_LEGACY_OVERLAY_AUDIT_ID,
        CBRT_ASNUMPY_AUDIT_ID,
    }:
        return (
            ExecutionProvenance.UNKNOWN,
            "asnumpy version/runtime identity %r has no provenance audit" % version,
            ("requires a version-specific source audit",),
        )
    if (
        audit_id
        in {
            ASNUMPY_AUDIT_ID,
            ASNUMPY_INTERNAL_COMPONENT_AUDIT_ID,
            ASNUMPY_SOURCE_BUILD_AUDIT_ID,
        }
        and name in _GENERIC_NESTED_PROVENANCE
    ):
        return _GENERIC_NESTED_PROVENANCE[name]
    if name not in SUPPORTED_OPS:
        return (
            ExecutionProvenance.UNKNOWN,
            "API is absent from the audited coverage snapshot",
            ("requires an operator-specific source audit",),
        )
    if name in _UNAUDITED_OPS:
        return (
            ExecutionProvenance.UNKNOWN,
            (
                "API is in NumPilot's demo coverage snapshot but absent from "
                "asnumpy@1e7177f1 source"
            ),
            ("requires an operator-specific source audit",),
        )
    if (
        audit_id in _P0_LINALG_AUDIT_IDS
        and name in {"outer", "matmul"}
    ):
        return (
            ExecutionProvenance.NATIVE_NPU,
            "asnumpy P0 hash-bound linalg audit: float32 rank-1/2 public path delegates to ACLNN",
            (
                "dtype=float32",
                "rank>=1",
                "rank<=2",
                "contiguous=true",
                "numel>=1",
            ),
        )
    if (
        audit_id in _P0_LINALG_AUDIT_IDS
        and name in _P0_SCOPED_MATH_SIGNATURES
    ):
        return (
            ExecutionProvenance.NATIVE_NPU,
            "asnumpy P0 hash-bound math audit: float32 public path delegates to ACLNN",
            (
                "dtype=float32",
                "rank>=1",
                "contiguous=true",
                "numel>=1",
            ),
        )
    if (
        audit_id
        in {
            ASNUMPY_INTERNAL_COMPONENT_AUDIT_ID,
            ASNUMPY_SOURCE_BUILD_AUDIT_ID,
        }
        and name == "matmul"
    ):
        return (
            ExecutionProvenance.NATIVE_NPU,
            (
                "author-owned AsNumpy static wheel delegates the admitted "
                "2-D float32 contiguous path to ACLNN aclnnMatmul; other "
                "inputs retain the NumPy compatibility fallback"
            ),
            (
                "dtype=float32",
                "rank>=2",
                "rank<=2",
                "contiguous=true",
                "numel>=1",
            ),
        )
    if name in _CPU_FALLBACKS:
        return _CPU_FALLBACKS[name]
    return (
        ExecutionProvenance.NATIVE_NPU,
        (
            "asnumpy cbrt showcase hash-bound base audit: public wrapper "
            "delegates to ACLNN"
            if audit_id == CBRT_ASNUMPY_AUDIT_ID
            else (
                "asnumpy P0 hash-bound source/binary audit: public wrapper "
                "delegates to ACLNN"
                if audit_id == P0_ASNUMPY_AUDIT_ID
                or audit_id in _P0_LINALG_AUDIT_IDS
                else (
                    "asnumpy@1e7177f1 internal-component receipt and "
                    "wheel audit: public wrapper delegates to an NPU binding"
                    if audit_id == ASNUMPY_INTERNAL_COMPONENT_AUDIT_ID
                    else (
                        "asnumpy@1e7177f1 pinned-source receipt and wrapper "
                        "audit: public wrapper delegates to an NPU binding"
                        if audit_id == ASNUMPY_SOURCE_BUILD_AUDIT_ID
                        else "asnumpy@1e7177f1 source audit: public wrapper "
                        "delegates to an NPU binding"
                    )
                )
            )
        ),
        (),
    )


def _signature_variants(name, audit_id):
    if (
        audit_id
        in {
            ASNUMPY_INTERNAL_COMPONENT_AUDIT_ID,
            ASNUMPY_SOURCE_BUILD_AUDIT_ID,
        }
        and name == "matmul"
    ):
        return (OperatorSignature(positional_args=2),)
    if audit_id == OPS_NN_MATMUL_ASNUMPY_AUDIT_ID and name == "matmul":
        return (OperatorSignature(positional_args=2),)
    if (
        audit_id in _P0_LINALG_AUDIT_IDS
        and name in {"outer", "matmul"}
    ):
        return (OperatorSignature(positional_args=2),)
    if (
        audit_id in _P0_LINALG_AUDIT_IDS
        and name in _P0_SCOPED_MATH_SIGNATURES
    ):
        return (
            OperatorSignature(positional_args=_P0_SCOPED_MATH_SIGNATURES[name]),
        )
    return ()


def _overlay_base_with_audit_id(snapshot, names, audit_id):
    """Reproduce one historical base only from a stricter live snapshot."""

    capabilities = {}
    for name in names:
        current = snapshot.operators[name]
        if not current.available:
            # An old publishable base required the API to be callable. Keeping
            # the current failure detail guarantees its hash cannot match.
            capabilities[name] = current
            continue
        provenance, evidence, constraints = _provenance_profile(
            name, snapshot.backend_version, audit_id
        )
        capabilities[name] = OperatorCapability(
            name=name,
            available=True,
            provenance=provenance,
            evidence=evidence,
            constraints=constraints,
            signature_variants=_signature_variants(name, audit_id),
        )
    return CapabilityManifest(
        backend=snapshot.backend,
        backend_version=snapshot.backend_version,
        operators=capabilities,
        environment=dict(snapshot.environment),
        schema_version=snapshot.schema_version,
        audit_id=audit_id,
    )


class NpuBackend(ExecutionBackend):
    name = "npu"

    def __init__(
        self,
        module=None,
        *,
        overlay_config=None,
        baseline_api_config=None,
        tensor_profile_extractor=None,
    ):
        # Injection keeps capability tests independent of CANN/NPU availability.
        self._module = module
        if overlay_config is not None and not isinstance(
            overlay_config, OverlayRuntimeConfig
        ):
            raise TypeError("overlay_config must be OverlayRuntimeConfig")
        self._overlay_config = overlay_config
        if baseline_api_config is not None:
            if not isinstance(baseline_api_config, BaselineApiRuntimeConfig):
                raise TypeError("baseline_api_config must be BaselineApiRuntimeConfig")
            if overlay_config is None:
                raise ValueError("baseline API admission requires an explicit official overlay")
        self._baseline_api_config = baseline_api_config
        if tensor_profile_extractor is not None and not callable(
            tensor_profile_extractor
        ):
            raise TypeError("tensor_profile_extractor must be callable")
        self._tensor_profile_extractor = tensor_profile_extractor

    def _extract_tensor_profile(self, value):
        """Read only metadata from the audited AsNumpy ndarray class."""

        if self._tensor_profile_extractor is not None:
            return self._tensor_profile_extractor(value)
        host_profile = tensor_profile_from_value(value)
        if host_profile is not None:
            return host_profile
        module = self.get_module()
        array_type = getattr(module, "ndarray", None)
        if not isinstance(array_type, type) or type(value) is not array_type:
            return None
        try:
            raw_shape = value.shape
            raw_dtype = value.dtype
            raw_strides = value.strides
            shape = tuple(raw_shape)
            strides = tuple(raw_strides)
        except (AttributeError, TypeError, ValueError):
            return None
        if (
            len(shape) > 64
            or len(shape) != len(strides)
            or any(
                isinstance(dimension, bool)
                or not isinstance(dimension, int)
                or dimension < 0
                or dimension > (2 ** 63 - 1)
                for dimension in shape
            )
            or any(
                isinstance(stride, bool)
                or not isinstance(stride, int)
                or stride < 0
                for stride in strides
            )
        ):
            return None
        try:
            import numpy as np

            dtype = np.dtype(raw_dtype)
            if not dtype.isnative:
                return None
            itemsize = int(dtype.itemsize)
        except (AttributeError, TypeError, ValueError):
            return None
        if itemsize <= 0:
            return None
        contiguous = True
        expected = itemsize
        if shape and not any(dimension == 0 for dimension in shape):
            for dimension, stride in zip(reversed(shape), reversed(strides)):
                if dimension > 1 and stride != expected:
                    contiguous = False
                    break
                expected *= max(dimension, 1)
        return TensorProfile(
            dtype=dtype.name,
            shape=shape,
            c_contiguous=contiguous,
        )

    def get_module(self):
        if self._module is None:
            try:
                import asnumpy
            except ImportError as e:
                raise RuntimeError(
                    "未找到 asnumpy。请先在 CANN 环境下 build/install asnumpy, "
                    "或改用默认 shim 后端(NUMPILOT_BACKEND=shim)。"
                ) from e
            self._module = asnumpy
        return self._module

    def runtime_identity(self):
        """Capture the generic audited runtime identity using read-only probes.

        Unlike execution and capability discovery, this method deliberately
        does not call :meth:`get_module`: locating an installed package,
        hashing its files, parsing the private bootstrap receipt and running
        ``npu-smi info`` do not initialise AsNumpy, CANN, or an NPU context.
        """

        from numpilot.bootstrap import (
            ASNUMPY_COMMIT,
            ASNUMPY_REPOSITORY,
            ASNUMPY_TREE,
        )

        unknown = UNKNOWN_ENVIRONMENT_VALUE
        module = (
            self._module
            if self._module is not None
            else _metadata_only_asnumpy_module()
        )
        try:
            raw_version = getattr(module, "__version__", None)
        except Exception:
            raw_version = None
        version = _identity_token(raw_version)

        if version == OPS_NN_MATMUL_ASNUMPY_VERSION:
            return _ops_nn_matmul_runtime_identity(module)

        file_facts = _runtime_identity_file_facts(module)
        receipt_facts = _runtime_identity_receipt_facts()
        receipt = receipt_facts.pop("receipt")
        cann_home, cann_version = _runtime_identity_cann_facts()
        # Strict identity is hardware-bound even when a caller supplies an
        # apparently valid ASCEND_SOC_VERSION/module attribute.
        soc_model = _read_only_npu_smi_soc_model()
        python_executable = _identity_path(
            sys.executable, require_existing=True
        )
        try:
            python_version = _identity_token(platform.python_version())
        except Exception:
            python_version = unknown

        try:
            audit_id = _identity_token(_runtime_audit_id(module, version))
        except Exception:
            audit_id = unknown
        try:
            source_receipt_matches = _source_build_receipt_identity_matches(module)
        except Exception:
            source_receipt_matches = False
        try:
            component_receipt_matches = (
                _internal_component_receipt_identity_matches(module)
            )
        except Exception:
            component_receipt_matches = False

        issues = set()
        if receipt_facts["asnumpy_repository"] != ASNUMPY_REPOSITORY:
            issues.add("AsNumpy repository does not match pinned source")
        if receipt_facts["asnumpy_commit"] != ASNUMPY_COMMIT:
            issues.add("AsNumpy commit does not match pinned source")
        if receipt_facts["asnumpy_tree"] != ASNUMPY_TREE:
            issues.add("AsNumpy tree does not match pinned source")
        if version != AUDITED_ASNUMPY_VERSION:
            issues.add("AsNumpy version does not match the generic audit")

        expected_wrappers = {
            relative: digest
            for relative, digest in _AUDITED_RUNTIME_FILE_SHA256.items()
            if relative != "_core"
        }
        for relative, expected_hash in expected_wrappers.items():
            if file_facts["wrapper_sha256"].get(relative) != expected_hash:
                issues.add("AsNumpy wrapper hash drift: %s" % relative)
        if file_facts["module_path"] == unknown:
            issues.add("AsNumpy module path is unavailable")
        if file_facts["module_sha256"] != expected_wrappers["__init__.py"]:
            issues.add("AsNumpy module hash does not match the generic audit")
        if file_facts["native_module_path"] == unknown:
            issues.add("AsNumpy native module path is unavailable")
        if file_facts["native_module_sha256"] == unknown:
            issues.add("AsNumpy native module hash is unavailable")

        if receipt is None:
            issues.add("bootstrap receipt is missing or invalid")
        if receipt_facts["bootstrap_receipt_path"] == unknown:
            issues.add("bootstrap receipt path is unavailable")
        if receipt_facts["bootstrap_receipt_sha256"] == unknown:
            issues.add("bootstrap receipt hash is unavailable")
        if not (source_receipt_matches or component_receipt_matches):
            issues.add("bootstrap receipt does not bind the observed runtime")
        if audit_id not in {
            ASNUMPY_SOURCE_BUILD_AUDIT_ID,
            ASNUMPY_INTERNAL_COMPONENT_AUDIT_ID,
        }:
            issues.add("generic AsNumpy audit ID is not receipt-bound")

        expected_cann_home = _identity_path(AUDITED_CANN_HOME)
        if (
            cann_home == unknown
            or expected_cann_home == unknown
            or cann_home != expected_cann_home
        ):
            issues.add("CANN home does not match the audited runtime")
        if cann_version != AUDITED_CANN_VERSION:
            issues.add("CANN version does not match the audited runtime")
        if soc_model != AUDITED_SOC_MODEL:
            issues.add("SoC model does not match the audited runtime")
        if python_version != AUDITED_PYTHON_VERSION:
            issues.add("Python version does not match the audited runtime")
        receipt_interpreter = _runtime_identity_receipt_interpreter(receipt)
        if python_executable == unknown:
            issues.add("Python executable is unavailable")
        if receipt_interpreter == unknown:
            issues.add("bootstrap receipt interpreter is unavailable")
        if python_executable != receipt_interpreter:
            issues.add("Python executable does not match the bootstrap receipt")

        status = "matched" if not issues else "audit_required"
        return RuntimeIdentity(
            asnumpy_repository=receipt_facts["asnumpy_repository"],
            asnumpy_commit=receipt_facts["asnumpy_commit"],
            asnumpy_tree=receipt_facts["asnumpy_tree"],
            asnumpy_version=version,
            wrapper_sha256=file_facts["wrapper_sha256"],
            module_path=file_facts["module_path"],
            module_sha256=file_facts["module_sha256"],
            native_module_path=file_facts["native_module_path"],
            native_module_sha256=file_facts["native_module_sha256"],
            cann_home=cann_home,
            cann_version=cann_version,
            soc_model=soc_model,
            python_executable=python_executable,
            python_version=python_version,
            bootstrap_receipt_path=receipt_facts["bootstrap_receipt_path"],
            bootstrap_receipt_sha256=receipt_facts[
                "bootstrap_receipt_sha256"
            ],
            audit_id=audit_id,
            status=status,
            issues=tuple(sorted(issues)),
        )

    def _raw_capability_manifest(self, required_ops, *, runtime_targets=None):
        module = self.get_module()
        raw_version = getattr(module, "__version__", None)
        version = None if raw_version is None else str(raw_version)
        audit_id = _runtime_audit_id(module, version)
        names = tuple(sorted({
            _canonical_capability_key(name) for name in required_ops
        }))
        capabilities = {}
        for name in names:
            provenance, evidence, constraints = _provenance_profile(
                name, version, audit_id
            )
            signatures = _signature_variants(name, audit_id)
            try:
                api = _resolve_canonical_runtime_api(module, name)
            except Exception as exc:  # asnumpy resolves public APIs lazily
                capabilities[name] = OperatorCapability(
                    name=name,
                    available=False,
                    provenance=provenance,
                    evidence=(
                        "runtime lookup failed: %s: %s; provenance profile: %s"
                        % (type(exc).__name__, exc, evidence)
                    ),
                    constraints=constraints,
                    signature_variants=signatures,
                )
                continue

            available = callable(api)
            if not available:
                capabilities[name] = OperatorCapability(
                    name=name,
                    available=False,
                    provenance=provenance,
                    evidence=(
                        "runtime attribute exists but is not callable; "
                        "provenance profile: %s" % evidence
                    ),
                    constraints=constraints,
                    signature_variants=signatures,
                )
                continue
            capabilities[name] = OperatorCapability(
                name=name,
                available=True,
                provenance=provenance,
                evidence=evidence,
                constraints=constraints,
                signature_variants=signatures,
            )
            if runtime_targets is not None:
                runtime_targets[name] = api

        return CapabilityManifest(
            backend=self.name,
            backend_version=version,
            operators=capabilities,
            environment=_npu_runtime_environment(module),
            audit_id=audit_id,
        )

    def _capability_manifest(
        self,
        required_ops=None,
        *,
        runtime_targets=None,
        runtime_identity=None,
    ):
        """Resolve runtime APIs and atomically revalidate an explicit overlay."""

        if self._overlay_config is None:
            requested = SUPPORTED_OPS if required_ops is None else required_ops
            if runtime_targets is None:
                return self._raw_capability_manifest(requested)
            return self._raw_capability_manifest(
                requested,
                runtime_targets=runtime_targets,
            )

        # Import lazily: qualification depends on the capability contracts, but
        # the backend must remain usable without the optional overlay path.
        from numpilot.gaps.qualification import (
            apply_overlay,
            capability_manifest_sha256,
            load_overlay,
        )
        from numpilot.gaps.official_qualification import is_official_overlay

        config = self._overlay_config
        overlay = load_overlay(config.overlay_path)
        overlay_names = tuple(entry.registry_key for entry in overlay.entries)
        baseline = None
        baseline_names = ()
        if self._baseline_api_config is not None:
            if not is_official_overlay(overlay):
                raise ValueError("baseline API admission requires an official qualification overlay")
            from numpilot.baseline_api_qualification import load_baseline_api_certificate

            baseline = load_baseline_api_certificate(self._baseline_api_config.certificate_path)
            baseline_names = tuple(entry.registry_key for entry in baseline.entries)
            if set(baseline_names) & set(overlay_names):
                raise ValueError("baseline and official overlay operator ownership overlaps")

        requested = (
            set(SUPPORTED_OPS) | set(overlay_names) | set(baseline_names)
            if required_ops is None
            else {str(name) for name in required_ops}
        )
        # Availability, version, module identity and environment must all come
        # from one snapshot.  A second lazy lookup here would create a TOCTOU
        # window in which admitted evidence could overwrite a drifted runtime.
        snapshot_ops = set(requested) | set(overlay_names) | set(baseline_names)
        snapshot_targets = {} if runtime_targets is not None else None
        if runtime_targets is None:
            snapshot = self._raw_capability_manifest(snapshot_ops)
        else:
            snapshot = self._raw_capability_manifest(
                snapshot_ops,
                runtime_targets=snapshot_targets,
            )

        # The overlay identity is deliberately based only on its own entries.
        # This keeps the base hash stable when unrelated static coverage grows.
        overlay_base = CapabilityManifest(
            backend=snapshot.backend,
            backend_version=snapshot.backend_version,
            operators={
                name: snapshot.operators[name]
                for name in overlay_names
            },
            environment=dict(snapshot.environment),
            schema_version=snapshot.schema_version,
            audit_id=snapshot.audit_id,
        )
        if (
            snapshot.audit_id == P0_MOVE_ASNUMPY_AUDIT_ID
            and capability_manifest_sha256(overlay_base)
            != overlay.base_manifest_sha256
        ):
            legacy_base = _overlay_base_with_audit_id(
                snapshot,
                overlay_names,
                P0_MOVE_LEGACY_OVERLAY_AUDIT_ID,
            )
            if (
                capability_manifest_sha256(legacy_base)
                == overlay.base_manifest_sha256
            ):
                overlay_base = legacy_base
        official = is_official_overlay(overlay)
        if official and runtime_identity is None:
            runtime_identity = self.runtime_identity()
        admitted = apply_overlay(
            overlay_base,
            overlay,
            evidence_root=config.evidence_root,
            trusted_producers=config.trusted_producers,
            current_identity=runtime_identity if official else None,
        )

        operators = dict(snapshot.operators)
        for name in overlay_names:
            operators[name] = admitted.operators[name]
        combined = CapabilityManifest(
            backend=snapshot.backend,
            backend_version=snapshot.backend_version,
            operators=operators,
            environment=dict(snapshot.environment),
            schema_version=snapshot.schema_version,
            audit_id=admitted.audit_id,
        )
        if baseline is not None:
            from numpilot.baseline_api_qualification import apply_baseline_api_certificate

            combined = apply_baseline_api_certificate(
                combined, baseline, evidence_root=self._baseline_api_config.evidence_root,
                current_identity=runtime_identity,
            )
        result = CapabilityManifest(
            backend=combined.backend,
            backend_version=combined.backend_version,
            operators={name: combined.operators[name] for name in requested},
            environment=dict(combined.environment),
            schema_version=combined.schema_version,
            audit_id=combined.audit_id,
        )
        if runtime_targets is not None:
            # Evidence validation needs the complete overlay/certificate union,
            # but this execution may use only one of those APIs. Publish only
            # requested callables after all validation succeeds, retaining the
            # exact objects from the single raw snapshot. The runtime guard's
            # strict key-set and callable-identity checks remain unchanged.
            runtime_targets.update({
                name: snapshot_targets[name]
                for name in requested if name in snapshot_targets
            })
        return result

    def capability_manifest(self, required_ops=None, *, runtime_identity=None):
        return self._capability_manifest(
            required_ops,
            runtime_identity=runtime_identity,
        )

    def qualified_overlay_admission(self, runtime_identity=None):
        """Return the exact route admission for one official ACLNN overlay.

        The returned object deliberately carries only public-call facts: the
        qualification identity, the content hash of the overlay file, the live
        runtime identity, and one scoped :class:`RouteEntry` per admitted API.
        Admission re-runs every fail-closed check by rebuilding the capability
        manifest from the live runtime.
        """

        if self._overlay_config is None:
            raise ValueError("no capability overlay is configured")
        from numpilot.capability_profile import (
            ComputationRoute,
            OverlayRouteAdmission,
            RouteEntry,
        )
        from numpilot.gaps.qualification import load_overlay
        from numpilot.gaps.official_qualification import (
            is_official_overlay,
            official_overlay_file_sha256,
        )

        overlay = load_overlay(self._overlay_config.overlay_path)
        if not is_official_overlay(overlay):
            # Custom capability overlays add evidence-backed capabilities but
            # carry no official route admission.  Returning ``None`` keeps the
            # pinned route analysis in force instead of inventing one.
            return None
        identity = (
            runtime_identity
            if runtime_identity is not None
            else self.runtime_identity()
        )
        manifest = self._capability_manifest(
            tuple(entry.registry_key for entry in overlay.entries),
            runtime_identity=identity,
        )
        entries = {}
        for entry in overlay.entries:
            capability = manifest.operators[entry.registry_key]
            entries[entry.registry_key] = RouteEntry(
                canonical_key=entry.registry_key,
                source_api=entry.qualified_api,
                asnumpy_api=entry.asnumpy_api,
                route=ComputationRoute.NATIVE_NPU,
                evidence=capability.evidence,
                constraints=tuple(capability.constraints),
                signature_variants=tuple(capability.signature_variants),
            )
        return OverlayRouteAdmission(
            qualification_id=overlay.qualification_id,
            overlay_sha256=official_overlay_file_sha256(overlay),
            base_manifest_sha256=overlay.base_manifest_sha256,
            audit_id=manifest.audit_id,
            runtime_identity_sha256=identity.identity_sha256,
            entries=entries,
        )

    def qualified_baseline_api_admission(self, runtime_identity=None):
        """Describe the independently validated baseline API certificate."""
        config = self._baseline_api_config
        if config is None:
            return None
        from numpilot.baseline_api_qualification import load_baseline_api_certificate

        certificate = load_baseline_api_certificate(config.certificate_path)
        identity = runtime_identity if runtime_identity is not None else self.runtime_identity()
        names = tuple(sorted(entry.registry_key for entry in certificate.entries))
        # Reuse the atomic raw/official/baseline composition; never trust a
        # certificate descriptor without validating its complete evidence.
        self._capability_manifest(names, runtime_identity=identity)
        digest = _sha256_private_regular_file(Path(config.certificate_path))
        if not _sha256_text(digest):
            raise ValueError("baseline API certificate is unreadable or unsafe")
        return {
            "qualification_id": certificate.qualification_id,
            "certificate_sha256": digest,
            "runtime_identity_sha256": identity.identity_sha256,
            "operators": list(names),
            "qualified_names": sorted(entry.qualified_api for entry in certificate.entries),
        }

    def prepare_profile_guard(self, required_ops):
        """Prepare a runtime session that guards every scoped call pre-dispatch."""

        required = tuple(sorted({
            _canonical_capability_key(name) for name in required_ops
        }))
        runtime_targets = {}
        module = self.get_module()
        if type(self).capability_manifest is NpuBackend.capability_manifest:
            manifest = self._capability_manifest(
                required,
                runtime_targets=runtime_targets,
            )
        else:
            # Compatibility for explicit test/private subclasses that provide
            # their own manifest.  Production NpuBackend uses the atomic path
            # above; custom subclasses should override this method when they
            # also need a stronger target-capture contract.
            manifest = self.capability_manifest(required)
            for name in required:
                try:
                    target = _resolve_canonical_runtime_api(module, name)
                except Exception:
                    continue
                if callable(target):
                    runtime_targets[name] = target
        return RuntimeProfileGuard(
            manifest,
            required,
            tensor_profile_extractor=self._extract_tensor_profile,
            runtime_module=_CanonicalRuntimeModuleView(module),
            runtime_targets=runtime_targets,
        )

    def resolve_profile_capability(
        self,
        qualified_name: str,
        profile: OperatorCallProfile,
    ) -> CapabilityMatch:
        """Resolve one exact canonical source call against the active overlay."""

        resolver = self.prepare_analysis_profile_resolver((qualified_name,))
        return resolver.resolve_profile_capability(qualified_name, profile)

    def prepare_analysis_profile_resolver(self, qualified_names):
        """Create one all-or-nothing overlay snapshot for a complete analysis."""

        names = []
        for qualified_name in qualified_names:
            names.append(
                _canonical_key_from_qualified_source_api(qualified_name)
            )
        return _AnalysisManifestResolver(
            self.capability_manifest(tuple(sorted(set(names))))
        )

    def validate_required_ops(
        self,
        required_ops,
        *,
        strict_npu=True,
        raise_on_error=True,
    ):
        return super().validate_required_ops(
            required_ops,
            strict_npu=strict_npu,
            raise_on_error=raise_on_error,
        )

    def validate_plan(self, plan, *, strict_npu=True, raise_on_error=True):
        return super().validate_plan(
            plan,
            strict_npu=strict_npu,
            raise_on_error=raise_on_error,
        )

    def to_device(self, np_array):
        return self.get_module().ndarray.from_numpy(np_array)

    def to_host(self, device_array):
        return device_array.to_numpy()
