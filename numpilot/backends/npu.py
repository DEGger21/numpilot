"""真实 asnumpy(NPU)后端适配器。需已 build/install asnumpy。"""
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import stat
from typing import Tuple

from numpilot.capabilities import (
    CapabilityMatch,
    CapabilityManifest,
    ExecutionProvenance,
    OperatorCallProfile,
    OperatorCapability,
    OperatorSignature,
    RuntimeProfileGuard,
    TensorProfile,
    detect_runtime_environment,
    match_capability_profile,
    tensor_profile_from_value,
)
from numpilot.backends.base import ExecutionBackend
from numpilot.coverage import SUPPORTED_OPS


# This is deliberately an exact version match.  A different wheel may change a
# Python wrapper without changing whether its public name exists, so version
# drift must be re-audited before it can make a strict-NPU performance claim.
AUDITED_ASNUMPY_VERSION = "0.3.1.dev4+g1929a3ba4"
ASNUMPY_AUDIT_ID = "asnumpy-0.3.1.dev4+g1929a3ba4-source-1929a3ba4-v1"

# Competition P0 wheel.  This is deliberately a narrow source/binary audit,
# not a declaration that every API in the locally built wheel is strict NPU.
P0_ASNUMPY_VERSION = "0.1.dev106+g21368abf9.d20260821"
P0_ASNUMPY_AUDIT_ID = (
    "asnumpy-0.1.dev106-g21368abf9-poisson-core-"
    "init4f4e8083-math62e0ba09-core964c17f3-v1"
)
P0_AUDITED_NATIVE_OPS = frozenset({"add", "multiply", "clip"})
_P0_RUNTIME_FILE_SHA256 = {
    "__init__.py": "4f4e8083755dde87bd6f0dc8c765da71984cb1346c97e8b6dbb97d714be34c04",
    "math.py": "62e0ba09576ead5d3b44f6b0b33311e849efd47ab0bb11f7875c0a43f499085c",
    "_core": "964c17f3c910f54992fa24f086493481dd471d060f523f2e124cb8e8d6c373f3",
}
P0_LINALG_ASNUMPY_AUDIT_ID = (
    "asnumpy-0.1.dev106-g21368abf9-poisson-linalg-"
    "init4f4e8083-math62e0ba09-direct4479c588-cored3ccb39f-v1"
)
P0_LINALG_AUDITED_NATIVE_OPS = frozenset(
    {
        "add",
        "clip",
        "divide",
        "exp",
        "matmul",
        "multiply",
        "negative",
        "outer",
        "sqrt",
        "subtract",
    }
)
P0_MOVE_ASNUMPY_VERSION = "0.1.dev106+g21368abf9.d20260822"
P0_MOVE_ASNUMPY_AUDIT_ID = (
    "asnumpy-0.1.dev106-g21368abf9-poisson-move-"
    "init4f4e8083-math62e0ba09-direct4479c588-"
    "utils1b298145-core4465ce6e-v1"
)
_P0_SCOPED_MATH_SIGNATURES = {
    "divide": 2,
    "exp": 1,
    "negative": 1,
    "sqrt": 1,
    "subtract": 2,
}
_P0_LINALG_RUNTIME_FILE_SHA256 = {
    "__init__.py": "4f4e8083755dde87bd6f0dc8c765da71984cb1346c97e8b6dbb97d714be34c04",
    "math.py": "62e0ba09576ead5d3b44f6b0b33311e849efd47ab0bb11f7875c0a43f499085c",
    "linalg/direct.py": "4479c588c57705e229611a44844b6c7497deb6d97a7f067155f645b1f2913dbf",
    "_core": "d3ccb39f23a24611d4e5f0f34605fdf6825c74a11ef8a1aa4dac498a0ff38821",
}
_P0_MOVE_RUNTIME_FILE_SHA256 = {
    "__init__.py": "4f4e8083755dde87bd6f0dc8c765da71984cb1346c97e8b6dbb97d714be34c04",
    "math.py": "62e0ba09576ead5d3b44f6b0b33311e849efd47ab0bb11f7875c0a43f499085c",
    "linalg/direct.py": "4479c588c57705e229611a44844b6c7497deb6d97a7f067155f645b1f2913dbf",
    "utils.py": "1b2981456b2e4258b2f3f9611a07c98dc5a79acb8d67d35e3f2595af13911435",
    "_core": "4465ce6e0589eb000d450a49c15488a4d2d9f211fa7867646992c56e7c040747",
}
_MAX_AUDITED_RUNTIME_FILE_BYTES = 64 * 1024 * 1024


_CPU_FALLBACKS = {
    "inner": (
        ExecutionProvenance.CPU_FALLBACK,
        "asnumpy@1929a3ba4: linalg/direct.py calls numpy.inner on host arrays",
        ("all inputs execute on CPU",),
    ),
    "matmul": (
        ExecutionProvenance.CPU_FALLBACK,
        "asnumpy@1929a3ba4: linalg/direct.py calls numpy.matmul on host arrays",
        ("all inputs execute on CPU",),
    ),
    "outer": (
        ExecutionProvenance.CPU_FALLBACK,
        "asnumpy@1929a3ba4: linalg/direct.py calls numpy.outer on host arrays",
        ("all inputs execute on CPU",),
    ),
    "dot": (
        ExecutionProvenance.CONDITIONAL_CPU_FALLBACK,
        "asnumpy@1929a3ba4: linalg/direct.py inspects host data and uses numpy.dot for float64",
        (
            "device inputs are copied to host for dtype inspection",
            "float64 input executes on CPU",
        ),
    ),
    "vdot": (
        ExecutionProvenance.CONDITIONAL_CPU_FALLBACK,
        "asnumpy@1929a3ba4: linalg/direct.py inspects host data and uses numpy.vdot for float64",
        (
            "device inputs are copied to host for dtype inspection",
            "float64 input executes on CPU",
        ),
    ),
    "floor": (
        ExecutionProvenance.CONDITIONAL_CPU_FALLBACK,
        "asnumpy@1929a3ba4: math.py inspects host data and uses numpy.floor for integer/bool",
        (
            "dtype=None copies device input to host for dtype inspection",
            "integer or bool input with dtype=None executes on CPU",
        ),
    ),
    "signbit": (
        ExecutionProvenance.HOST_ASSISTED,
        "asnumpy@1929a3ba4: math.py copies input to host to patch IEEE -0.0",
        ("always performs host-side negative-zero detection",),
    ),
}


# ``coverage.py`` contains a demo snapshot that also lists ``angle``, but the
# audited 1929a3ba4 public mapping/source does not contain that API.  If a wheel
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
class _AnalysisManifestResolver:
    """Immutable capability snapshot used for every callsite in one analysis."""

    manifest: CapabilityManifest

    def resolve_profile_capability(
        self,
        qualified_name: str,
        profile: OperatorCallProfile,
    ) -> CapabilityMatch:
        if (
            not isinstance(qualified_name, str)
            or qualified_name.split(".", 1)[0] not in {"numpy", "operator"}
            or qualified_name.count(".") != 1
        ):
            raise ValueError("profile capability requires a top-level source API")
        if not isinstance(profile, OperatorCallProfile):
            raise TypeError("profile must be OperatorCallProfile")
        name = qualified_name.split(".", 1)[1]
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
        "_core": _sha256_private_regular_file(core_paths[0]),
    }
    return observed == _P0_RUNTIME_FILE_SHA256


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
        "math.py": _sha256_private_regular_file(parent / "math.py"),
        "linalg/direct.py": _sha256_private_regular_file(linalg_dir / "direct.py"),
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
        "math.py": _sha256_private_regular_file(parent / "math.py"),
        "linalg/direct.py": _sha256_private_regular_file(linalg_dir / "direct.py"),
        "utils.py": _sha256_private_regular_file(parent / "utils.py"),
        "_core": _sha256_private_regular_file(core_paths[0]),
    }
    return observed == _P0_MOVE_RUNTIME_FILE_SHA256


def _runtime_audit_id(module, version):
    if version == AUDITED_ASNUMPY_VERSION:
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
    return "unaudited-asnumpy-version"


def _provenance_profile(name, version, audit_id):
    """Return source-audited provenance independently of runtime availability."""

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
        audit_id in {P0_LINALG_ASNUMPY_AUDIT_ID, P0_MOVE_ASNUMPY_AUDIT_ID}
        and name not in P0_LINALG_AUDITED_NATIVE_OPS
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
        P0_ASNUMPY_AUDIT_ID,
        P0_LINALG_ASNUMPY_AUDIT_ID,
        P0_MOVE_ASNUMPY_AUDIT_ID,
    }:
        return (
            ExecutionProvenance.UNKNOWN,
            "asnumpy version/runtime identity %r has no provenance audit" % version,
            ("requires a version-specific source audit",),
        )
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
                "asnumpy@1929a3ba4 source"
            ),
            ("requires an operator-specific source audit",),
        )
    if (
        audit_id in {P0_LINALG_ASNUMPY_AUDIT_ID, P0_MOVE_ASNUMPY_AUDIT_ID}
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
        audit_id in {P0_LINALG_ASNUMPY_AUDIT_ID, P0_MOVE_ASNUMPY_AUDIT_ID}
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
    if name in _CPU_FALLBACKS:
        return _CPU_FALLBACKS[name]
    return (
        ExecutionProvenance.NATIVE_NPU,
        (
            "asnumpy P0 hash-bound source/binary audit: public wrapper delegates "
            "to ACLNN"
            if audit_id == P0_ASNUMPY_AUDIT_ID
            or audit_id in {
                P0_LINALG_ASNUMPY_AUDIT_ID,
                P0_MOVE_ASNUMPY_AUDIT_ID,
            }
            else "asnumpy@1929a3ba4 source audit: public wrapper delegates to an NPU binding"
        ),
        (),
    )


def _signature_variants(name, audit_id):
    if (
        audit_id in {P0_LINALG_ASNUMPY_AUDIT_ID, P0_MOVE_ASNUMPY_AUDIT_ID}
        and name in {"outer", "matmul"}
    ):
        return (OperatorSignature(positional_args=2),)
    if (
        audit_id in {P0_LINALG_ASNUMPY_AUDIT_ID, P0_MOVE_ASNUMPY_AUDIT_ID}
        and name in _P0_SCOPED_MATH_SIGNATURES
    ):
        return (
            OperatorSignature(positional_args=_P0_SCOPED_MATH_SIGNATURES[name]),
        )
    return ()


class NpuBackend(ExecutionBackend):
    name = "npu"

    def __init__(
        self,
        module=None,
        *,
        overlay_config=None,
        tensor_profile_extractor=None,
    ):
        # Injection keeps capability tests independent of CANN/NPU availability.
        self._module = module
        if overlay_config is not None and not isinstance(
            overlay_config, OverlayRuntimeConfig
        ):
            raise TypeError("overlay_config must be OverlayRuntimeConfig")
        self._overlay_config = overlay_config
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

    def _raw_capability_manifest(self, required_ops, *, runtime_targets=None):
        module = self.get_module()
        raw_version = getattr(module, "__version__", None)
        version = None if raw_version is None else str(raw_version)
        audit_id = _runtime_audit_id(module, version)
        names = tuple(sorted({str(name) for name in required_ops}))
        capabilities = {}
        for name in names:
            provenance, evidence, constraints = _provenance_profile(
                name, version, audit_id
            )
            signatures = _signature_variants(name, audit_id)
            try:
                api = getattr(module, name)
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
            environment=detect_runtime_environment(module),
            audit_id=audit_id,
        )

    def _capability_manifest(self, required_ops=None, *, runtime_targets=None):
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
        from numpilot.gaps.qualification import apply_overlay, load_overlay

        config = self._overlay_config
        overlay = load_overlay(config.overlay_path)
        overlay_names = tuple(entry.registry_key for entry in overlay.entries)

        requested = (
            set(SUPPORTED_OPS) | set(overlay_names)
            if required_ops is None
            else {str(name) for name in required_ops}
        )
        # Availability, version, module identity and environment must all come
        # from one snapshot.  A second lazy lookup here would create a TOCTOU
        # window in which admitted evidence could overwrite a drifted runtime.
        snapshot_ops = set(requested) | set(overlay_names)
        if runtime_targets is None:
            snapshot = self._raw_capability_manifest(snapshot_ops)
        else:
            snapshot = self._raw_capability_manifest(
                snapshot_ops,
                runtime_targets=runtime_targets,
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
        admitted = apply_overlay(
            overlay_base,
            overlay,
            evidence_root=config.evidence_root,
            trusted_producers=config.trusted_producers,
        )

        operators = {
            name: snapshot.operators[name]
            for name in requested
        }
        for name in set(requested) & set(overlay_names):
            operators[name] = admitted.operators[name]
        return CapabilityManifest(
            backend=snapshot.backend,
            backend_version=snapshot.backend_version,
            operators=operators,
            environment=dict(snapshot.environment),
            schema_version=snapshot.schema_version,
            audit_id=admitted.audit_id,
        )

    def capability_manifest(self, required_ops=None):
        return self._capability_manifest(required_ops)

    def prepare_profile_guard(self, required_ops):
        """Prepare a runtime session that guards every scoped call pre-dispatch."""

        required = tuple(sorted({str(name) for name in required_ops}))
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
                    target = getattr(module, name)
                except Exception:
                    continue
                if callable(target):
                    runtime_targets[name] = target
        return RuntimeProfileGuard(
            manifest,
            required,
            tensor_profile_extractor=self._extract_tensor_profile,
            runtime_module=module,
            runtime_targets=runtime_targets,
        )

    def resolve_profile_capability(
        self,
        qualified_name: str,
        profile: OperatorCallProfile,
    ) -> CapabilityMatch:
        """Resolve one exact top-level source call against the active overlay."""

        resolver = self.prepare_analysis_profile_resolver((qualified_name,))
        return resolver.resolve_profile_capability(qualified_name, profile)

    def prepare_analysis_profile_resolver(self, qualified_names):
        """Create one all-or-nothing overlay snapshot for a complete analysis."""

        names = []
        for qualified_name in qualified_names:
            if (
                not isinstance(qualified_name, str)
                or qualified_name.split(".", 1)[0] not in {"numpy", "operator"}
                or qualified_name.count(".") != 1
            ):
                raise ValueError(
                    "profile capability requires top-level source APIs"
                )
            names.append(qualified_name.split(".", 1)[1])
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
