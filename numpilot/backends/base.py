"""执行后端端口:让验证阶段在不同后端(shim / 真实 NPU)运行迁移代码。"""
import os
from abc import ABC, abstractmethod

from numpilot.capabilities import (
    BackendCapabilityError,
    CapabilityManifest,
    ExecutionProvenance,
    OperatorCapability,
    detect_runtime_environment,
    validate_manifest,
)
from numpilot.coverage import SUPPORTED_OPS


class ExecutionBackend(ABC):
    name: str = "base"

    @abstractmethod
    def get_module(self):
        """返回一个可作为 `asnumpy` 使用的模块对象。"""

    @abstractmethod
    def to_device(self, np_array):
        ...

    @abstractmethod
    def to_host(self, device_array):
        ...

    def capability_manifest(self, required_ops=None):
        """Build a runtime capability snapshot for this concrete module.

        Generic backends can prove only that an API is callable.  Subclasses such
        as :class:`NpuBackend` may attach a stronger, version-specific provenance
        audit.  Keeping this method concrete preserves compatibility for existing
        third-party/shim backends.
        """

        module = self.get_module()
        requested = SUPPORTED_OPS if required_ops is None else required_ops
        names = tuple(sorted({str(name) for name in requested}))
        provenance = (
            ExecutionProvenance.SHIM_CPU
            if self.name == "shim"
            else ExecutionProvenance.UNKNOWN
        )
        capabilities = {}
        for name in names:
            try:
                api = getattr(module, name)
            except Exception as exc:  # lazy modules may fail while resolving an API
                capabilities[name] = OperatorCapability(
                    name=name,
                    available=False,
                    evidence="runtime lookup failed: %s: %s"
                    % (type(exc).__name__, exc),
                )
                continue
            available = callable(api)
            capabilities[name] = OperatorCapability(
                name=name,
                available=available,
                provenance=provenance if available else ExecutionProvenance.UNKNOWN,
                evidence=(
                    "runtime callable on %s backend" % self.name
                    if available
                    else "runtime attribute exists but is not callable"
                ),
            )
        version = getattr(module, "__version__", None)
        return CapabilityManifest(
            backend=self.name,
            backend_version=None if version is None else str(version),
            operators=capabilities,
            environment=detect_runtime_environment(module),
            audit_id="runtime-%s-v1" % self.name,
        )

    def validate_required_ops(
        self,
        required_ops,
        *,
        strict_npu=False,
        raise_on_error=True,
    ):
        """Validate runtime availability and optional strict-NPU provenance."""

        required_ops = tuple(str(name) for name in required_ops)
        manifest = self.capability_manifest(required_ops)
        validation = validate_manifest(
            manifest,
            required_ops,
            strict_npu=strict_npu,
        )
        if raise_on_error and not validation.ok:
            raise BackendCapabilityError(validation)
        return validation

    def validate_plan(self, plan, *, strict_npu=False, raise_on_error=True):
        """Validate only APIs that an offload plan places on the NPU island."""

        placement = getattr(plan, "placement", None)
        if not isinstance(placement, dict):
            raise TypeError("offload plan must expose a dict 'placement' field")
        required_ops = {
            str(name)
            for name, target in placement.items()
            if str(target).upper() == "NPU"
        }
        return self.validate_required_ops(
            required_ops,
            strict_npu=strict_npu,
            raise_on_error=raise_on_error,
        )


def select_backend(name=None):
    name = name or os.getenv("NUMPILOT_BACKEND", "shim")
    if name == "shim":
        from numpilot.backends.shim import ShimBackend
        return ShimBackend()
    if name == "npu":
        from numpilot.backends.npu import NpuBackend
        return NpuBackend()
    raise ValueError(f"未知后端: {name!r}(可选 shim | npu)")
