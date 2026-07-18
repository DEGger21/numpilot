"""真实 asnumpy(NPU)后端适配器。需已 build/install asnumpy。"""
from numpilot.capabilities import (
    CapabilityManifest,
    ExecutionProvenance,
    OperatorCapability,
    detect_runtime_environment,
)
from numpilot.backends.base import ExecutionBackend
from numpilot.coverage import SUPPORTED_OPS


# This is deliberately an exact version match.  A different wheel may change a
# Python wrapper without changing whether its public name exists, so version
# drift must be re-audited before it can make a strict-NPU performance claim.
AUDITED_ASNUMPY_VERSION = "0.3.1.dev4+g1929a3ba4"
ASNUMPY_AUDIT_ID = "asnumpy-0.3.1.dev4+g1929a3ba4-source-1929a3ba4-v1"


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


def _provenance_profile(name, version):
    """Return source-audited provenance independently of runtime availability."""

    if version != AUDITED_ASNUMPY_VERSION:
        return (
            ExecutionProvenance.UNKNOWN,
            "asnumpy version %r has no provenance audit" % version,
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
    if name in _CPU_FALLBACKS:
        return _CPU_FALLBACKS[name]
    return (
        ExecutionProvenance.NATIVE_NPU,
        "asnumpy@1929a3ba4 source audit: public wrapper delegates to an NPU binding",
        (),
    )


class NpuBackend(ExecutionBackend):
    name = "npu"

    def __init__(self, module=None):
        # Injection keeps capability tests independent of CANN/NPU availability.
        self._module = module

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

    def capability_manifest(self, required_ops=None):
        """Resolve APIs at runtime and apply the audited wheel provenance."""

        module = self.get_module()
        raw_version = getattr(module, "__version__", None)
        version = None if raw_version is None else str(raw_version)
        audited = version == AUDITED_ASNUMPY_VERSION
        requested = SUPPORTED_OPS if required_ops is None else required_ops
        names = tuple(sorted({str(name) for name in requested}))
        capabilities = {}
        for name in names:
            provenance, evidence, constraints = _provenance_profile(name, version)
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
                )
                continue
            capabilities[name] = OperatorCapability(
                name=name,
                available=True,
                provenance=provenance,
                evidence=evidence,
                constraints=constraints,
            )

        return CapabilityManifest(
            backend=self.name,
            backend_version=version,
            operators=capabilities,
            environment=detect_runtime_environment(module),
            audit_id=ASNUMPY_AUDIT_ID if audited else "unaudited-asnumpy-version",
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
