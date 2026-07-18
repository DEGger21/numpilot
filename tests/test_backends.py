import numpy as np

from numpilot.backends.base import select_backend
from numpilot.backends.shim import ShimBackend
from numpilot.capabilities import ExecutionProvenance
from numpilot.contracts import OffloadPlan


def test_default_backend_is_shim():
    b = select_backend()
    assert b.name == "shim"


def test_shim_module_matmul_matches_numpy():
    ap = ShimBackend().get_module()
    x = np.ones((4, 4), dtype=np.float32)
    xd = ap.ndarray.from_numpy(x)
    out = ap.matmul(xd, xd).to_numpy()
    assert np.allclose(out, np.matmul(x, x))


def test_shim_relu_and_softmax():
    ap = ShimBackend().get_module()
    x = np.array([-1.0, 0.0, 2.0], dtype=np.float32)
    xd = ap.ndarray.from_numpy(x)
    assert np.allclose(ap.relu(xd).to_numpy(), np.maximum(x, 0))
    sm = ap.softmax(ap.ndarray.from_numpy(x)).to_numpy()
    assert np.allclose(sm.sum(), 1.0, atol=1e-5)


def test_shim_capability_validation_preserves_availability_mode():
    backend = ShimBackend()
    plan = OffloadPlan(placement={"matmul": "NPU"}, threshold_elements=1)

    compatible = backend.validate_plan(plan, strict_npu=False)
    strict = backend.validate_plan(
        plan, strict_npu=True, raise_on_error=False
    )

    assert compatible.ok
    assert not strict.ok
    assert strict.non_strict == ("matmul",)
    assert strict.capabilities["matmul"].provenance is ExecutionProvenance.SHIM_CPU
