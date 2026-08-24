import numpy as np
import pytest

from numpilot.backends.base import select_backend
from numpilot.backends.shim import ShimBackend
from numpilot.capabilities import ExecutionProvenance
from numpilot.contracts import OffloadPlan
from numpilot.coverage import SUPPORTED_OPS


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


def test_shim_amax_matches_numpy_with_axis():
    ap = ShimBackend().get_module()
    x = np.array([[1.0, 5.0, 2.0], [4.0, 3.0, 6.0]], dtype=np.float32)
    xd = ap.ndarray.from_numpy(x)

    out = ap.amax(xd)
    assert isinstance(out, float)
    assert np.allclose(out, np.amax(x))

    out_axis = ap.amax(xd, axis=0)
    assert isinstance(out_axis, ap.ndarray)
    assert np.allclose(out_axis.to_numpy(), np.amax(x, axis=0))


def test_shim_linspace_matches_asnumpy_signature():
    ap = ShimBackend().get_module()

    out = ap.linspace(0.0, 1.0, 5)
    assert isinstance(out, ap.ndarray)
    assert np.allclose(out.to_numpy(), np.linspace(0.0, 1.0, num=5))

    out_steps = ap.linspace(2.0, 3.0, steps=4)
    assert isinstance(out_steps, ap.ndarray)
    assert np.allclose(out_steps.to_numpy(), np.linspace(2.0, 3.0, num=4))

    out_default = ap.linspace(0.0, 1.0)
    assert out_default.to_numpy().shape == (50,)


def test_shim_linspace_dtype_is_applied():
    ap = ShimBackend().get_module()

    out = ap.linspace(0.0, 1.0, 10, dtype=np.float32)
    assert out.to_numpy().dtype == np.float32
    assert np.allclose(
        out.to_numpy(), np.linspace(0.0, 1.0, num=10).astype(np.float32)
    )


def test_shim_linspace_rejects_numpy_only_keywords():
    ap = ShimBackend().get_module()

    with pytest.raises(TypeError):
        ap.linspace(0.0, 1.0, num=4)

    with pytest.raises(TypeError):
        ap.linspace(0.0, 1.0, 4, endpoint=False)


def test_shim_migrated_apis_satisfy_non_strict_validation():
    backend = ShimBackend()
    plan = OffloadPlan(
        placement={"amax": "NPU", "linspace": "NPU"}, threshold_elements=1
    )

    assert backend.validate_plan(plan, strict_npu=False).ok


def test_shim_exposes_every_static_asnumpy_api_as_callable():
    module = ShimBackend().get_module()

    assert {
        name for name in SUPPORTED_OPS
        if not callable(getattr(module, name, None))
    } == set()


def test_shim_extended_surface_matches_numpy_semantics():
    ap = ShimBackend().get_module()
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    xd = ap.ndarray.from_numpy(x)

    assert np.array_equal(
        ap.full_like(xd, 7.0, dtype=np.float32).to_numpy(),
        np.full_like(x, 7.0, dtype=np.float32),
    )
    assert np.array_equal(
        ap.cumsum(xd, axis=1).to_numpy(), np.cumsum(x, axis=1)
    )
    assert np.array_equal(
        ap.all(xd, axis=(1,)).to_numpy(), np.all(x, axis=(1,))
    )
    assert np.allclose(
        ap.einsum("ij,ij->i", xd, xd).to_numpy(),
        np.einsum("ij,ij->i", x, x),
    )


def test_shim_ndarray_exposes_asnumpy_read_only_metadata():
    ap = ShimBackend().get_module()
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    xd = ap.add(ap.ndarray.from_numpy(x), ap.ndarray.from_numpy(x))
    expected = np.add(x, x)

    assert xd.shape == expected.shape
    assert xd.dtype == expected.dtype
    assert xd.ndim == expected.ndim
    assert xd.itemsize == expected.itemsize
    assert xd.nbytes == expected.nbytes
    assert xd.strides == expected.strides

    with pytest.raises(AttributeError):
        xd.shape = (6,)


@pytest.mark.parametrize(
    "call",
    [
        lambda ap, x: ap.zeros((2,)),
        lambda ap, x: ap.empty_like(x),
        lambda ap, x: ap.all(x, axis=0),
        lambda ap, x: ap.sum(x, axis=(0, 1)),
        lambda ap, x: ap.cumsum(x),
        lambda ap, x: ap.cross(x, x),
        lambda ap, x: ap.sort(x, axis=None),
        lambda ap, x: ap.einsum(x, [0]),
        lambda ap, x: ap.add(x, 2.0),
        lambda ap, x: ap.greater(2.0, x),
    ],
)
def test_shim_exposes_real_asnumpy_contract_failures(call):
    ap = ShimBackend().get_module()
    x = ap.ndarray.from_numpy(np.ones((2, 2), dtype=np.float32))

    with pytest.raises((TypeError, ValueError)):
        call(ap, x)


def test_shim_supports_audited_axis_and_scalar_operand_forms():
    ap = ShimBackend().get_module()
    source = np.array([[1.0, 0.0], [2.0, 3.0]], dtype=np.float32)
    x = ap.ndarray.from_numpy(source)

    assert np.array_equal(
        ap.all(x, axis=(0,)).to_numpy(), np.all(source, axis=(0,))
    )
    assert np.array_equal(
        ap.sum(x, axis=0).to_numpy(), np.sum(source, axis=0)
    )
    assert np.array_equal(
        ap.cumsum(x, axis=0).to_numpy(), np.cumsum(source, axis=0)
    )
    assert np.array_equal(
        ap.greater(x, 1.5).to_numpy(), np.greater(source, 1.5)
    )
    assert np.array_equal(
        ap.power(x, 2).to_numpy(), np.power(source, 2)
    )
