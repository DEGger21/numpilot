import numpy as np
from numpilot.backends.base import select_backend
from numpilot.backends.shim import ShimBackend

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
