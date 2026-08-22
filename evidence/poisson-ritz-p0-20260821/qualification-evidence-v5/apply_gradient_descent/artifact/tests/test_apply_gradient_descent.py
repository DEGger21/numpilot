"""Tests for asnumpy.apply_gradient_descent.

Stage 16/17 of the apply_gradient_descent registry-invoke operator project.

Two layers of tests:
  1. Pure NumPy oracle tests (always collected, always run) that verify the
     semantic contract of ``apply_gradient_descent_oracle``:
         out = var - alpha * delta
     for the admitted scope: float32, rank >= 1, contiguous inputs.
  2. Runtime/public-API tests against ``asnumpy.apply_gradient_descent``.
     These are guarded: they are skipped cleanly when the ``asnumpy``
     package or the custom operator runtime is unavailable. A missing
     runtime is never reported as a CPU golden-test pass.
"""

import importlib
import unittest

import numpy as np

# ---------------------------------------------------------------------------
# NumPy oracle (authoritative semantics)
# ---------------------------------------------------------------------------


def apply_gradient_descent_oracle(var, alpha, delta):
    """ApplyGradientDescent 的非原地 NumPy oracle。"""
    return np.subtract(var, np.multiply(alpha, delta))


# ---------------------------------------------------------------------------
# Pure NumPy oracle tests (no asnumpy import, always run)
# ---------------------------------------------------------------------------


class TestOracleSemantics(unittest.TestCase):
    """Golden semantic tests against the NumPy oracle only."""

    def setUp(self):
        self.rng = np.random.default_rng(60)  # deterministic

    def test_basic_scalar_alpha(self):
        var = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        alpha = np.float32(0.5)
        delta = np.array([0.5, 1.0, -1.5], dtype=np.float32)
        out = apply_gradient_descent_oracle(var, alpha, delta)
        expected = np.array([1.0 - 0.5 * 0.5,
                             2.0 - 0.5 * 1.0,
                             3.0 - 0.5 * (-1.5)], dtype=np.float32)
        np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)
        self.assertEqual(out.dtype, np.float32)

    def test_rank1_random(self):
        var = self.rng.standard_normal(1024).astype(np.float32)
        alpha = np.float32(0.1)
        delta = self.rng.standard_normal(1024).astype(np.float32)
        out = apply_gradient_descent_oracle(var, alpha, delta)
        np.testing.assert_allclose(
            out, var - np.float32(0.1) * delta, rtol=1e-6, atol=1e-6)

    def test_rank2_broadcast_alpha(self):
        var = self.rng.standard_normal((8, 16)).astype(np.float32)
        alpha = np.array(0.25, dtype=np.float32)
        delta = self.rng.standard_normal((8, 16)).astype(np.float32)
        out = apply_gradient_descent_oracle(var, alpha, delta)
        self.assertEqual(out.shape, var.shape)
        np.testing.assert_allclose(
            out, var - alpha * delta, rtol=1e-6, atol=1e-6)

    def test_rank3(self):
        var = self.rng.standard_normal((2, 3, 4)).astype(np.float32)
        alpha = np.float32(-1.5)
        delta = self.rng.standard_normal((2, 3, 4)).astype(np.float32)
        out = apply_gradient_descent_oracle(var, alpha, delta)
        self.assertEqual(out.shape, (2, 3, 4))
        np.testing.assert_allclose(
            out, var - alpha * delta, rtol=1e-6, atol=1e-6)

    def test_rank4_large(self):
        var = self.rng.standard_normal((2, 4, 8, 32)).astype(np.float32)
        alpha = np.float32(0.01)
        delta = self.rng.standard_normal((2, 4, 8, 32)).astype(np.float32)
        out = apply_gradient_descent_oracle(var, alpha, delta)
        self.assertEqual(out.shape, var.shape)
        np.testing.assert_allclose(
            out, var - alpha * delta, rtol=1e-6, atol=1e-6)

    def test_non_inplace(self):
        """Oracle must not mutate its inputs."""
        var = np.array([1.0, -2.0, 3.0], dtype=np.float32)
        delta = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        var_copy = var.copy()
        delta_copy = delta.copy()
        _ = apply_gradient_descent_oracle(var, np.float32(2.0), delta)
        np.testing.assert_array_equal(var, var_copy)
        np.testing.assert_array_equal(delta, delta_copy)

    def test_zero_alpha_returns_var(self):
        var = self.rng.standard_normal(64).astype(np.float32)
        delta = self.rng.standard_normal(64).astype(np.float32)
        out = apply_gradient_descent_oracle(var, np.float32(0.0), delta)
        np.testing.assert_array_equal(out, var)

    def test_zero_delta_returns_var(self):
        var = self.rng.standard_normal(64).astype(np.float32)
        delta = np.zeros(64, dtype=np.float32)
        out = apply_gradient_descent_oracle(var, np.float32(3.7), delta)
        np.testing.assert_array_equal(out, var)

    def test_extreme_values(self):
        var = np.array([1e20, -1e20, 0.0], dtype=np.float32)
        delta = np.array([1e-20, 1e20, 1.0], dtype=np.float32)
        out = apply_gradient_descent_oracle(var, np.float32(1.0), delta)
        np.testing.assert_allclose(
            out, var - delta, rtol=1e-5, atol=1e-6)

    def test_dtype_float32_preserved(self):
        var = np.ones((5,), dtype=np.float32)
        delta = np.ones((5,), dtype=np.float32)
        out = apply_gradient_descent_oracle(var, np.float32(0.5), delta)
        self.assertEqual(out.dtype, np.float32)


# ---------------------------------------------------------------------------
# Runtime / public-API tests (guarded; never fake a pass)
# ---------------------------------------------------------------------------


def _try_import_asnumpy():
    """Import asnumpy lazily; return module or None."""
    try:
        return importlib.import_module("asnumpy")
    except Exception:  # noqa: BLE001 - any import failure means unavailable
        return None


def _asnumpy_has_apply_gradient_descent(asnp):
    return hasattr(asnp, "apply_gradient_descent")


class TestAsnumpyPublicAPI(unittest.TestCase):
    """Public API tests calling asnumpy.apply_gradient_descent.

    Skipped (not silently passed) when asnumpy or the custom operator
    runtime is unavailable.
    """

    def setUp(self):
        asnp = _try_import_asnumpy()
        if asnp is None:
            self.skipTest("asnumpy package not available; runtime test skipped")
        if not _asnumpy_has_apply_gradient_descent(asnp):
            self.skipTest(
                "asnumpy.apply_gradient_descent not registered; "
                "custom operator runtime not activated; runtime test skipped"
            )
        self.asnp = asnp
        self.rng = np.random.default_rng(61)

    def test_public_api_rank1(self):
        var = self.rng.standard_normal(128).astype(np.float32)
        delta = self.rng.standard_normal(128).astype(np.float32)
        alpha = np.float32(0.1)
        out = self.asnp.apply_gradient_descent(var, alpha, delta)
        out_np = np.asarray(out)
        self.assertEqual(out_np.dtype, np.float32)
        self.assertEqual(out_np.shape, var.shape)
        np.testing.assert_allclose(
            out_np,
            apply_gradient_descent_oracle(var, alpha, delta),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_public_api_rank2(self):
        var = self.rng.standard_normal((16, 32)).astype(np.float32)
        delta = self.rng.standard_normal((16, 32)).astype(np.float32)
        alpha = np.float32(0.5)
        out = self.asnp.apply_gradient_descent(var, alpha, delta)
        out_np = np.asarray(out)
        self.assertEqual(out_np.shape, var.shape)
        np.testing.assert_allclose(
            out_np,
            apply_gradient_descent_oracle(var, alpha, delta),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_public_api_non_inplace(self):
        var = np.array([1.0, 2.0, 4.0], dtype=np.float32)
        delta = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        var_copy = var.copy()
        _ = self.asnp.apply_gradient_descent(var, np.float32(1.0), delta)
        np.testing.assert_array_equal(var, var_copy)


if __name__ == "__main__":
    unittest.main()