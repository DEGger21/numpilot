"""Tests for smooth_l1_loss (mean reduction, beta=1, float32).

Pure NumPy oracle tests always run. Runtime tests calling the public
``asnumpy.smooth_l1_loss`` API are guarded and skipped when the asnumpy
package or the custom operator runtime is unavailable.
"""

import sys

import numpy as np

import pytest


def smooth_l1_loss_v2_mean_beta1(predict, label):
    """SmoothL1V2 fixed mean/beta=1 NumPy oracle."""

    error = np.subtract(predict, label)
    absolute = np.abs(error)
    bounded = np.minimum(absolute, np.float32(1.0))
    quadratic = np.divide(
        np.multiply(bounded, bounded),
        np.float32(2.0),
    )
    linear = np.subtract(absolute, bounded)
    return np.mean(np.add(quadratic, linear))


def _make_inputs(shape, seed=0):
    rng = np.random.default_rng(seed)
    predict = rng.standard_normal(shape).astype(np.float32)
    label = rng.standard_normal(shape).astype(np.float32)
    return predict, label


def asnumpy_available():
    """Return the asnumpy module if importable and the op registered, else None."""
    try:
        import asnumpy  # noqa: F401  (deferred import, never at module scope)
    except Exception:
        return None
    try:
        import asnumpy as mod

        if not hasattr(mod, "smooth_l1_loss"):
            return None
        return mod
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Pure NumPy oracle tests (always run, no runtime dependency)
# ---------------------------------------------------------------------------


def test_oracle_zero_error():
    x = np.full((4, 5), 2.5, dtype=np.float32)
    y = np.full((4, 5), 2.5, dtype=np.float32)
    expected = np.float32(0.0)
    got = smooth_l1_loss_v2_mean_beta1(x, y)
    assert np.allclose(got, expected, rtol=1e-6, atol=1e-7)


def test_oracle_small_error_quadratic_branch():
    # |error| < 1 everywhere -> pure quadratic branch: mean(err^2 / 2)
    predict, label = _make_inputs((8,), seed=1)
    predict = np.clip(predict, -0.4, 0.4).astype(np.float32)
    label = np.clip(label, -0.4, 0.4).astype(np.float32)
    err = np.subtract(predict, label)
    expected = np.mean(np.multiply(err, err) / np.float32(2.0))
    got = smooth_l1_loss_v2_mean_beta1(predict, label)
    assert np.allclose(got, expected, rtol=1e-6, atol=1e-7)


def test_oracle_large_error_linear_branch():
    # |error| >= 1 everywhere -> pure linear branch: mean(|err| - 0.5)
    predict = np.array([10.0, -10.0, 5.0, -5.0], dtype=np.float32)
    label = np.zeros(4, dtype=np.float32)
    expected = np.mean(np.abs(np.subtract(predict, label)) - np.float32(0.5))
    got = smooth_l1_loss_v2_mean_beta1(predict, label)
    assert np.allclose(got, expected, rtol=1e-6, atol=1e-7)


def test_oracle_mixed_branches():
    predict, label = _make_inputs((16, 8), seed=2)
    # Force some elements into each branch.
    predict = predict * np.float32(3.0)
    err = np.abs(np.subtract(predict, label))
    assert np.any(err < 1.0) and np.any(err >= 1.0)
    got = smooth_l1_loss_v2_mean_beta1(predict, label)
    # Independent reference implementation.
    abs_err = np.abs(predict - label)
    ref = np.mean(np.where(
        abs_err < 1.0,
        (abs_err * abs_err) / 2.0,
        abs_err - 0.5,
    ).astype(np.float32))
    assert np.allclose(got, ref, rtol=1e-5, atol=1e-6)


def test_oracle_boundary_at_one():
    # |error| == 1 contributes 0.5; |error| = 2 contributes 1.5; 0 contributes 0.
    # errors [1, -1, 2, 0] -> losses [0.5, 0.5, 1.5, 0.0] -> mean 0.625
    predict = np.array([1.0, -1.0, 2.0, 0.0], dtype=np.float32)
    label = np.zeros(4, dtype=np.float32)
    got = smooth_l1_loss_v2_mean_beta1(predict, label)
    assert np.allclose(got, np.float32(0.625), rtol=1e-6, atol=1e-7)


def test_oracle_rank1_minimum():
    # |error| = 1.0 exactly -> loss = 1.0 - 0.5 = 0.5
    predict = np.array([0.5], dtype=np.float32)
    label = np.array([-0.5], dtype=np.float32)
    expected = np.float32(0.5)
    got = smooth_l1_loss_v2_mean_beta1(predict, label)
    assert np.allclose(got, expected, rtol=1e-6, atol=1e-7)


def test_oracle_scalar_mean_over_all_elements():
    shape = (3, 4, 5)
    predict, label = _make_inputs(shape, seed=3)
    got = smooth_l1_loss_v2_mean_beta1(predict, label)
    assert np.ndim(got) == 0
    total = np.float32(shape[0] * shape[1] * shape[2])
    abs_err = np.abs(predict - label)
    per = np.where(abs_err < 1.0, (abs_err * abs_err) / 2.0, abs_err - 0.5)
    expected = np.sum(per.astype(np.float32)) / total
    assert np.allclose(got, expected, rtol=1e-5, atol=1e-6)


def test_oracle_dtype_float32():
    predict, label = _make_inputs((6,), seed=4)
    got = smooth_l1_loss_v2_mean_beta1(predict, label)
    assert np.issubdtype(np.asarray(got).dtype, np.floating)


def test_oracle_symmetry_in_sign():
    # Loss depends only on |predict - label|, so swapping sign of both keeps it.
    predict, label = _make_inputs((10,), seed=5)
    a = smooth_l1_loss_v2_mean_beta1(predict, label)
    b = smooth_l1_loss_v2_mean_beta1(-predict, -label)
    assert np.allclose(a, b, rtol=1e-6, atol=1e-7)


# ---------------------------------------------------------------------------
# Runtime tests: public asnumpy.smooth_l1_loss API (guarded)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    asnumpy_available() is None,
    reason="asnumpy or the smooth_l1_loss custom operator runtime is unavailable",
)
class TestAsnumpySmoothL1LossRuntime:
    def _call(self, predict, label):
        import asnumpy

        return asnumpy.smooth_l1_loss(predict, label)

    def test_runtime_matches_oracle_small(self):
        predict, label = _make_inputs((64,), seed=11)
        predict = np.clip(predict, -0.4, 0.4).astype(np.float32)
        label = np.clip(label, -0.4, 0.4).astype(np.float32)
        expected = smooth_l1_loss_v2_mean_beta1(predict, label)
        got = self._call(predict, label)
        got_np = np.asarray(got, dtype=np.float32)
        assert np.allclose(got_np, expected, rtol=1e-4, atol=1e-5)

    def test_runtime_matches_oracle_mixed(self):
        predict, label = _make_inputs((16, 8), seed=12)
        predict = (predict * np.float32(3.0)).astype(np.float32)
        expected = smooth_l1_loss_v2_mean_beta1(predict, label)
        got = self._call(predict, label)
        got_np = np.asarray(got, dtype=np.float32)
        assert np.allclose(got_np, expected, rtol=1e-4, atol=1e-5)

    def test_runtime_zero_error(self):
        x = np.full((4, 5), 1.25, dtype=np.float32)
        got = self._call(x, x.copy())
        got_np = np.asarray(got, dtype=np.float32)
        assert np.allclose(got_np, np.float32(0.0), rtol=1e-5, atol=1e-6)

    def test_runtime_scalar_output(self):
        predict, label = _make_inputs((3, 4), seed=13)
        got = self._call(predict, label)
        assert np.ndim(np.asarray(got)) == 0


def main():
    sys.exit(pytest.main([__file__, "-v"]))


if __name__ == "__main__":
    main()