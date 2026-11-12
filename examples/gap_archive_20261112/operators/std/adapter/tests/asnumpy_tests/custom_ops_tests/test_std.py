"""Tests for the verified ``asnumpy.std`` custom-operator adapter."""

import numpy
import pytest

import asnumpy
from asnumpy import testing


def _create_array(xp, data, dtype=numpy.float32):
    np_arr = numpy.array(data, dtype=dtype, order="C")
    if xp is numpy:
        return np_arr
    return xp.ndarray.from_numpy(np_arr)


def _make_asnumpy(data):
    return asnumpy.ndarray.from_numpy(numpy.array(data, dtype=numpy.float32, order="C"))


def _data(shape):
    rng = numpy.random.RandomState(20261112)
    return rng.standard_normal(shape).astype(numpy.float32)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-5)
def test_std_observed_2048x2048_axis1(xp, dtype):
    a = _create_array(xp, _data((2048, 2048)), dtype)
    return xp.std(a, axis=1)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-5)
def test_std_small_shapes_axis1(xp, dtype):
    a = _create_array(xp, _data((6, 18)), dtype)
    return xp.std(a, axis=1)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-5)
def test_std_negative_axis_alias(xp, dtype):
    a = _create_array(xp, _data((41, 8)), dtype)
    return xp.std(a, axis=-1)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-5)
def test_std_constant_rows(xp, dtype):
    a = _create_array(xp, numpy.full((18, 64), 3.5), dtype)
    return xp.std(a, axis=1)


def test_std_output_metadata():
    out = asnumpy.std(_make_asnumpy(_data((6, 18))), axis=1)
    assert isinstance(out, asnumpy.ndarray)
    assert tuple(out.shape) == (6,)
    assert numpy.dtype(out.dtype) == numpy.dtype(numpy.float32)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_std_nan_propagation_mask(xp, dtype):
    data = _data((6, 18))
    data = numpy.array(data, dtype=dtype, order="C")
    data[2, 5] = numpy.nan
    a = _create_array(xp, data, dtype)
    out = xp.std(a, axis=1)
    return xp.not_equal(out, out)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_std_repeated_calls_same_input(xp, dtype):
    a = _create_array(xp, _data((6, 18)), dtype)
    first = xp.std(a, axis=1)
    second = xp.std(a, axis=1)
    return xp.equal(first, second)


def test_std_rejects_non_tensor_input():
    with pytest.raises(TypeError):
        asnumpy.std(numpy.zeros((6, 18), dtype=numpy.float32))


@pytest.mark.parametrize("dtype", [numpy.float16, numpy.int32])
def test_std_rejects_non_float32_dtype(dtype):
    a = asnumpy.ndarray.from_numpy(numpy.zeros((6, 18), dtype=dtype))
    with pytest.raises(TypeError):
        asnumpy.std(a, axis=1)


def test_std_rejects_rank1_input():
    a = _make_asnumpy(numpy.zeros((18,), dtype=numpy.float32))
    with pytest.raises(ValueError):
        asnumpy.std(a, axis=0)


def test_std_rejects_axis0():
    a = _make_asnumpy(_data((6, 18)))
    with pytest.raises(ValueError):
        asnumpy.std(a, axis=0)


def test_std_rejects_bool_axis():
    a = _make_asnumpy(_data((6, 18)))
    with pytest.raises(ValueError):
        asnumpy.std(a, axis=True)


def test_std_rejects_unknown_keyword():
    a = _make_asnumpy(_data((6, 18)))
    with pytest.raises(TypeError):
        asnumpy.std(a, axis=1, keepdims=True)
