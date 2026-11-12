"""Tests for the verified ``asnumpy.tril`` custom-operator adapter."""

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
@testing.numpy_asnumpy_array_equal()
def test_tril_observed_2048x2048(xp, dtype):
    a = _create_array(xp, _data((2048, 2048)), dtype)
    return xp.tril(a)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_tril_small_shapes(xp, dtype):
    a = _create_array(xp, _data((6, 18)), dtype)
    return xp.tril(a)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_tril_multi_core_shape(xp, dtype):
    a = _create_array(xp, _data((41, 8)), dtype)
    return xp.tril(a)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_tril_wide_rows_multichunk(xp, dtype):
    a = _create_array(xp, _data((3, 8193)), dtype)
    return xp.tril(a)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_tril_tall_and_narrow(xp, dtype):
    a = _create_array(xp, _data((5, 2)), dtype)
    return xp.tril(a)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_tril_single_element(xp, dtype):
    a = _create_array(xp, _data((1, 1)), dtype)
    return xp.tril(a)


def test_tril_output_metadata():
    out = asnumpy.tril(_make_asnumpy(_data((6, 18))))
    assert isinstance(out, asnumpy.ndarray)
    assert tuple(out.shape) == (6, 18)
    assert numpy.dtype(out.dtype) == numpy.dtype(numpy.float32)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_tril_repeated_calls_same_input(xp, dtype):
    a = _create_array(xp, _data((6, 18)), dtype)
    first = xp.tril(a)
    second = xp.tril(a)
    return xp.equal(first, second)


def test_tril_rejects_non_tensor_input():
    with pytest.raises(TypeError):
        asnumpy.tril(numpy.zeros((6, 18), dtype=numpy.float32))


@pytest.mark.parametrize("dtype", [numpy.float16, numpy.int32])
def test_tril_rejects_non_float32_dtype(dtype):
    a = asnumpy.ndarray.from_numpy(numpy.zeros((6, 18), dtype=dtype))
    with pytest.raises(TypeError):
        asnumpy.tril(a)


def test_tril_rejects_rank1_input():
    a = _make_asnumpy(numpy.zeros((18,), dtype=numpy.float32))
    with pytest.raises(ValueError):
        asnumpy.tril(a)


def test_tril_rejects_unknown_keyword():
    a = _make_asnumpy(_data((6, 18)))
    with pytest.raises(TypeError):
        asnumpy.tril(a, k=0)
