"""Tests for the verified ``asnumpy.ediff1d`` custom-operator adapter."""

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
def test_ediff1d_observed_2048x2048(xp, dtype):
    a = _create_array(xp, _data((2048, 2048)), dtype)
    return xp.ediff1d(a)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-5)
def test_ediff1d_small_shape(xp, dtype):
    a = _create_array(xp, _data((6, 18)), dtype)
    return xp.ediff1d(a)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-5)
def test_ediff1d_single_row(xp, dtype):
    a = _create_array(xp, _data((1, 9)))
    return xp.ediff1d(a)


def test_ediff1d_output_metadata():
    out = asnumpy.ediff1d(_make_asnumpy(_data((6, 18))))
    assert isinstance(out, asnumpy.ndarray)
    assert tuple(out.shape) == (107,)
    assert numpy.dtype(out.dtype) == numpy.dtype(numpy.float32)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_ediff1d_repeated_calls_same_input(xp, dtype):
    a = _create_array(xp, _data((6, 18)), dtype)
    first = xp.ediff1d(a)
    second = xp.ediff1d(a)
    return xp.equal(first, second)


def test_ediff1d_rejects_non_tensor_input():
    with pytest.raises(TypeError):
        asnumpy.ediff1d(numpy.zeros((6, 18), dtype=numpy.float32))


@pytest.mark.parametrize("dtype", [numpy.float16, numpy.int32])
def test_ediff1d_rejects_non_float32_dtype(dtype):
    a = asnumpy.ndarray.from_numpy(numpy.zeros((6, 18), dtype=dtype))
    with pytest.raises(TypeError):
        asnumpy.ediff1d(a)


def test_ediff1d_rejects_rank1_input():
    a = _make_asnumpy(numpy.zeros((18,), dtype=numpy.float32))
    with pytest.raises(ValueError):
        asnumpy.ediff1d(a)


def test_ediff1d_rejects_unknown_keyword():
    a = _make_asnumpy(_data((6, 18)))
    with pytest.raises(TypeError):
        asnumpy.ediff1d(a, to_begin=0.0)
