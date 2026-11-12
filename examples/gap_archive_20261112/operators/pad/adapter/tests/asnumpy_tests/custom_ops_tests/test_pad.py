"""Tests for the verified ``asnumpy.pad`` custom-operator adapter."""

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
def test_pad_observed_2048x2048(xp, dtype):
    a = _create_array(xp, _data((2048, 2048)), dtype)
    return xp.pad(a, [[1, 1], [2, 2]], mode="constant")


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_pad_symmetric_pair(xp, dtype):
    a = _create_array(xp, _data((6, 18)), dtype)
    return xp.pad(a, (2, 3), mode="constant")


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_pad_scalar_width(xp, dtype):
    a = _create_array(xp, _data((5, 7)), dtype)
    return xp.pad(a, 1, mode="constant")


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_pad_one_sided_widths(xp, dtype):
    a = _create_array(xp, _data((9, 4)), dtype)
    return xp.pad(a, [[0, 5], [3, 0]], mode="constant")


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_pad_zero_width_identity(xp, dtype):
    a = _create_array(xp, _data((6, 6)), dtype)
    return xp.pad(a, [[0, 0], [0, 0]], mode="constant")


def test_pad_output_metadata():
    out = asnumpy.pad(_make_asnumpy(_data((6, 18))), [[1, 1], [2, 2]], mode="constant")
    assert isinstance(out, asnumpy.ndarray)
    assert tuple(out.shape) == (8, 22)
    assert numpy.dtype(out.dtype) == numpy.dtype(numpy.float32)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_pad_repeated_calls_same_input(xp, dtype):
    a = _create_array(xp, _data((6, 18)), dtype)
    first = xp.pad(a, [[1, 1], [2, 2]], mode="constant")
    second = xp.pad(a, [[1, 1], [2, 2]], mode="constant")
    return xp.equal(first, second)


def test_pad_rejects_non_tensor_input():
    with pytest.raises(TypeError):
        asnumpy.pad(numpy.zeros((6, 18), dtype=numpy.float32), 1)


@pytest.mark.parametrize("dtype", [numpy.float16, numpy.int32])
def test_pad_rejects_non_float32_dtype(dtype):
    a = asnumpy.ndarray.from_numpy(numpy.zeros((6, 18), dtype=dtype))
    with pytest.raises(TypeError):
        asnumpy.pad(a, 1)


def test_pad_rejects_rank1_input():
    a = _make_asnumpy(numpy.zeros((18,), dtype=numpy.float32))
    with pytest.raises(ValueError):
        asnumpy.pad(a, 1)


def test_pad_rejects_rank3_input():
    a = _make_asnumpy(numpy.zeros((2, 3, 4), dtype=numpy.float32))
    with pytest.raises(ValueError):
        asnumpy.pad(a, 1)


def test_pad_rejects_non_constant_mode():
    a = _make_asnumpy(_data((6, 18)))
    with pytest.raises(ValueError):
        asnumpy.pad(a, [[1, 1], [2, 2]], mode="reflect")


def test_pad_rejects_negative_width():
    a = _make_asnumpy(_data((6, 18)))
    with pytest.raises(ValueError):
        asnumpy.pad(a, [[-1, 0], [0, 0]])


def test_pad_rejects_malformed_width():
    a = _make_asnumpy(_data((6, 18)))
    with pytest.raises(TypeError):
        asnumpy.pad(a, [[1, 1]])
