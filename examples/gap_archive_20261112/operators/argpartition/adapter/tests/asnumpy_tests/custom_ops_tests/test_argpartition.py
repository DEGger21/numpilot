"""Tests for the verified ``asnumpy.argpartition`` custom-operator adapter."""

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
def test_argpartition_observed_32x16_kth3(xp, dtype):
    a = _create_array(xp, _data((32, 16)), dtype)
    return xp.argpartition(a, 3, axis=1)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_argpartition_multi_core_shape(xp, dtype):
    a = _create_array(xp, _data((41, 16)), dtype)
    return xp.argpartition(a, 3, axis=1)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_argpartition_max_axis_length(xp, dtype):
    a = _create_array(xp, _data((5, 32)), dtype)
    return xp.argpartition(a, 3, axis=1)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_argpartition_short_rows(xp, dtype):
    a = _create_array(xp, _data((7, 5)), dtype)
    return xp.argpartition(a, 3, axis=1)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_argpartition_duplicate_values(xp, dtype):
    data = numpy.stack([numpy.arange(16) % 4 for _ in range(6)]).astype(numpy.float32)
    a = _create_array(xp, data, dtype)
    return xp.argpartition(a, 3, axis=1)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_argpartition_special_values(xp, dtype):
    row = numpy.array(
        [numpy.nan, numpy.inf, -numpy.inf, 0.0, -0.0, 1.5, -2.5, 3.25] * 2,
        dtype=numpy.float32,
    )
    a = _create_array(xp, numpy.stack([row] * 4), dtype)
    return xp.argpartition(a, 3, axis=1)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_argpartition_default_axis_and_negative_kth(xp, dtype):
    a = _create_array(xp, _data((32, 16)), dtype)
    return xp.argpartition(a, -13, axis=-1)


def test_argpartition_output_metadata():
    out = asnumpy.argpartition(_make_asnumpy(_data((32, 16))), 3)
    assert isinstance(out, asnumpy.ndarray)
    assert tuple(out.shape) == (32, 16)
    assert numpy.dtype(out.dtype) == numpy.dtype(numpy.int64)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_argpartition_repeated_calls_same_input(xp, dtype):
    a = _create_array(xp, _data((32, 16)), dtype)
    first = xp.argpartition(a, 3, axis=1)
    second = xp.argpartition(a, 3, axis=1)
    return xp.equal(first, second)


def test_argpartition_rejects_non_tensor_input():
    with pytest.raises(TypeError):
        asnumpy.argpartition(numpy.zeros((32, 16), dtype=numpy.float32), 3)


@pytest.mark.parametrize("dtype", [numpy.float16, numpy.int32])
def test_argpartition_rejects_non_float32_dtype(dtype):
    a = asnumpy.ndarray.from_numpy(numpy.zeros((32, 16), dtype=dtype))
    with pytest.raises(TypeError):
        asnumpy.argpartition(a, 3)


def test_argpartition_rejects_rank1_input():
    a = _make_asnumpy(numpy.zeros((16,), dtype=numpy.float32))
    with pytest.raises(ValueError):
        asnumpy.argpartition(a, 3)


def test_argpartition_rejects_long_axis():
    a = _make_asnumpy(numpy.zeros((2, 33), dtype=numpy.float32))
    with pytest.raises(ValueError):
        asnumpy.argpartition(a, 3)


def test_argpartition_rejects_non_admitted_kth():
    a = _make_asnumpy(_data((32, 16)))
    with pytest.raises(ValueError):
        asnumpy.argpartition(a, 4)


def test_argpartition_rejects_sequence_kth():
    a = _make_asnumpy(_data((32, 16)))
    with pytest.raises(TypeError):
        asnumpy.argpartition(a, [3])


def test_argpartition_rejects_bool_kth():
    a = _make_asnumpy(_data((32, 16)))
    with pytest.raises(TypeError):
        asnumpy.argpartition(a, True)


def test_argpartition_rejects_axis0():
    a = _make_asnumpy(_data((32, 16)))
    with pytest.raises(ValueError):
        asnumpy.argpartition(a, 3, axis=0)


def test_argpartition_rejects_unknown_keyword():
    a = _make_asnumpy(_data((32, 16)))
    with pytest.raises(TypeError):
        asnumpy.argpartition(a, 3, kind="quicksort")
