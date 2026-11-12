"""Custom AsNumpy operator wrapper for ``numpy.argpartition``.

The operator itself is produced by the NumPilot gap lane (Ascend C kernel +
ACLNN adapter); this module is the thin AsNumpy-facing wrapper that admits only
the profile verified on the NPU harness.

Public mapping:
    asnumpy.argpartition
        -> asnumpy._C.argpartition
        -> NumpyArgpartition
        -> aclnnArgpartitionGetWorkspaceSize / aclnnArgpartition
"""

from __future__ import annotations

import numpy as np

from ._core import argpartition as _native_argpartition
from .utils import ndarray

__all__ = ["argpartition"]

_INPUT_TYPE_ERROR = "asnumpy.argpartition expects a single AsNumpy float32 tensor input"
_DTYPE_ERROR = "asnumpy.argpartition supports float32 input only; implicit casting is not allowed"
_RANK_ERROR = "asnumpy.argpartition candidate admits rank 2 float32 input only"
_EMPTY_ERROR = "asnumpy.argpartition candidate rejects empty input"
_KTH_TYPE_ERROR = "asnumpy.argpartition expects an integer kth; bool, float and sequences are rejected"
_KTH_ERROR = "asnumpy.argpartition candidate admits only normalized kth=3"
_AXIS_TYPE_ERROR = "asnumpy.argpartition expects an integer axis; bool and None are rejected"
_AXIS_ERROR = "asnumpy.argpartition candidate admits only axis=1 or its rank-2 alias axis=-1"

def _validate_input(a: ndarray) -> tuple[int, ...]:
    if not isinstance(a, ndarray):
        raise TypeError(_INPUT_TYPE_ERROR)
    if np.dtype(a.dtype) != np.dtype(np.float32):
        raise TypeError(_DTYPE_ERROR)
    shape = tuple(int(dim) for dim in a.shape)
    if len(shape) != 2:
        raise ValueError(_RANK_ERROR)
    if shape[0] <= 0 or shape[1] <= 0:
        raise ValueError(_EMPTY_ERROR)
    return shape


def argpartition(a, /, kth, *, axis=-1) -> ndarray:
    """NumPy-compatible ``argpartition`` on the NPU.

    Admitted profile: rank 2 float32 input, axis = 1 (or its rank-2 alias -1), 0 <= kth < a.shape[1] <= 32, int64 output of the same shape.
    """
    _validate_input(a)
    if isinstance(kth, (bool, np.bool_)) or not isinstance(kth, (int, np.integer)):
        raise TypeError(_KTH_TYPE_ERROR)
    normalized_kth = int(kth)
    if normalized_kth < 0:
        normalized_kth += int(a.shape[1])
    if normalized_kth != 3:
        raise ValueError(_KTH_ERROR)
    if isinstance(axis, (bool, np.bool_)) or not isinstance(axis, (int, np.integer)):
        raise TypeError(_AXIS_TYPE_ERROR)
    if int(axis) not in (1, -1):
        raise ValueError(_AXIS_ERROR)
    return ndarray(_native_argpartition(a, normalized_kth, int(axis)))
