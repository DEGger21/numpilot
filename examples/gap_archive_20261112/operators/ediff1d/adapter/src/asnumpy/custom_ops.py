"""Custom AsNumpy operator wrapper for ``numpy.ediff1d``.

The operator itself is produced by the NumPilot gap lane (Ascend C kernel +
ACLNN adapter); this module is the thin AsNumpy-facing wrapper that admits only
the profile verified on the NPU harness.

Public mapping:
    asnumpy.ediff1d
        -> asnumpy._C.ediff1d
        -> NumpyEdiff1d
        -> aclnnEdiff1dGetWorkspaceSize / aclnnEdiff1d
"""

from __future__ import annotations

import numpy as np

from ._core import ediff1d as _native_ediff1d
from .utils import ndarray

__all__ = ["ediff1d"]

_INPUT_TYPE_ERROR = "asnumpy.ediff1d expects a single AsNumpy float32 tensor input"
_DTYPE_ERROR = "asnumpy.ediff1d supports float32 input only; implicit casting is not allowed"
_RANK_ERROR = "asnumpy.ediff1d candidate admits rank 2 float32 input only"
_EMPTY_ERROR = "asnumpy.ediff1d candidate rejects empty input"


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


def ediff1d(a, /) -> ndarray:
    """NumPy-compatible ``ediff1d`` on the NPU.

    Admitted profile: rank 2 float32, flattened output of length numel-1.
    """
    _validate_input(a)
    return ndarray(_native_ediff1d(a))
