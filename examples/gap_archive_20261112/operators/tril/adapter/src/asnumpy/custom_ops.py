"""Custom AsNumpy operator wrapper for ``numpy.tril``.

The operator itself is produced by the NumPilot gap lane (Ascend C kernel +
ACLNN adapter); this module is the thin AsNumpy-facing wrapper that admits only
the profile verified on the NPU harness.

Public mapping:
    asnumpy.tril
        -> asnumpy._C.tril
        -> NumpyTril
        -> aclnnTrilGetWorkspaceSize / aclnnTril
"""

from __future__ import annotations

import numpy as np

from ._core import tril as _native_tril
from .utils import ndarray

__all__ = ["tril"]

_INPUT_TYPE_ERROR = "asnumpy.tril expects a single AsNumpy float32 tensor input"
_DTYPE_ERROR = "asnumpy.tril supports float32 input only; implicit casting is not allowed"
_RANK_ERROR = "asnumpy.tril candidate admits rank 2 float32 input only"
_EMPTY_ERROR = "asnumpy.tril candidate rejects empty input"


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


def tril(a, /) -> ndarray:
    """NumPy-compatible ``tril`` on the NPU.

    Admitted profile: rank 2 float32, y[i, j] = a[i, j] for j <= i and 0 otherwise.
    """
    _validate_input(a)
    return ndarray(_native_tril(a))
