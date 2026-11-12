"""Custom AsNumpy operator wrapper for ``numpy.flip``.

The operator itself is produced by the NumPilot gap lane (Ascend C kernel +
ACLNN adapter); this module is the thin AsNumpy-facing wrapper that admits only
the profile verified on the NPU harness.

Public mapping:
    asnumpy.flip
        -> asnumpy._C.flip
        -> NumpyFlip
        -> aclnnFlipGetWorkspaceSize / aclnnFlip
"""

from __future__ import annotations

import numpy as np

from ._core import flip as _native_flip
from .utils import ndarray

__all__ = ["flip"]

_INPUT_TYPE_ERROR = "asnumpy.flip expects a single AsNumpy float32 tensor input"
_DTYPE_ERROR = "asnumpy.flip supports float32 input only; implicit casting is not allowed"
_RANK_ERROR = "asnumpy.flip candidate admits rank 2 float32 input only"
_EMPTY_ERROR = "asnumpy.flip candidate rejects empty input"
_AXIS_ERROR = "asnumpy.flip candidate admits only axis=1 or its rank-2 alias axis=-1"

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


def flip(a, /, *, axis=1) -> ndarray:
    """NumPy-compatible ``flip`` on the NPU.

    Admitted profile: rank 2 float32, axis = 1 (or its rank-2 alias -1).
    """
    _validate_input(a)
    if isinstance(axis, (bool, np.bool_)) or not isinstance(axis, (int, np.integer)):
        raise ValueError(_AXIS_ERROR)
    if int(axis) not in (1, -1):
        raise ValueError(_AXIS_ERROR)
    return ndarray(_native_flip(a, axis=1))
