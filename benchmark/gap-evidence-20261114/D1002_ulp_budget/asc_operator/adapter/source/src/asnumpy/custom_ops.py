"""Custom AsNumpy wrapper for ``numpy.spacing`` (generated-operator overlay)."""
from __future__ import annotations

import numpy as np

from ._core import spacing as _native_spacing
from .utils import ndarray

__all__ = ["spacing"]


def spacing(a):
    """NumPy-compatible ``spacing`` on the NPU (float32, rank 2)."""
    if not isinstance(a, ndarray):
        raise TypeError("asnumpy.spacing expects a single AsNumpy float32 tensor input")
    if np.dtype(a.dtype) != np.dtype(np.float32):
        raise TypeError("asnumpy.spacing supports float32 input only; implicit casting is not allowed")
    shape = tuple(int(dim) for dim in a.shape)
    if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
        raise ValueError("asnumpy.spacing candidate admits non-empty rank 2 float32 input only")
    return ndarray(_native_spacing(a))
