"""Custom AsNumpy operator wrapper for ``numpy.pad``.

The operator itself is produced by the NumPilot gap lane (Ascend C kernel +
ACLNN adapter); this module is the thin AsNumpy-facing wrapper that admits only
the profile verified on the NPU harness.

Public mapping:
    asnumpy.pad
        -> asnumpy._C.pad
        -> Pad
        -> aclnnPadGetWorkspaceSize / aclnnPad
"""

from __future__ import annotations

import numpy as np

from ._core import pad as _native_pad
from .utils import ndarray

__all__ = ["pad"]

_INPUT_TYPE_ERROR = "asnumpy.pad expects a single AsNumpy float32 tensor input"
_DTYPE_ERROR = "asnumpy.pad supports float32 input only; implicit casting is not allowed"
_RANK_ERROR = "asnumpy.pad candidate admits rank 2 float32 input only"
_EMPTY_ERROR = "asnumpy.pad candidate rejects empty input"
_PAD_WIDTH_ERROR = "asnumpy.pad expects pad_width as an int or two (before, after) pairs"
_MODE_ERROR = "asnumpy.pad candidate admits only mode='constant'"
_NEGATIVE_ERROR = "asnumpy.pad candidate rejects negative pad widths"
_SHAPE_ERROR = "asnumpy.pad candidate requires a positive output shape"


def _is_int(value) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(
        value, (bool, np.bool_)
    )


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


def _normalize_pad_width(pad_width) -> list[int]:
    """Flatten the NumPy pad-width forms into four (before, after) entries."""

    if _is_int(pad_width):
        pair = [int(pad_width), int(pad_width)]
        return pair + pair
    if isinstance(pad_width, (str, bytes)) or pad_width is None:
        raise TypeError(_PAD_WIDTH_ERROR)
    try:
        items = list(pad_width)
    except TypeError:
        raise TypeError(_PAD_WIDTH_ERROR) from None
    if len(items) == 2 and all(_is_int(value) for value in items):
        pair = [int(items[0]), int(items[1])]
        return pair + pair
    if len(items) == 2:
        flat: list[int] = []
        for item in items:
            if isinstance(item, (str, bytes)) or not hasattr(item, "__len__"):
                raise TypeError(_PAD_WIDTH_ERROR)
            pair = list(item)
            if len(pair) != 2 or not all(_is_int(value) for value in pair):
                raise TypeError(_PAD_WIDTH_ERROR)
            flat.extend(int(value) for value in pair)
        return flat
    raise TypeError(_PAD_WIDTH_ERROR)


def pad(a, pad_width, /, *, mode="constant") -> ndarray:
    """NumPy-compatible ``pad`` on the NPU.

    Admitted profile: rank 2 float32 input, mode='constant' with constant
    value 0, non-negative pad widths.
    """
    shape = _validate_input(a)
    if mode != "constant":
        raise ValueError(_MODE_ERROR)
    pads = _normalize_pad_width(pad_width)
    if any(value < 0 for value in pads):
        raise ValueError(_NEGATIVE_ERROR)
    if shape[0] + pads[0] + pads[1] <= 0 or shape[1] + pads[2] + pads[3] <= 0:
        raise ValueError(_SHAPE_ERROR)
    return ndarray(_native_pad(a, pads))
