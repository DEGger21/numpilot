#!/usr/bin/env python3
"""Call one generated custom ACLNN operator on the NPU and compare with NumPy.

This is a deterministic device probe, not a model call.  It loads the operator
package's ``libcust_opapi.so``, creates ACL tensors through ``libopapi.so`` and
executes the generated two-stage ACLNN API.  No CPU fallback exists in this
harness: an ACLNN failure is reported as a failure.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time
from pathlib import Path

import acl
import numpy as np

ACL_FLOAT = 0
ACL_FORMAT_ND = 2
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2


def _check(rc, label):
    if rc != 0:
        raise RuntimeError("%s failed rc=%s" % (label, rc))


def _load_libs(vendor: Path):
    cannn = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann-9.2.0-beta.2"))
    # Tensor descriptor constructors are supplied by libnnopbase on this CANN
    # build; libopapi exports a same-named symbol with an incompatible ABI.
    tensor_lib = ctypes.CDLL(
        str(cannn / "aarch64-linux/lib64/libnnopbase.so"), mode=ctypes.RTLD_GLOBAL
    )
    ctypes.CDLL(str(cannn / "aarch64-linux/lib64/libopapi.so"), mode=ctypes.RTLD_GLOBAL)
    lib = ctypes.CDLL(str(vendor / "op_api/lib/libcust_opapi.so"), mode=ctypes.RTLD_GLOBAL)
    tensor_lib.aclCreateTensor.restype = ctypes.c_void_p
    tensor_lib.aclCreateTensor.argtypes = [
        ctypes.POINTER(ctypes.c_int64), ctypes.c_uint64, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int64), ctypes.c_uint64, ctypes.c_void_p,
    ]
    tensor_lib.aclDestroyTensor.argtypes = [ctypes.c_void_p]
    tensor_lib.aclDestroyTensor.restype = ctypes.c_int
    return lib, tensor_lib


def _strides(shape):
    out = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        out[i] = out[i + 1] * shape[i + 1]
    return out


def _make_tensor(lib, array: np.ndarray, *, upload: bool):
    array = np.ascontiguousarray(array)
    dev, rc = acl.rt.malloc(array.nbytes, 0)
    _check(rc, "aclrtMalloc")
    if upload:
        _check(acl.rt.memcpy(dev, array.nbytes, array.ctypes.data, array.nbytes,
                             ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy H2D")
    shape = (ctypes.c_int64 * array.ndim)(*array.shape)
    strides = (ctypes.c_int64 * array.ndim)(*_strides(array.shape))
    tensor = lib.aclCreateTensor(
        shape, array.ndim, ACL_FLOAT, strides, 0, ACL_FORMAT_ND,
        shape, array.ndim, ctypes.c_void_p(dev),
    )
    if not tensor:
        raise RuntimeError("aclCreateTensor returned NULL")
    return tensor, dev, array


def _download(array: np.ndarray, dev):
    _check(acl.rt.memcpy(array.ctypes.data, array.nbytes, dev, array.nbytes,
                         ACL_MEMCPY_DEVICE_TO_HOST), "aclrtMemcpy D2H")


def _call(lib, op_name, x_tensor, scalars, out_tensor, stream):
    get_ws = getattr(lib, "aclnn%sGetWorkspaceSize" % op_name)
    run = getattr(lib, "aclnn%s" % op_name)
    get_ws.restype = ctypes.c_int
    run.restype = ctypes.c_int
    # ctypes cannot infer pointer/int widths from Python values; without
    # argtypes the generated ACLNN call receives truncated descriptors and
    # faults inside libcust_opapi.
    if op_name == "Spacing":
        get_ws.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                           ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)]
    elif op_name == "Diff":
        get_ws.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p,
                           ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)]
    elif op_name == "Roll":
        get_ws.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p,
                           ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)]
    run.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p]
    workspace = ctypes.c_uint64(0)
    executor = ctypes.c_void_p()
    rc = get_ws(x_tensor, *scalars, out_tensor, ctypes.byref(workspace), ctypes.byref(executor))
    if rc != 0:
        raise RuntimeError("aclnn%sGetWorkspaceSize failed rc=%d" % (op_name, rc))
    workspace_ptr = ctypes.c_void_p(0)
    if workspace.value:
        workspace_ptr, malloc_rc = acl.rt.malloc(workspace.value, 0)
        _check(malloc_rc, "aclrtMalloc workspace")
    rc = run(workspace_ptr, workspace.value, executor, stream)
    if rc != 0:
        raise RuntimeError("aclnn%s failed rc=%d" % (op_name, rc))
    _check(acl.rt.synchronize_stream(stream), "aclrtSynchronizeStream")
    return workspace.value


def _oracle(profile, x, axis, shift):
    if profile == "spacing":
        return np.spacing(x)
    if profile == "diff":
        return np.diff(x, axis=axis)
    if profile == "roll":
        return np.roll(x, shift, axis=axis)
    raise ValueError(profile)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("spacing", "diff", "roll"), required=True)
    parser.add_argument("--vendor", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--axis", type=int, default=1)
    parser.add_argument("--shift", type=int, default=1)
    parser.add_argument("--shape", default="1024x1024")
    args = parser.parse_args(argv)
    shape = tuple(int(part) for part in args.shape.lower().split("x"))
    x = np.random.default_rng(20261114).standard_normal(shape, dtype=np.float32)
    x.reshape(-1)[:6] = np.array([0.0, -0.0, np.inf, -np.inf, np.nan, 1.0], dtype=np.float32)
    expected = _oracle(args.profile, x, args.axis, args.shift)
    vendor = Path(args.vendor).resolve()
    lib, tensor_lib = _load_libs(vendor)
    _check(acl.init(), "aclInit")
    _check(acl.rt.set_device(0), "aclrtSetDevice")
    context, rc = acl.rt.create_context(0)
    _check(rc, "aclrtCreateContext")
    _check(acl.rt.set_context(context), "aclrtSetContext")
    stream, rc = acl.rt.create_stream()
    _check(rc, "aclrtCreateStream")

    x_tensor = out_tensor = None
    x_dev = out_dev = 0
    out = np.empty(expected.shape, dtype=np.float32)
    try:
        x_tensor, x_dev, _ = _make_tensor(tensor_lib, x, upload=True)
        out_tensor, out_dev, _ = _make_tensor(tensor_lib, out, upload=False)
        if args.profile == "spacing":
            scalars = ()
        elif args.profile == "diff":
            scalars = (ctypes.c_int64(args.axis),)
        else:
            scalars = (ctypes.c_int64(args.shift), ctypes.c_int64(args.axis))
        started = time.perf_counter()
        workspace_bytes = _call(lib, args.profile.capitalize(), x_tensor, scalars, out_tensor, stream)
        elapsed = time.perf_counter() - started
        _download(out, out_dev)
    finally:
        for tensor in (x_tensor, out_tensor):
            if tensor:
                tensor_lib.aclDestroyTensor(tensor)
        for ptr in (x_dev, out_dev):
            if ptr:
                acl.rt.free(ptr)
        acl.rt.destroy_stream(stream)
        acl.rt.destroy_context(context)
        acl.rt.reset_device(0)
        acl.finalize()

    actual_bits = np.ascontiguousarray(out).view(np.uint32)
    expected_bits = np.ascontiguousarray(expected).view(np.uint32)
    exact = bool(np.array_equal(actual_bits, expected_bits))
    expected_nan = np.isnan(expected)
    actual_nan = np.isnan(out)
    nan_positions_equal = bool(np.array_equal(expected_nan, actual_nan))
    non_nan = ~expected_nan
    non_nan_bits_equal = bool(np.array_equal(actual_bits[non_nan], expected_bits[non_nan]))
    semantic_equal = bool(nan_positions_equal and non_nan_bits_equal)
    mismatch = actual_bits != expected_bits
    mismatch_count = int(mismatch.sum())
    samples = []
    for flat_index in np.flatnonzero(mismatch.reshape(-1))[:8]:
        samples.append({
            "x": repr(float(x.reshape(-1)[flat_index])),
            "expected": repr(float(expected.reshape(-1)[flat_index])),
            "actual": repr(float(out.reshape(-1)[flat_index])),
            "expected_bits": int(expected_bits.reshape(-1)[flat_index]),
            "actual_bits": int(actual_bits.reshape(-1)[flat_index]),
        })
    diff = np.abs(out.astype(np.float64) - expected.astype(np.float64))
    finite = np.isfinite(diff)
    max_abs = float(diff[finite].max()) if finite.any() else 0.0
    result = {
        "profile": args.profile,
        "shape": list(shape),
        "axis": args.axis if args.profile != "spacing" else None,
        "shift": args.shift if args.profile == "roll" else None,
        "output_shape": list(out.shape),
        "workspace_bytes": workspace_bytes,
        "elapsed_seconds": elapsed,
        "exact_equal": exact,
        "semantic_equal": semantic_equal,
        "nan_positions_equal": nan_positions_equal,
        "non_nan_bits_equal": non_nan_bits_equal,
        "mismatch_count": mismatch_count,
        "mismatch_samples": samples,
        "max_abs_diff": max_abs,
        "numpy_oracle": "numpy.%s" % args.profile,
        "status": "pass" if semantic_equal else "fail",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0 if exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
