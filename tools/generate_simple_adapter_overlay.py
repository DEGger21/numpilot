#!/usr/bin/env python3
"""Materialize a deterministic AsNumpy adapter overlay for one generated op.

This is the integration step after an Ascend C operator draft has built and
passed its native ACLNN probe.  It copies the local AsNumpy source tree and adds
one thin public binding, wrapper, and C++ ACLNN bridge.  It never calls a model
and never edits the shared AsNumpy checkout.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

SPECS = {
    "spacing": {
        "class": "Spacing",
        "params": "",
        "args": "",
        "py_def": "def spacing(a):",
        "py_doc": "NumPy-compatible ``spacing`` on the NPU (float32, rank 2).",
        "validate": "rank 2, float32, non-empty",
        "call": "_native_spacing(a)",
    },
    "diff": {
        "class": "Diff",
        "params": ", int64_t axis = 1",
        "args": ", axis",
        "py_def": "def diff(a, n=1, axis=-1):",
        "py_doc": "NumPy-compatible first difference on the NPU (float32, rank 2, axis=1).",
        "validate": "rank 2, float32, non-empty, n=1, axis=1",
        "call": "_native_diff(a, axis)",
    },
    "roll": {
        "class": "Roll",
        "params": ", int64_t shift, int64_t axis = 1",
        "args": ", shift, axis",
        "py_def": "def roll(a, shift, axis):",
        "py_doc": "NumPy-compatible roll on the NPU (float32, rank 2, shift=1, axis=1).",
        "validate": "rank 2, float32, non-empty, shift=1, axis=1",
        "call": "_native_roll(a, shift, axis)",
    },
    "nextafter": {
        "class": "Nextafter",
        "params": ", const NPUArray& b",
        "args": ", b",
        "py_def": "def nextafter(a, b):",
        "py_doc": "NumPy-compatible nextafter on the NPU (float32, rank 2).",
        "validate": "rank 2, float32, same shape",
        "call": "_native_nextafter(a, b)",
    },
    "repeat": {
        "class": "Repeat",
        "params": ", int64_t repeats, int64_t axis = 1",
        "args": ", repeats, axis",
        "py_def": "def repeat(a, repeats, axis=1):",
        "py_doc": "NumPy-compatible repeat on the NPU (float32, rank 2, repeats=3, axis=1).",
        "validate": "rank 2, float32, repeats=3, axis=1",
        "call": "_native_repeat(a, repeats, axis)",
    },
    "argmax": {
        "class": "Argmax",
        "params": ", int64_t axis = 1",
        "args": ", axis",
        "py_def": "def argmax(a, axis=1):",
        "py_doc": "NumPy-compatible argmax on the NPU (float32, rank 2, axis=1, int64 output).",
        "validate": "rank 2, float32, axis=1, int64 output",
        "call": "_native_argmax(a, axis)",
    },
}


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _cpp(op: str, spec) -> str:
    cls = spec["class"]
    params = spec["params"]
    result_dtype = "a.dtype"
    if op == "spacing":
        call = "getWorkspaceSize(a.tensorPtr, result.tensorPtr, &workspaceSize, &executor);"
        header = "aclnn_spacing.h"
        validate = """    if (a.shape.size() != 2 || a.shape[0] <= 0 || a.shape[1] <= 0) {
        throw std::invalid_argument("asnumpy.spacing admits non-empty rank 2 input only");
    }"""
        out_shape = "    const std::vector<int64_t> outShape = a.shape;"
        validate_signature = "const NPUArray& a"
        validate_call = f"Validate{cls}Profile(a);"
        axis_init = ""
    elif op == "diff":
        call = "getWorkspaceSize(a.tensorPtr, normalizedAxis, result.tensorPtr, &workspaceSize, &executor);"
        header = "aclnn_diff.h"
        validate = """    if (a.shape.size() != 2 || a.shape[0] <= 0 || a.shape[1] < 2) {
        throw std::invalid_argument("asnumpy.diff admits rank 2 input with at least two columns");
    }
    if (normalizedAxis != 1) {
        throw std::invalid_argument("asnumpy.diff admits only axis=1 or axis=-1");
    }"""
        out_shape = """    std::vector<int64_t> outShape = a.shape;
    outShape[1] -= 1;"""
        validate_signature = "const NPUArray& a, int64_t normalizedAxis"
        validate_call = f"Validate{cls}Profile(a, normalizedAxis);"
        axis_init = "    int64_t normalizedAxis = axis < 0 ? axis + static_cast<int64_t>(a.shape.size()) : axis;\n"
    elif op == "nextafter":
        call = "getWorkspaceSize(a.tensorPtr, b.tensorPtr, result.tensorPtr, &workspaceSize, &executor);"
        header = "aclnn_nextafter.h"
        validate = """    if (b.tensorPtr == nullptr) {
        throw std::invalid_argument("asnumpy.nextafter received a null target tensor");
    }
    if (b.aclDtype != ACL_FLOAT) {
        throw std::invalid_argument("asnumpy.nextafter supports float32 target only");
    }
    if (a.shape != b.shape) {
        throw std::invalid_argument("asnumpy.nextafter requires identical input shapes");
    }
    if (a.shape.size() != 2 || a.shape[0] <= 0 || a.shape[1] <= 0) {
        throw std::invalid_argument("asnumpy.nextafter admits non-empty rank 2 input only");
    }"""
        out_shape = "    const std::vector<int64_t> outShape = a.shape;"
        validate_signature = "const NPUArray& a, const NPUArray& b"
        validate_call = f"Validate{cls}Profile(a, b);"
        axis_init = ""
    elif op == "repeat":
        call = "getWorkspaceSize(a.tensorPtr, repeats, normalizedAxis, result.tensorPtr, &workspaceSize, &executor);"
        header = "aclnn_repeat.h"
        validate = """    if (normalizedAxis != 1) {
        throw std::invalid_argument("asnumpy.repeat admits only axis=1 or axis=-1");
    }
    if (repeats != 3) {
        throw std::invalid_argument("asnumpy.repeat candidate admits only repeats=3");
    }
    if (a.shape.size() != 2 || a.shape[0] <= 0 || a.shape[1] <= 0) {
        throw std::invalid_argument("asnumpy.repeat admits non-empty rank 2 input only");
    }"""
        out_shape = """    std::vector<int64_t> outShape = a.shape;
    outShape[1] *= repeats;"""
        validate_signature = "const NPUArray& a, int64_t repeats, int64_t normalizedAxis"
        validate_call = f"Validate{cls}Profile(a, repeats, normalizedAxis);"
        axis_init = "    int64_t normalizedAxis = axis < 0 ? axis + static_cast<int64_t>(a.shape.size()) : axis;\n"
    elif op == "argmax":
        call = "getWorkspaceSize(a.tensorPtr, normalizedAxis, result.tensorPtr, &workspaceSize, &executor);"
        header = "aclnn_argmax.h"
        validate = """    if (normalizedAxis != 1) {
        throw std::invalid_argument("asnumpy.argmax admits only axis=1 or axis=-1");
    }
    if (a.shape.size() != 2 || a.shape[0] <= 0 || a.shape[1] <= 0) {
        throw std::invalid_argument("asnumpy.argmax admits non-empty rank 2 input only");
    }"""
        out_shape = "    const std::vector<int64_t> outShape{a.shape[0]};"
        validate_signature = "const NPUArray& a, int64_t normalizedAxis"
        validate_call = f"Validate{cls}Profile(a, normalizedAxis);"
        axis_init = "    int64_t normalizedAxis = axis < 0 ? axis + static_cast<int64_t>(a.shape.size()) : axis;\n"
        result_dtype = "ACL_INT64"
    else:
        call = "getWorkspaceSize(a.tensorPtr, shift, normalizedAxis, result.tensorPtr, &workspaceSize, &executor);"
        header = "aclnn_roll.h"
        validate = """    if (a.shape.size() != 2 || a.shape[0] <= 0 || a.shape[1] <= 0) {
        throw std::invalid_argument("asnumpy.roll admits non-empty rank 2 input only");
    }
    if (shift != 1) {
        throw std::invalid_argument("asnumpy.roll candidate admits only shift=1");
    }
    if (normalizedAxis != 1) {
        throw std::invalid_argument("asnumpy.roll admits only axis=1 or axis=-1");
    }"""
        out_shape = "    const std::vector<int64_t> outShape = a.shape;"
        validate_signature = "const NPUArray& a, int64_t shift, int64_t normalizedAxis"
        validate_call = f"Validate{cls}Profile(a, shift, normalizedAxis);"
        axis_init = "    int64_t normalizedAxis = axis < 0 ? axis + static_cast<int64_t>(a.shape.size()) : axis;\n"

    definition_params = params.replace(" = 1", "")
    return f'''#include <asnumpy/custom_ops/{op}.hpp>
#include <asnumpy/custom_ops/custom_opp_symbols.hpp>
#include <asnumpy/utils/acl_executor.hpp>
#include <asnumpy/utils/acl_resource.hpp>
#include <asnumpy/utils/npu_array.hpp>
#include <asnumpy/utils/status_handler.hpp>

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <{header}>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace asnumpy {{
namespace {{

void Validate{cls}Profile({validate_signature}) {{
    if (a.tensorPtr == nullptr) {{
        throw std::invalid_argument("asnumpy.{op} received a null input tensor");
    }}
    if (a.aclDtype != ACL_FLOAT) {{
        throw std::invalid_argument("asnumpy.{op} supports float32 input only; implicit casting is not allowed");
    }}
{validate}
}}

}} // namespace

NPUArray Numpy{cls}(const NPUArray& a{definition_params}) {{
{axis_init}{out_shape}
    {validate_call}

    NPUArray result(outShape, {result_dtype});
    using GetWorkspaceSizeFn = decltype(&aclnn{cls}GetWorkspaceSize);
    using ExecuteFn = decltype(&aclnn{cls});
    const auto getWorkspaceSize = custom_opp::LoadSymbol<GetWorkspaceSizeFn>("aclnn{cls}GetWorkspaceSize");
    const auto execute = custom_opp::LoadSymbol<ExecuteFn>("aclnn{cls}");

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = {call}
    ACLNN_CHECK(error, "aclnn{cls}GetWorkspaceSize");
    error = aclSetAclOpExecutorRepeatable(executor);
    ACLNN_CHECK(error, "aclSetAclOpExecutorRepeatable");
    AclWorkspace workspace(workspaceSize);
    error = execute(workspace.get(), workspaceSize, executor, nullptr);
    ACLNN_CHECK(error, "aclnn{cls}");
    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    error = aclDestroyAclOpExecutor(executor);
    ACLNN_CHECK(error, "aclDestroyAclOpExecutor");
    return result;
}}

}} // namespace asnumpy
'''


def _py(op: str, spec) -> str:
    if op == "spacing":
        body = '''    if not isinstance(a, ndarray):
        raise TypeError("asnumpy.spacing expects a single AsNumpy float32 tensor input")
    if np.dtype(a.dtype) != np.dtype(np.float32):
        raise TypeError("asnumpy.spacing supports float32 input only; implicit casting is not allowed")
    shape = tuple(int(dim) for dim in a.shape)
    if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
        raise ValueError("asnumpy.spacing candidate admits non-empty rank 2 float32 input only")
    return ndarray(_native_spacing(a))'''
    elif op == "diff":
        body = '''    if not isinstance(a, ndarray):
        raise TypeError("asnumpy.diff expects a single AsNumpy float32 tensor input")
    if np.dtype(a.dtype) != np.dtype(np.float32):
        raise TypeError("asnumpy.diff supports float32 input only; implicit casting is not allowed")
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or int(n) != 1:
        raise ValueError("asnumpy.diff candidate admits only n=1")
    if isinstance(axis, (bool, np.bool_)) or not isinstance(axis, (int, np.integer)):
        raise ValueError("asnumpy.diff axis must be an integer")
    shape = tuple(int(dim) for dim in a.shape)
    if len(shape) != 2 or shape[0] <= 0 or shape[1] < 2:
        raise ValueError("asnumpy.diff candidate admits rank 2 input with at least two columns")
    normalized = int(axis) + 2 if int(axis) < 0 else int(axis)
    if normalized != 1:
        raise ValueError("asnumpy.diff candidate admits only axis=1 or axis=-1")
    return ndarray(_native_diff(a, normalized))'''
    elif op == "nextafter":
        body = '''    if not isinstance(a, ndarray):
        raise TypeError("asnumpy.nextafter expects AsNumpy float32 tensor inputs")
    if not isinstance(b, ndarray):
        raise TypeError("asnumpy.nextafter expects AsNumpy float32 tensor inputs")
    if np.dtype(a.dtype) != np.dtype(np.float32) or np.dtype(b.dtype) != np.dtype(np.float32):
        raise TypeError("asnumpy.nextafter supports float32 inputs only; implicit casting is not allowed")
    if tuple(int(dim) for dim in a.shape) != tuple(int(dim) for dim in b.shape):
        raise ValueError("asnumpy.nextafter requires identical input shapes")
    shape = tuple(int(dim) for dim in a.shape)
    if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
        raise ValueError("asnumpy.nextafter candidate admits non-empty rank 2 float32 input only")
    return ndarray(_native_nextafter(a, b))'''
    elif op == "repeat":
        body = '''    if not isinstance(a, ndarray):
        raise TypeError("asnumpy.repeat expects a single AsNumpy float32 tensor input")
    if np.dtype(a.dtype) != np.dtype(np.float32):
        raise TypeError("asnumpy.repeat supports float32 input only; implicit casting is not allowed")
    if isinstance(repeats, (bool, np.bool_)) or not isinstance(repeats, (int, np.integer)) or int(repeats) != 3:
        raise ValueError("asnumpy.repeat candidate admits only repeats=3")
    if isinstance(axis, (bool, np.bool_)) or not isinstance(axis, (int, np.integer)):
        raise ValueError("asnumpy.repeat axis must be an integer")
    shape = tuple(int(dim) for dim in a.shape)
    if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
        raise ValueError("asnumpy.repeat candidate admits non-empty rank 2 float32 input only")
    normalized = int(axis) + 2 if int(axis) < 0 else int(axis)
    if normalized != 1:
        raise ValueError("asnumpy.repeat admits only axis=1 or axis=-1")
    return ndarray(_native_repeat(a, 3, 1))'''
    elif op == "argmax":
        body = '''    if not isinstance(a, ndarray):
        raise TypeError("asnumpy.argmax expects a single AsNumpy float32 tensor input")
    if np.dtype(a.dtype) != np.dtype(np.float32):
        raise TypeError("asnumpy.argmax supports float32 input only; implicit casting is not allowed")
    if isinstance(axis, (bool, np.bool_)) or not isinstance(axis, (int, np.integer)):
        raise ValueError("asnumpy.argmax axis must be an integer")
    shape = tuple(int(dim) for dim in a.shape)
    if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
        raise ValueError("asnumpy.argmax candidate admits non-empty rank 2 float32 input only")
    normalized = int(axis) + 2 if int(axis) < 0 else int(axis)
    if normalized != 1:
        raise ValueError("asnumpy.argmax admits only axis=1 or axis=-1")
    return ndarray(_native_argmax(a, 1))'''
    else:
        body = '''    if not isinstance(a, ndarray):
        raise TypeError("asnumpy.roll expects a single AsNumpy float32 tensor input")
    if np.dtype(a.dtype) != np.dtype(np.float32):
        raise TypeError("asnumpy.roll supports float32 input only; implicit casting is not allowed")
    if isinstance(shift, (bool, np.bool_)) or not isinstance(shift, (int, np.integer)) or int(shift) != 1:
        raise ValueError("asnumpy.roll candidate admits only shift=1")
    if isinstance(axis, (bool, np.bool_)) or not isinstance(axis, (int, np.integer)):
        raise ValueError("asnumpy.roll axis must be an integer")
    shape = tuple(int(dim) for dim in a.shape)
    if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
        raise ValueError("asnumpy.roll candidate admits non-empty rank 2 float32 input only")
    normalized = int(axis) + 2 if int(axis) < 0 else int(axis)
    if normalized != 1:
        raise ValueError("asnumpy.roll admits only axis=1 or axis=-1")
    return ndarray(_native_roll(a, 1, 1))'''
    return f'''"""Custom AsNumpy wrapper for ``numpy.{op}`` (generated-operator overlay)."""
from __future__ import annotations

import numpy as np

from ._core import {op} as _native_{op}
from .utils import ndarray

__all__ = ["{op}"]


{spec["py_def"]}
    """{spec["py_doc"]}"""
{body}
'''


def _binding(op: str, spec) -> str:
    params = spec["params"]
    call_args = ""
    if op == "diff":
        call_args = ", py::arg(\"axis\") = 1"
    elif op == "roll":
        call_args = ", py::arg(\"shift\"), py::arg(\"axis\") = 1"
    elif op == "nextafter":
        call_args = ", py::arg(\"b\").noconvert()"
    elif op == "repeat":
        call_args = ", py::arg(\"repeats\"), py::arg(\"axis\") = 1"
    elif op == "argmax":
        call_args = ", py::arg(\"axis\") = 1"
    definition_params = params.replace(" = 1", "")
    return f'''#include <asnumpy/custom_ops/{op}.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace asnumpy {{
void bind_custom_ops(py::module_& module) {{
    module.def("{op}", &Numpy{spec["class"]}, py::arg("a").noconvert(){call_args},
               "NumPy-compatible asnumpy.{op} on the NPU.");
}}
}} // namespace asnumpy

void bind_custom_ops(py::module_& module) {{
    asnumpy::bind_custom_ops(module);
}}
'''


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", choices=tuple(SPECS), required=True)
    parser.add_argument("--source", default="/home/ma-user/work/asnumpy")
    parser.add_argument("--dest", required=True)
    args = parser.parse_args(argv)
    spec = SPECS[args.op]
    source = Path(args.source)
    dest = Path(args.dest)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(source, dest, symlinks=True)
    _write(dest / "include/asnumpy/custom_ops" / f"{args.op}.hpp", f'''#pragma once
#include <asnumpy/utils/npu_array.hpp>
#include <cstdint>
namespace asnumpy {{
NPUArray Numpy{spec["class"]}(const NPUArray& a{spec["params"]});
}}
''')
    _write(dest / "csrc/custom_ops" / f"{args.op}.cpp", _cpp(args.op, spec))
    _write(dest / "csrc/custom_ops/CMakeLists.txt", f'''add_library(custom_ops OBJECT {args.op}.cpp)
target_include_directories(custom_ops PUBLIC
    ${{CMAKE_SOURCE_DIR}}/include
    ${{ASNUMPY_CUSTOM_OPP_VENDOR}}/op_api/include)
target_compile_definitions(custom_ops PUBLIC
    ASNUMPY_CUSTOM_OPP_LIBRARY=\\"${{ASNUMPY_CUSTOM_OPP_VENDOR}}/op_api/lib/libcust_opapi.so\\")
target_link_libraries(custom_ops PUBLIC fmt::fmt ascend_sdk pybind11::pybind11)
''')
    _write(dest / "bindings/python/bind_custom_ops.cpp", _binding(args.op, spec))
    _write(dest / "src/asnumpy/custom_ops.py", _py(args.op, spec))
    # The reusable symbol loader is identical to the reviewed flip/std overlay.
    source_symbols = Path("/home/ma-user/work/documents/gap-cases-20261112/cases-20261112/flip/custom-v1/adapter-build-v1/asnumpy/include/asnumpy/custom_ops/custom_opp_symbols.hpp")
    if source_symbols.is_file():
        _write(dest / "include/asnumpy/custom_ops/custom_opp_symbols.hpp", source_symbols.read_text(encoding="utf-8"))
    array_py = dest / "src/asnumpy/array.py"
    text = array_py.read_text()
    text = text.replace(
        "return ndarray(_zeros_like(other, _convert_dtype(dtype)))",
        "return ndarray(_zeros_like(other, _convert_dtype(dtype if dtype is not None else getattr(other, \"dtype\", None))))",
    )
    text = text.replace(
        "return ndarray(_full_like(other, value, _convert_dtype(dtype)))",
        "return ndarray(_full_like(other, value, _convert_dtype(dtype if dtype is not None else getattr(other, \"dtype\", None))))",
    )
    text = text.replace(
        "return ndarray(_empty_like(prototype, _convert_dtype(dtype)))",
        "return ndarray(_empty_like(prototype, _convert_dtype(dtype if dtype is not None else getattr(prototype, \"dtype\", None))))",
    )
    text = text.replace(
        "return ndarray(_ones_like(other, _convert_dtype(dtype)))",
        "return ndarray(_ones_like(other, _convert_dtype(dtype if dtype is not None else getattr(other, \"dtype\", None))))",
    )
    array_py.write_text(text)
    csrc = dest / "csrc/CMakeLists.txt"
    text = csrc.read_text()
    if "ASNUMPY_CUSTOM_OPP_VENDOR" not in text:
        text = text.replace(
            "add_library(ascend_sdk INTERFACE)\n",
            "add_library(ascend_sdk INTERFACE)\n\n"
            "set(ASNUMPY_CUSTOM_OPP_VENDOR \"$ENV{ASNUMPY_CUSTOM_OPP_VENDOR}\" CACHE PATH\n"
            "    \"Installed custom OPP vendor directory for generated operators.\")\n"
            "if(ASNUMPY_CUSTOM_OPP_VENDOR STREQUAL \"\")\n"
            "    message(FATAL_ERROR \"ASNUMPY_CUSTOM_OPP_VENDOR must point at the installed custom OPP vendor\")\n"
            "endif()\n",
        )
    text = text.replace("add_subdirectory(utils)\n", "add_subdirectory(utils)\nadd_subdirectory(custom_ops)\n")
    text = text.replace("nn utils)", "nn utils custom_ops)")
    csrc.write_text(text)
    bind_cmake = dest / "bindings/python/CMakeLists.txt"
    text = bind_cmake.read_text()
    text = text.replace("bind_utils.cpp)", "bind_utils.cpp\n    bind_custom_ops.cpp)")
    text = text.replace("    nn\n)", "    nn\n    custom_ops\n)")
    if "ASNUMPY_CUSTOM_OPP_VENDOR}/op_api/include" not in text:
        text = text.replace(
            "    Python::NumPy\n)",
            "    Python::NumPy\n)\ntarget_include_directories(_core PRIVATE ${ASNUMPY_CUSTOM_OPP_VENDOR}/op_api/include)",
        )
    if "libcust_opapi.so" not in text:
        text += """
# Generated custom-operator adapter.
target_link_libraries(_core PUBLIC ${ASNUMPY_CUSTOM_OPP_VENDOR}/op_api/lib/libcust_opapi.so)
set_target_properties(_core PROPERTIES
    BUILD_RPATH "${ASNUMPY_CUSTOM_OPP_VENDOR}/op_api/lib"
    INSTALL_RPATH "${ASNUMPY_CUSTOM_OPP_VENDOR}/op_api/lib")
"""
    bind_cmake.write_text(text)
    module_cpp = dest / "bindings/python/module.cpp"
    text = module_cpp.read_text()
    if "void bind_custom_ops" not in text:
        text = text.replace(
            "void bind_utils(pybind11::module_& utils);\n",
            "void bind_utils(pybind11::module_& utils);\nvoid bind_custom_ops(pybind11::module_& module);\n",
        )
        text = text.replace("    bind_utils(module);\n", "    bind_utils(module);\n    bind_custom_ops(module);\n")
        module_cpp.write_text(text)
    init = dest / "src/asnumpy/__init__.py"
    text = init.read_text()
    marker = "_LAZY_MAPPING = {"
    start = text.index(marker)
    end = text.index("\n}", start)
    text = text[:end] + f'\n    # .custom_ops (generated custom-operator adapter)\n    "{args.op}": ".custom_ops",' + text[end:]
    init.write_text(text)
    print(dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
