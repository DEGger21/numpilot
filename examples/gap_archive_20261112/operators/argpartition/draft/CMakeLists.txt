# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
#
# Argpartition custom operator project (registry-invoke layout).
#
# Public API mapping (must stay identical across every generated stage):
#   asnumpy.argpartition  ->  Python wrapper (asnumpy/argpartition.py)
#                         ->  pybind symbol (argpartition)
#                         ->  native adapter implementation
#                         ->  aclnnArgpartitionGetWorkspaceSize / aclnnArgpartition
#                         ->  L0 op_api Argpartition
#                         ->  Ascend C kernel Argpartition
#
# Supported scope for this draft (declared, still unverified):
#   * SoC: Ascend910B3 only (ASCEND_COMPUTE_UNIT=ascend910b, arch22 sources)
#   * CANN: 9.2.0-beta.2
#   * Input dtype: float32, C-contiguous, rank 2 (observed profile [32, 16])
#   * Output dtype: int64, shape equal to the input shape
#   * Signature: 2 positional arguments + keyword 'axis'
#   Anything outside this scope must be rejected explicitly by the runtime layer;
#   no implicit CPU fallback and no implicit dtype casting is permitted.
#
# This file is stage 1/17 (build_01) of a deterministic generation chain.
# It only declares the top-level build graph; the concrete host, kernel,
# graph and ACLNN targets are contributed by their own directories.
# ----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16.0)

project(argpartition_op_prj)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(ASC REQUIRED)

# =============================================================================
# Architecture / SoC configuration
#
# arch22: ascend910b (this project's target: Ascend910B3)
# arch35: ascend950 (not targeted by this draft)
# =============================================================================

set(ARCH22_COMPUTE_UNITS ascend910b)
set(ARCH35_COMPUTE_UNITS ascend950)

# Ascend910B3 is the only SoC this draft is allowed to build for.
set(ARGPARTITION_SUPPORTED_COMPUTE_UNITS ascend910b)
set(ARGPARTITION_DEFAULT_COMPUTE_UNIT ascend910b)

if(NOT DEFINED ASCEND_COMPUTE_UNIT OR ASCEND_COMPUTE_UNIT STREQUAL "")
    set(ASCEND_COMPUTE_UNIT ${ARGPARTITION_DEFAULT_COMPUTE_UNIT})
endif()

foreach(argpartition_compute_unit IN LISTS ASCEND_COMPUTE_UNIT)
    string(TOLOWER "${argpartition_compute_unit}" argpartition_compute_unit_lc)
    list(FIND ARGPARTITION_SUPPORTED_COMPUTE_UNITS "${argpartition_compute_unit_lc}" argpartition_unit_index)
    if(argpartition_unit_index EQUAL -1)
        message(FATAL_ERROR
            "Unsupported ASCEND_COMPUTE_UNIT '${argpartition_compute_unit}'. "
            "This Argpartition draft targets Ascend910B3 only, so the accepted value is: "
            "${ARGPARTITION_SUPPORTED_COMPUTE_UNITS}. "
            "Widen the declared support scope and re-verify before enabling other SoCs.")
    endif()
endforeach()

message(STATUS "Argpartition: building for ASCEND_COMPUTE_UNIT=${ASCEND_COMPUTE_UNIT} (target SoC Ascend910B3)")

# =============================================================================
# Operator package
# =============================================================================

set(package_name argpartition_custom)

npu_op_package(${package_name}
    TYPE RUN
    CONFIG
        INSTALL_PATH ${CMAKE_BINARY_DIR}
)

# =============================================================================
# Sub-directories
#
#   op_host   -> operator definition, infer-shape, tiling, ACLNN libraries
#   op_kernel -> Ascend C kernel entry (arch22 for ascend910b)
#   op_graph  -> graph-mode (GE IR) prototype
#
# Each directory is optional at configure time so a partial checkout still
# gives a readable CMake error instead of a silent half-built package.
# =============================================================================

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/op_host")
    add_subdirectory(op_host)
else()
    message(STATUS "Argpartition: op_host directory not present in this stage; host libraries will be skipped")
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/op_kernel")
    add_subdirectory(op_kernel)
else()
    message(STATUS "Argpartition: op_kernel directory not present in this stage; kernel binary will be skipped")
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/op_graph")
    add_subdirectory(op_graph)
else()
    message(STATUS "Argpartition: op_graph directory not present in this stage; graph prototype will be skipped")
endif()

# =============================================================================
# IMPORTANT: cmake's default 'all' target does NOT build 'binary'!
# To build the kernel binary and the install package, use one of:
#   1. bash build.sh --soc=ascend910b          (recommended)
#   2. cmake --build . --target all binary package
# =============================================================================

message(WARNING "cmake 'make' does NOT build the Argpartition kernel binary by default. Use: bash build.sh --soc=ascend910b")