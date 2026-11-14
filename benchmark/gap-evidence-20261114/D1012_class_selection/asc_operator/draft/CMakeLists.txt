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
# Root build configuration for the Argmax custom operator package.
#
# Operator identity (kept in sync with op_host/op_kernel/op_api and the public AsNumpy adapter):
#   - C++ OpType        : Argmax
#   - file/API stem     : argmax
#   - public Python API : asnumpy.argmax
#   - ACLNN two-stage   : aclnnArgmaxGetWorkspaceSize / aclnnArgmax
#
# This package is a draft candidate. It targets Ascend910B3 (SoC version "ascend910b", arch22) only and
# makes no claim of qualification: numerical and device gates are executed by the deterministic validator.
# ----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16.0)
project(argmax_op_prj)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ASC REQUIRED)

# =============================================================================
# Architecture generation configuration
#   arch22: ascend910b (Ascend910B3), ascend910_93 -> op_kernel/argmax_arch22.cpp
#   arch35: ascend950                            -> op_kernel/argmax_arch35.cpp
# The kernel-generation switch mirrors op_host/CMakeLists.txt (tiling source selection) and
# op_kernel/CMakeLists.txt (kernel entry selection) so all three stay consistent.
# =============================================================================
set(ARCH32_COMPUTE_UNITS ascend910b ascend910_93)
set(ARCH35_COMPUTE_UNITS ascend950)
set(ARGMAX_SUPPORTED_COMPUTE_UNITS ${ARCH32_COMPUTE_UNITS} ${ARCH35_COMPUTE_UNITS})

# The declared target of this draft is Ascend910B3, i.e. SoC version "ascend910b".
# When the caller does not select a SoC explicitly, build only the declared target instead of
# silently widening the build to architectures that have not been designed or validated.
if(NOT DEFINED ASCEND_COMPUTE_UNIT OR ASCEND_COMPUTE_UNIT STREQUAL "")
    set(ASCEND_COMPUTE_UNIT ascend910b)
endif()

list(FIND ARGMAX_SUPPORTED_COMPUTE_UNITS "${ASCEND_COMPUTE_UNIT}" ARGMAX_SOC_INDEX)
if(ARGMAX_SOC_INDEX EQUAL -1)
    message(FATAL_ERROR
        "Unsupported ASCEND_COMPUTE_UNIT='${ASCEND_COMPUTE_UNIT}'. "
        "Supported compute units: ${ARGMAX_SUPPORTED_COMPUTE_UNITS}")
endif()
message(STATUS "[argmax] Target compute unit: ${ASCEND_COMPUTE_UNIT}")

set(package_name argmax_custom)

# =============================================================================
# Package definition
# =============================================================================
npu_op_package(${package_name}
    TYPE RUN
    CONFIG
        INSTALL_PATH ${CMAKE_BINARY_DIR}
)

# =============================================================================
# Subdirectories (each one carries its own source list; none is optional in content,
# the existence checks only keep the tree usable while a subdirectory is absent).
#   op_host   : schema / infer-shape / infer-dtype / tiling registration
#   op_kernel : Ascend C kernel entry (Init -> CopyIn -> Compute -> CopyOut)
#   op_graph  : graph-mode proto adaptation
# =============================================================================
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/op_host")
    add_subdirectory(op_host)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/op_kernel")
    add_subdirectory(op_kernel)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/op_graph")
    add_subdirectory(op_graph)
endif()

# =============================================================================
# IMPORTANT: cmake's default 'all' target does NOT build 'binary'!
# To build the kernel binary, use one of:
#   1. bash build.sh --soc=ascend910b          (recommended)
#   2. cmake --build . --target all binary package
# =============================================================================
message(WARNING "cmake 'make' does NOT build the argmax kernel binary by default. Use: bash build.sh --soc=ascend910b")