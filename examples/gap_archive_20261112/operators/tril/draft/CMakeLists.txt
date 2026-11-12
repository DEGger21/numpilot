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
# Tril operator project (numpy.tril -> asnumpy.tril)
#   OpType : Tril
#   Stem   : tril
#   Target : Ascend910B3 (CANN 9.2.0-beta.2) -> ascend910b compute unit (arch22)
#
# Supported build scope for this candidate:
#   - runtime input profile: float32, rank 2, shape [2048, 2048], C contiguous
#   - output               : float32, same shape as input (single tensor return)
#   - signature            : tril(a)   -- 1 positional argument, no keywords
# Any dtype/shape outside the declared support scope is rejected at the ACLNN
# parameter-check layer; there is no implicit cast and no host fallback.
#
# IMPORTANT: cmake's default 'all' target does NOT build the kernel binary.
# Use one of:
#   1. bash build.sh --soc=ascend910b           (recommended)
#   2. cmake --build . --target all binary package
# ----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16.0)

project(tril_op_prj)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(ASC REQUIRED)

# =============================================================================
# Architecture generation split
#   arch22 -> ascend910b, ascend910_93   : op_kernel/tril_arch22.cpp
#   arch35 -> ascend950                  : op_kernel/tril_arch35.cpp
# =============================================================================
set(ARCH32_COMPUTE_UNITS ascend910b ascend910_93)
set(ARCH35_COMPUTE_UNITS ascend950)

# If the caller did not pin a compute unit, build every supported generation.
# The declared target for this candidate is Ascend910B3 => ascend910b.
if(NOT DEFINED ASCEND_COMPUTE_UNIT OR ASCEND_COMPUTE_UNIT STREQUAL "")
    set(ASCEND_COMPUTE_UNIT ${ARCH32_COMPUTE_UNITS} ${ARCH35_COMPUTE_UNITS})
endif()

# Package name is shared with op_host/ and op_kernel/ sub-directories.
set(package_name tril_custom)

npu_op_package(${package_name}
    TYPE RUN
    CONFIG
        INSTALL_PATH ${CMAKE_BINARY_DIR}
)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/op_host")
    add_subdirectory(op_host)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/op_kernel")
    add_subdirectory(op_kernel)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/op_graph")
    add_subdirectory(op_graph)
endif()

message(STATUS "tril: package_name       = ${package_name}")
message(STATUS "tril: ASCEND_COMPUTE_UNIT = ${ASCEND_COMPUTE_UNIT}")
message(WARNING "cmake 'make' does NOT build kernel binary by default. Use: bash build.sh --soc=ascend910b")