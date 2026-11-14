# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16.0)
project(diff_op_prj LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(ASC REQUIRED)

# =============================================================================
# Architecture generations used by the registry-invoke template
# arch22: ascend910b, ascend910_93 -> diff_arch22.cpp
# arch35: ascend950             -> diff_arch35.cpp
#
# The qualified target for this generated operator is Ascend910B3, which is
# selected by the ascend910b compute unit.  If no unit is supplied, default to
# the qualified target rather than building every known architecture.
# =============================================================================
set(ARCH32_COMPUTE_UNITS ascend910b ascend910_93)
set(ARCH35_COMPUTE_UNITS ascend950)

if(NOT DEFINED ASCEND_COMPUTE_UNIT OR ASCEND_COMPUTE_UNIT STREQUAL "")
    set(ASCEND_COMPUTE_UNIT ascend910b)
endif()

message(STATUS "Diff operator target compute unit: ${ASCEND_COMPUTE_UNIT}")

set(package_name diff_custom)

npu_op_package(${package_name}
    TYPE RUN
    CONFIG
        INSTALL_PATH ${CMAKE_BINARY_DIR}
)

# -----------------------------------------------------------------------------
# Host definitions, shape/dtype inference, tiling, and ACLNN implementation.
# The op_api directory is compiled by op_host/CMakeLists.txt, matching the
# registry-invoke template; it is intentionally not added as a separate
# subdirectory to avoid duplicate ACLNN symbol definitions.
# -----------------------------------------------------------------------------
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/op_host/CMakeLists.txt")
    add_subdirectory(op_host)
else()
    message(WARNING "op_host/CMakeLists.txt is not present; host operator targets will not be built")
endif()

# -----------------------------------------------------------------------------
# Ascend C kernel implementation.
# -----------------------------------------------------------------------------
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/CMakeLists.txt")
    add_subdirectory(op_kernel)
else()
    message(WARNING "op_kernel/CMakeLists.txt is not present; kernel targets will not be built")
endif()

# -----------------------------------------------------------------------------
# Optional graph-mode adaptation.
# -----------------------------------------------------------------------------
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/op_graph/CMakeLists.txt")
    add_subdirectory(op_graph)
endif()

# =============================================================================
# IMPORTANT: cmake's default 'all' target does not build the kernel binary.
# Build the complete package with one of:
#   1. bash build.sh --soc=ascend910b
#   2. cmake --build . --target all binary package
# =============================================================================
message(STATUS "Use 'bash build.sh --soc=ascend910b' or "
               "'cmake --build . --target all binary package' to build the kernel binary and package")