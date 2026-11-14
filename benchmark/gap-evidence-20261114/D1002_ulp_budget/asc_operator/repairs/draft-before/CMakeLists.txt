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
project(spacing_op_prj)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(ASC REQUIRED)

# =============================================================================
# Architecture selection
#
# This draft operator targets Ascend910B3 only. Ascend910B3 is built with the
# Ascend910B compute unit (arch22). Other architectures are intentionally not
# admitted by this build until their kernels and tiling implementations exist.
# =============================================================================

set(ARCH32_COMPUTE_UNITS ascend910b)
set(ARCH35_COMPUTE_UNITS "")
set(SPACING_SUPPORTED_COMPUTE_UNITS ${ARCH32_COMPUTE_UNITS})

if(NOT DEFINED ASCEND_COMPUTE_UNIT OR ASCEND_COMPUTE_UNIT STREQUAL "")
    set(ASCEND_COMPUTE_UNIT ascend910b)
endif()

list(FIND SPACING_SUPPORTED_COMPUTE_UNITS "${ASCEND_COMPUTE_UNIT}" SPACING_SOC_INDEX)
if(SPACING_SOC_INDEX EQUAL -1)
    message(FATAL_ERROR
        "Unsupported ASCEND_COMPUTE_UNIT='${ASCEND_COMPUTE_UNIT}'. "
        "The Spacing draft supports only: ${SPACING_SUPPORTED_COMPUTE_UNITS}."
    )
endif()

message(STATUS "Spacing operator build target: ${ASCEND_COMPUTE_UNIT}")

set(package_name spacing_custom)

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

# =============================================================================
# IMPORTANT: cmake's default 'all' target does NOT build 'binary'!
# To build kernel binary, use one of:
#   1. bash build.sh --soc=ascend910b
#   2. cmake --build . --target all binary package
# =============================================================================
message(WARNING "cmake 'make' does NOT build kernel binary by default. Use: bash build.sh --soc=ascend910b")