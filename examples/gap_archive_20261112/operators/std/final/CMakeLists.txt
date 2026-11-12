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
project(std_op_prj LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(ASC REQUIRED)

# =============================================================================
# Ascend architecture generation mapping
#
# arch22: ascend910b, ascend910_93 -> std_arch22.cpp
# arch35: ascend950                 -> std_arch35.cpp
#
# This generated candidate targets Ascend910B3 only. The operator must not
# silently claim support for other SoCs or CANN versions.
# =============================================================================

set(ARCH32_COMPUTE_UNITS ascend910b ascend910_93)
set(ARCH35_COMPUTE_UNITS ascend950)

set(STD_TARGET_SOC "Ascend910B3")
set(STD_TARGET_COMPUTE_UNIT "ascend910b")
set(STD_TARGET_CANN_VERSION "9.2.0-beta.2")

if(NOT DEFINED ASCEND_COMPUTE_UNIT OR ASCEND_COMPUTE_UNIT STREQUAL "")
    set(ASCEND_COMPUTE_UNIT "${STD_TARGET_COMPUTE_UNIT}")
endif()

if(NOT ASCEND_COMPUTE_UNIT STREQUAL "${STD_TARGET_COMPUTE_UNIT}")
    message(FATAL_ERROR
        "std targets ${STD_TARGET_SOC} / CANN ${STD_TARGET_CANN_VERSION} only; "
        "requested ASCEND_COMPUTE_UNIT='${ASCEND_COMPUTE_UNIT}', "
        "expected '${STD_TARGET_COMPUTE_UNIT}'")
endif()

set(package_name std_custom)

# =============================================================================
# Runtime package
# =============================================================================
npu_op_package(${package_name}
    TYPE RUN
    CONFIG
        INSTALL_PATH ${CMAKE_BINARY_DIR}
)

# =============================================================================
# Host, kernel, and graph components
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
# IMPORTANT: cmake's default 'all' target does NOT build the kernel binary.
# To build the kernel binary and the runtime package, use:
#   1. bash build.sh --soc=ascend910b
#   2. cmake --build . --target all binary package
# =============================================================================
message(STATUS "std operator target: ${STD_TARGET_SOC} (${STD_TARGET_COMPUTE_UNIT}), CANN ${STD_TARGET_CANN_VERSION}")
message(WARNING "cmake 'make' does NOT build the kernel binary by default. Use: bash build.sh --soc=ascend910b")