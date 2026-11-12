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
project(flip_op_prj)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(ASC REQUIRED)

# =============================================================================
# Target architecture configuration
# Ascend910B3 is selected through ASCEND_COMPUTE_UNIT=ascend910b.
# This draft deliberately targets only the observed Ascend910B3 profile.
# =============================================================================
set(FLIP_SUPPORTED_COMPUTE_UNITS ascend910b)

if(NOT DEFINED ASCEND_COMPUTE_UNIT OR ASCEND_COMPUTE_UNIT STREQUAL "")
    set(ASCEND_COMPUTE_UNIT ascend910b)
endif()
set(ASCEND_COMPUTE_UNIT "${ASCEND_COMPUTE_UNIT}" CACHE STRING "Ascend compute unit for the flip operator" FORCE)

if(NOT ASCEND_COMPUTE_UNIT STREQUAL "ascend910b")
    message(FATAL_ERROR
        "flip draft supports Ascend910B3 only. "
        "Expected ASCEND_COMPUTE_UNIT=ascend910b, got '${ASCEND_COMPUTE_UNIT}'.")
endif()

set(package_name flip_custom)

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
# IMPORTANT: cmake's default 'all' target does NOT build the kernel binary.
# Use one of:
#   1. bash build.sh --soc=ascend910b
#   2. cmake --build . --target all binary package
# =============================================================================
message(STATUS "flip: target SoC is Ascend910B3 (ascend910b); package=${package_name}")
message(STATUS "flip: cmake 'make' alone does not build the kernel binary. Use build.sh --soc=ascend910b")