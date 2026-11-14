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
project(nextafter_op_prj LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(ASC REQUIRED)

# The verified generation target is Ascend910B3, which maps to the ascend910b
# compute-unit name used by the CANN operator build system.
set(ASCEND_SUPPORTED_COMPUTE_UNITS "ascend910b")

if(NOT DEFINED ASCEND_COMPUTE_UNIT OR ASCEND_COMPUTE_UNIT STREQUAL "")
    set(ASCEND_COMPUTE_UNIT "ascend910b" CACHE STRING "Ascend SoC version to build for")
endif()

string(TOLOWER "${ASCEND_COMPUTE_UNIT}" ASCEND_COMPUTE_UNIT)

list(FIND ASCEND_SUPPORTED_COMPUTE_UNITS "${ASCEND_COMPUTE_UNIT}" ASCEND_COMPUTE_UNIT_INDEX)
if(ASCEND_COMPUTE_UNIT_INDEX EQUAL -1)
    message(FATAL_ERROR
        "Unsupported ASCEND_COMPUTE_UNIT='${ASCEND_COMPUTE_UNIT}'. "
        "This Nextafter candidate targets only ascend910b (Ascend910B3).")
endif()

set(ASCEND_COMPUTE_UNIT "${ASCEND_COMPUTE_UNIT}" CACHE STRING "Ascend SoC version to build for" FORCE)

set(package_name nextafter_custom)

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

# cmake's default 'all' target does not build the kernel binary. The operator
# package must be produced with a target list that includes binary and package.
message(STATUS "Nextafter operator package: ${package_name}")
message(STATUS "Target Ascend compute unit: ${ASCEND_COMPUTE_UNIT}")
message(WARNING "cmake 'make' does not build kernel binary by default. Use: bash build.sh --soc=ascend910b")