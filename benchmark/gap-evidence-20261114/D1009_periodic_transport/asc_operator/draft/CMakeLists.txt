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
project(roll_op_prj LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(ASC REQUIRED)

# This draft is bounded to the only SoC recorded by the GapSpec: Ascend910B3 -> ascend910b.
# Do not silently broaden the build to other SoCs before their kernels, tiling, and tests exist.
set(ROLL_TARGET_SOC "ascend910b")

if(NOT DEFINED ASCEND_COMPUTE_UNIT OR ASCEND_COMPUTE_UNIT STREQUAL "")
    set(ASCEND_COMPUTE_UNIT "${ROLL_TARGET_SOC}" CACHE STRING "Ascend SoC version for this Roll draft" FORCE)
endif()

string(TOLOWER "${ASCEND_COMPUTE_UNIT}" ASCEND_COMPUTE_UNIT)
if(NOT ASCEND_COMPUTE_UNIT STREQUAL "${ROLL_TARGET_SOC}")
    message(FATAL_ERROR
        "Roll draft supports only ${ROLL_TARGET_SOC}; received ASCEND_COMPUTE_UNIT=${ASCEND_COMPUTE_UNIT}. "
        "Update the GapSpec, kernels, tiling, and qualification tests before enabling another SoC.")
endif()

set(package_name roll_custom)

npu_op_package(${package_name}
    TYPE RUN
    CONFIG
        INSTALL_PATH ${CMAKE_BINARY_DIR}
)

# Host-side operator definition, infer-shape, tiling, ACLNN API, and generated graph glue are
# organized by op_host in the registry-invoke project layout.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/op_host")
    add_subdirectory(op_host)
endif()

# Ascend C kernel sources and binary registration. The kernel sources are selected by op_kernel
# for the configured architecture.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/op_kernel")
    add_subdirectory(op_kernel)
endif()

# Optional graph-mode adaptation. It is built only when the graph stage has generated it.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/op_graph")
    add_subdirectory(op_graph)
endif()

# IMPORTANT: cmake's default 'all' target does NOT build the Ascend C kernel binary.
# Use one of:
#   1. bash build.sh --soc=ascend910b
#   2. cmake --build . --target all binary package
message(STATUS "Roll operator build configured for SoC: ${ASCEND_COMPUTE_UNIT}")
message(STATUS "Roll package name: ${package_name}")
message(WARNING "cmake 'make' does NOT build kernel binary by default. Use: bash build.sh --soc=ascend910b")