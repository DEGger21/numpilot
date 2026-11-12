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
project(ediff1d_op_prj)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ASC REQUIRED)

# =============================================================================
# Target configuration
# Ascend910B3 maps to the ascend910b compute unit. This draft is intentionally
# restricted to the SoC recorded by the GapSpec; other SoCs are rejected here
# rather than being built without numerical/device qualification.
# =============================================================================
set(EDIF1D_TARGET_SOC "ascend910b")
set(EDIF1D_CANN_VERSION "9.2.0-beta.2")

set(ARCH32_COMPUTE_UNITS ascend910b)
set(ARCH35_COMPUTE_UNITS "")

if(NOT DEFINED ASCEND_COMPUTE_UNIT OR ASCEND_COMPUTE_UNIT STREQUAL "")
    set(ASCEND_COMPUTE_UNIT ${EDIF1D_TARGET_SOC})
endif()

foreach(_ediff1d_soc IN LISTS ASCEND_COMPUTE_UNIT)
    if(NOT _ediff1d_soc STREQUAL "ascend910b")
        message(FATAL_ERROR
            "Ediff1d draft supports only Ascend910B3/ascend910b for CANN ${EDIF1D_CANN_VERSION}; "
            "received ASCEND_COMPUTE_UNIT=${_ediff1d_soc}")
    endif()
endforeach()

set(package_name ediff1d_custom)

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
# Use the project build script for kernel/package targets:
#   bash build.sh --soc=ascend910b
# or explicitly:
#   cmake --build . --target all binary package
# =============================================================================
message(STATUS "Ediff1d package: ${package_name}")
message(STATUS "Ediff1d target SoC: ${EDIF1D_TARGET_SOC} (CANN ${EDIF1D_CANN_VERSION})")
message(WARNING "cmake 'make' does NOT build the kernel binary by default. Use: bash build.sh --soc=ascend910b")