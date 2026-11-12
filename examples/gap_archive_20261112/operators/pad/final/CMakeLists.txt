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
project(pad_op_prj LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ASC REQUIRED)

# This draft is scoped to the observed Ascend910B3 profile only.
# CANN uses "ascend910b" as the Ascend910B3 compute-unit selector.
set(PAD_SUPPORTED_COMPUTE_UNITS ascend910b)

if(NOT DEFINED ASCEND_COMPUTE_UNIT OR ASCEND_COMPUTE_UNIT STREQUAL "")
    set(ASCEND_COMPUTE_UNIT ${PAD_SUPPORTED_COMPUTE_UNITS})
endif()

foreach(compute_unit IN LISTS ASCEND_COMPUTE_UNIT)
    if(NOT "${compute_unit}" STREQUAL "ascend910b")
        message(FATAL_ERROR
            "Unsupported ASCEND_COMPUTE_UNIT='${compute_unit}'. "
            "This pad draft supports only Ascend910B3 via ascend910b.")
    endif()
endforeach()

set(package_name pad_custom)

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

# CMake's default 'all' target does not build the Ascend C kernel binary.
# Use build.sh, or explicitly build the 'binary' and 'package' targets.
message(STATUS "pad: package_name=${package_name}; ASCEND_COMPUTE_UNIT=${ASCEND_COMPUTE_UNIT}")
message(STATUS "pad: build kernel artifacts with: bash build.sh --soc=ascend910b -j8")