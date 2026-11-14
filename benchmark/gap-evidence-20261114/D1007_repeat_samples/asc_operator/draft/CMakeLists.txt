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

project(repeat_op_prj)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(ASC REQUIRED)

# This draft is deliberately scoped to the observed target:
#   Ascend910B3 / CANN 9.2.0-beta.2 / SoC build token: ascend910b
# Other SoCs remain unsupported until separately qualified.
set(REPEAT_SUPPORTED_COMPUTE_UNIT "ascend910b")

if(NOT DEFINED ASCEND_COMPUTE_UNIT OR ASCEND_COMPUTE_UNIT STREQUAL "")
    set(ASCEND_COMPUTE_UNIT "${REPEAT_SUPPORTED_COMPUTE_UNIT}")
endif()

if(NOT ASCEND_COMPUTE_UNIT STREQUAL "${REPEAT_SUPPORTED_COMPUTE_UNIT}")
    message(FATAL_ERROR
        "Repeat is qualified only for ${REPEAT_SUPPORTED_COMPUTE_UNIT}; "
        "requested ASCEND_COMPUTE_UNIT=${ASCEND_COMPUTE_UNIT}")
endif()

set(package_name repeat_custom)

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

# cmake's default 'all' target does not build the kernel binary.
# Use: cmake --build . --target all binary package
message(STATUS "Repeat project configured for ${ASCEND_COMPUTE_UNIT}")