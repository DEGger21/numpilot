# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# =============================================================================
# Argmax host build
#
# Target: Ascend910B3 (ascend910b, arch22)
# OpType: Argmax
# File/API stem: argmax
#
# This file intentionally keeps one authoritative kernel-facing tiling schema:
#   op_host/arch22/argmax_tiling_data.h
# The host tiling source and kernel entry both consume that header. Do not add
# a second tiling-data schema here or in the kernel.
# =============================================================================

# =============================================================================
# Host source files - Op Definition
# =============================================================================
set(host_ops_def_srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/argmax_def.cpp
)

# =============================================================================
# Host source files - InferShape
# =============================================================================
set(host_ops_infershape_srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/argmax_infershape.cpp
)

# =============================================================================
# Host source files - Tiling (architecture-specific)
# =============================================================================
set(argmax_arch22_tiling_dir ${CMAKE_CURRENT_SOURCE_DIR}/arch22)
set(host_ops_tiling_srcs)

if(ASCEND_COMPUTE_UNIT STREQUAL "ascend950")
    message(FATAL_ERROR
        "Argmax draft supports Ascend910B3 (ascend910b/arch22) only; "
        "arch35 tiling is not generated for this operator.")
else()
    list(APPEND host_ops_tiling_srcs
        ${argmax_arch22_tiling_dir}/argmax_tiling.cpp
    )
endif()

# =============================================================================
# Combine all host sources
# =============================================================================
set(host_ops_srcs
    ${host_ops_def_srcs}
    ${host_ops_infershape_srcs}
    ${host_ops_tiling_srcs}
)

# =============================================================================
# Run autogen to set framework properties (hand-written sources are used)
# =============================================================================
npu_op_code_gen(
    SRC ${host_ops_srcs}
    PACKAGE ${package_name}
    OUT_DIR ${ASCEND_AUTOGEN_PATH}
    COMPILE_OPTIONS
        -I${argmax_arch22_tiling_dir}
        -I$ENV{ASCEND_HOME_PATH}/aarch64-linux/include
        -I$ENV{ASCEND_HOME_PATH}/aarch64-linux/asc/include/tiling
        -I$ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc
        -I$ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/op_common
        -I$ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/base
        -I$ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/exe_graph
        -I$ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/graph
)

npu_op_library(cust_optiling TILING
    ${host_ops_srcs}
)

target_include_directories(cust_optiling
    PUBLIC
        ${argmax_arch22_tiling_dir}
    PRIVATE
        $ENV{ASCEND_HOME_PATH}/aarch64-linux/include
        $ENV{ASCEND_HOME_PATH}/aarch64-linux/asc/include/tiling
        $ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc
        $ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/op_common
        $ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/base
)

set(SECURE_COMPILE_OPTIONS -fPIC -fstack-protector-all)
set(SECURE_LINK_OPTIONS -Wl,-z,relro -Wl,-z,now -Wl,-z,noexecstack)

target_compile_options(cust_optiling PRIVATE ${SECURE_COMPILE_OPTIONS})
target_link_options(cust_optiling PRIVATE ${SECURE_LINK_OPTIONS})

# =============================================================================
# ACLNN library (hand-written)
#
# Public mapping:
#   asnumpy.argmax
#     -> Python wrapper
#     -> pybind native module
#     -> aclnnArgmaxGetWorkspaceSize / aclnnArgmax
#     -> l0 argmax GetWorkspaceSize / execute
# =============================================================================
set(op_api_dir ${CMAKE_CURRENT_SOURCE_DIR}/../op_api)
set(op_api_srcs
    ${op_api_dir}/aclnn_argmax.cpp
    ${op_api_dir}/argmax.cpp
)

npu_op_library(cust_opapi ACLNN
    ${op_api_srcs}
)

target_include_directories(cust_opapi
    PRIVATE
        ${op_api_dir}
        ${argmax_arch22_tiling_dir}
        $ENV{ASCEND_HOME_PATH}/aarch64-linux/include
        $ENV{ASCEND_HOME_PATH}/aarch64-linux/include/aclnn
        $ENV{ASCEND_HOME_PATH}/aarch64-linux/asc/include
        $ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc
        $ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/op_common
        $ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/base
        $ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/aicpu
)

target_compile_options(cust_opapi PRIVATE ${SECURE_COMPILE_OPTIONS})
target_link_options(cust_opapi PRIVATE ${SECURE_LINK_OPTIONS})

target_link_directories(cust_opapi PRIVATE
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/lib64
)
target_link_libraries(cust_opapi PRIVATE opapi)

# =============================================================================
# Optional CANN opapi linkage for L0 sources that use l0op:: symbols
#
# The L0 argmax implementation may call CANN's l0op:: dispatch helpers. When
# that is true, the l0op:: symbols are resolved from CANN's opapi library.
# Scan the actual op_api sources so the build only links opapi when required.
# =============================================================================
set(argmax_l0_uses_l0op FALSE)
foreach(argmax_api_src ${op_api_srcs})
    if(EXISTS "${argmax_api_src}")
        file(READ "${argmax_api_src}" argmax_api_src_content)
        string(FIND "${argmax_api_src_content}" "l0op::" argmax_l0op_pos)
        if(NOT argmax_l0op_pos EQUAL -1)
            set(argmax_l0_uses_l0op TRUE)
        endif()
    endif()
endforeach()

if(argmax_l0_uses_l0op)
    find_library(CANN_OPAPI_LIBRARY
        NAMES opapi
        HINTS
            $ENV{ASCEND_HOME_PATH}/aarch64-linux/lib64
            $ENV{ASCEND_HOME_PATH}/aarch64-linux/lib
            $ENV{ASCEND_HOME_PATH}/lib64
            $ENV{ASCEND_HOME_PATH}/lib
    )
    if(NOT CANN_OPAPI_LIBRARY)
        message(FATAL_ERROR
            "Argmax op_api sources reference l0op:: symbols, but CANN's opapi "
            "library was not found under ASCEND_HOME_PATH. Set ASCEND_HOME_PATH "
            "to the CANN 9.2.0-beta.2 installation for Ascend910B3.")
    endif()
    target_link_libraries(cust_opapi PRIVATE ${CANN_OPAPI_LIBRARY})
    message(STATUS "Argmax: linking CANN opapi library ${CANN_OPAPI_LIBRARY} into cust_opapi")
else()
    message(STATUS "Argmax: no l0op:: references detected in op_api sources; opapi link not required")
endif()

# =============================================================================
# Graph proto library (auto-generated)
# =============================================================================
file(GLOB proto_src ${ASCEND_AUTOGEN_PATH}/op_proto.cc)
set_source_files_properties(${proto_src} PROPERTIES GENERATED TRUE)

npu_op_library(cust_op_proto GRAPH
    ${host_ops_srcs}
    ${proto_src}
)

target_include_directories(cust_op_proto
    PRIVATE
        ${argmax_arch22_tiling_dir}
        $ENV{ASCEND_HOME_PATH}/aarch64-linux/include
        $ENV{ASCEND_HOME_PATH}/aarch64-linux/asc/include/tiling
        $ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc
        $ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/op_common
        $ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/base
)

target_compile_options(cust_op_proto PRIVATE ${SECURE_COMPILE_OPTIONS})
target_link_options(cust_op_proto PRIVATE ${SECURE_LINK_OPTIONS})

# =============================================================================
# Add libraries to package
# =============================================================================
npu_op_package_add(${package_name}
    LIBRARY
        cust_optiling
        cust_opapi
        cust_op_proto
)