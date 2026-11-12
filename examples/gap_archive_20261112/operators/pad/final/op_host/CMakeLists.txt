# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
#
# Host side build for the custom Pad operator (unverified candidate).
#   OpType : Pad
#   Stem   : pad
#   Target : Ascend910B3  (ascend910b, arch22)  CANN 9.2.0-beta.2
#
# This file only builds the pieces that live in op_host/ plus the hand-written
# ACLNN sources that live in op_api/. Kernel binaries are built by
# op_kernel/CMakeLists.txt; the top level CMakeLists.txt owns package_name.
#
# Supported scope of this candidate (must stay in sync with op_host/pad_def.cpp,
# op_host/pad_infershape.cpp, op_host/arch22/pad_tiling.cpp, op_api/*, op_kernel/*):
#   * input  x       : float32, rank 2, C contiguous
#   * input  paddings: float32, rank 1, shape [4]  (2 * rank, pairs aligned as
#                     before_0, after_0, before_1, after_1)
#   * input  constant: float32, rank 1, shape [1]  (filled when mode == 0)
#   * attr   mode    : int64 scalar, 0 == 'constant' (only admitted value)
#   * output y       : float32, rank 2, C contiguous
# Anything outside this scope is rejected in the ACLNN parameter check; there is
# no implicit cast and no host fallback.
# ----------------------------------------------------------------------------------------------------------

# =============================================================================
# Host source files - Op definition (schema / registration)
# =============================================================================
set(host_ops_def_srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/pad_def.cpp
)

# =============================================================================
# Host source files - Shape / dtype inference
# =============================================================================
set(host_ops_infershape_srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/pad_infershape.cpp
)

# =============================================================================
# Host source files - Tiling registration and tiling data
# Architecture specific. The candidate is qualified only for arch22
# (ascend910b / ascend910_93). Requesting arch35 without an implementation is
# an explicit configure-time error instead of a silent fallback.
# =============================================================================
set(host_ops_tiling_srcs)
set(host_ops_tiling_include_dirs)

if(ASCEND_COMPUTE_UNIT STREQUAL "ascend950")
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/arch35/pad_tiling.cpp")
        list(APPEND host_ops_tiling_srcs
            ${CMAKE_CURRENT_SOURCE_DIR}/arch35/pad_tiling.cpp
        )
        list(APPEND host_ops_tiling_include_dirs
            ${CMAKE_CURRENT_SOURCE_DIR}/arch35
        )
    else()
        message(FATAL_ERROR
            "pad: ASCEND_COMPUTE_UNIT=ascend950 requires op_host/arch35/pad_tiling.cpp. "
            "This candidate only ships the arch22 (ascend910b / Ascend910B3) implementation.")
    endif()
else()
    # Arch32 chips: ascend910b, ascend910_93
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/arch22/pad_tiling.cpp")
        message(FATAL_ERROR
            "pad: missing op_host/arch22/pad_tiling.cpp for ASCEND_COMPUTE_UNIT=${ASCEND_COMPUTE_UNIT}.")
    endif()
    list(APPEND host_ops_tiling_srcs
        ${CMAKE_CURRENT_SOURCE_DIR}/arch22/pad_tiling.cpp
    )
    list(APPEND host_ops_tiling_include_dirs
        ${CMAKE_CURRENT_SOURCE_DIR}/arch22
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
# Run autogen to set framework properties (we still compile hand-written sources)
# =============================================================================
npu_op_code_gen(
    SRC ${host_ops_srcs}
    PACKAGE ${package_name}
    OUT_DIR ${ASCEND_AUTOGEN_PATH}
    COMPILE_OPTIONS 
        -I$ENV{ASCEND_HOME_PATH}/aarch64-linux/include
        -I$ENV{ASCEND_HOME_PATH}/aarch64-linux/asc/include/tiling
        -I$ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc
        -I$ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/op_common
        -I$ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/base
        -I$ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/exe_graph
        -I$ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/graph
)

# =============================================================================
# Tiling library (host side: schema + shape/dtype inference + tiling)
# =============================================================================
npu_op_library(cust_optiling TILING
    ${host_ops_srcs}
)

target_include_directories(cust_optiling PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${host_ops_tiling_include_dirs}
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
#   op_api/aclnn_pad.cpp : L2 entry, aclnnPadGetWorkspaceSize + aclnnPad
#   op_api/pad.cpp       : L0 entry, pad GetWorkspaceSize + pad execute path
# =============================================================================
set(op_api_dir ${CMAKE_CURRENT_SOURCE_DIR}/../op_api)

if(NOT EXISTS "${op_api_dir}/aclnn_pad.cpp")
    message(FATAL_ERROR "pad: missing ${op_api_dir}/aclnn_pad.cpp (ACLNN L2 entry).")
endif()
if(NOT EXISTS "${op_api_dir}/pad.cpp")
    message(FATAL_ERROR "pad: missing ${op_api_dir}/pad.cpp (ACLNN L0 entry).")
endif()

set(op_api_srcs
    ${op_api_dir}/aclnn_pad.cpp
    ${op_api_dir}/pad.cpp
)

npu_op_library(cust_opapi ACLNN
    ${op_api_srcs}
)

target_include_directories(cust_opapi PRIVATE
    ${op_api_dir}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${host_ops_tiling_include_dirs}
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

# The L0 adapter calls l0op::Contiguous / l0op::ViewCopy from CANN's
# libopapi.so.  Link it explicitly so the package's shared-library
# validation and the runtime loader resolve those symbols without relying
# on process-global state.
target_link_directories(cust_opapi PRIVATE
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/lib64
)
target_link_libraries(cust_opapi PRIVATE opapi)

# =============================================================================
# Graph proto library (auto-generated from the op definition)
# =============================================================================
file(GLOB proto_src ${ASCEND_AUTOGEN_PATH}/op_proto.cc)
if(proto_src)
    set_source_files_properties(${proto_src} PROPERTIES GENERATED TRUE)
endif()

npu_op_library(cust_op_proto GRAPH
    ${host_ops_srcs}
    ${proto_src}
)

target_include_directories(cust_op_proto PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${host_ops_tiling_include_dirs}
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