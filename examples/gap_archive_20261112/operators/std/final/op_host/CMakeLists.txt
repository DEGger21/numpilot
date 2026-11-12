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
# Std host sources
#
# OpType: NumpyStd (kernel entry symbol: numpy_std)
# API stem: std
# Public API: asnumpy.std
# Target: Ascend910B3 (ascend910b, arch22)
# =============================================================================

set(host_ops_def_srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/std_def.cpp
)

set(host_ops_infershape_srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/std_infershape.cpp
)

# Ascend910B3 belongs to the ascend910b family and uses the arch22 tiling path.
# Ascend950 remains isolated on the arch35 path so this operator does not
# accidentally compile architecture-incompatible host tiling code.
set(host_ops_tiling_srcs)

if(ASCEND_COMPUTE_UNIT STREQUAL "ascend950")
    list(APPEND host_ops_tiling_srcs
        ${CMAKE_CURRENT_SOURCE_DIR}/arch35/std_tiling.cpp
    )
else()
    list(APPEND host_ops_tiling_srcs
        ${CMAKE_CURRENT_SOURCE_DIR}/arch22/std_tiling.cpp
    )
endif()

set(host_ops_srcs
    ${host_ops_def_srcs}
    ${host_ops_infershape_srcs}
    ${host_ops_tiling_srcs}
)

# =============================================================================
# Generate framework registration sources from the hand-written OpDef,
# InferShape and Tiling implementation.
# =============================================================================
npu_op_code_gen(
    SRC ${host_ops_srcs}
    PACKAGE ${package_name}
    OUT_DIR ${ASCEND_AUTOGEN_PATH}
    COMPILE_OPTIONS
        -I${CMAKE_CURRENT_SOURCE_DIR}
        -I${CMAKE_CURRENT_SOURCE_DIR}/arch22
        -I${CMAKE_CURRENT_SOURCE_DIR}/arch35
        -I$ENV{ASCEND_HOME_PATH}/aarch64-linux/include
        -I$ENV{ASCEND_HOME_PATH}/aarch64-linux/asc/include/tiling
        -I$ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc
        -I$ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/op_common
        -I$ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/base
        -I$ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/exe_graph
        -I$ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/graph
)

# =============================================================================
# Tiling library
# =============================================================================
npu_op_library(cust_optiling TILING
    ${host_ops_srcs}
)

target_include_directories(cust_optiling PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/arch22
    ${CMAKE_CURRENT_SOURCE_DIR}/arch35
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
# ACLNN two-stage API library
#
# L2: aclnn_std.h / aclnn_std.cpp
# L0: std.h / std.cpp
# =============================================================================
set(op_api_dir ${CMAKE_CURRENT_SOURCE_DIR}/../op_api)
set(op_api_srcs
    ${op_api_dir}/aclnn_std.cpp
    ${op_api_dir}/std.cpp
)

npu_op_library(cust_opapi ACLNN
    ${op_api_srcs}
)

target_include_directories(cust_opapi PRIVATE
    ${op_api_dir}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/arch22
    ${CMAKE_CURRENT_SOURCE_DIR}/arch35
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/include
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/include/aclnn
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/asc/include
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/op_common
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/base
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/aicpu
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/exe_graph
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/graph
)

target_compile_options(cust_opapi PRIVATE ${SECURE_COMPILE_OPTIONS})
target_link_options(cust_opapi PRIVATE ${SECURE_LINK_OPTIONS})

# The generated L0 adapter calls l0op::Contiguous / l0op::Std /
# l0op::ViewCopy from CANN's libopapi.so.  Link it explicitly so both the
# package's shared-library validation and the runtime loader resolve the
# symbols without relying on process-global state.
target_link_directories(cust_opapi PRIVATE
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/lib64
)
target_link_libraries(cust_opapi PRIVATE opapi)


# =============================================================================
# Graph proto library
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
    ${CMAKE_CURRENT_SOURCE_DIR}/arch22
    ${CMAKE_CURRENT_SOURCE_DIR}/arch35
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/include
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/asc/include/tiling
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/op_common
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/base
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/exe_graph
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/graph
)

target_compile_options(cust_op_proto PRIVATE ${SECURE_COMPILE_OPTIONS})
target_link_options(cust_op_proto PRIVATE ${SECURE_LINK_OPTIONS})

# =============================================================================
# Package registration
# =============================================================================
npu_op_package_add(${package_name}
    LIBRARY
        cust_optiling
        cust_opapi
        cust_op_proto
)