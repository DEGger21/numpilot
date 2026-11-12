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
# Tril host build
#
# Scope:
#   C++ OpType: Tril
#   Public API: asnumpy.tril
#   ACLNN entry: aclnnTril
#   Target SoC: Ascend910B3 (ascend910b, arch22)
#   Input dtype: float32 only for this draft
#
# The generated operator is intentionally restricted to the observed and
# declared draft profile. Unsupported dtypes, shapes, signatures, and
# architectures must be rejected explicitly at the host API boundary rather
# than compiled into a broader compatibility claim.
# =============================================================================

if(DEFINED ASCEND_COMPUTE_UNIT
   AND NOT ASCEND_COMPUTE_UNIT STREQUAL ""
   AND NOT ASCEND_COMPUTE_UNIT STREQUAL "ascend910b")
    message(WARNING
        "tril draft is qualified only for Ascend910B3 (ascend910b). "
        "Requested ASCEND_COMPUTE_UNIT='${ASCEND_COMPUTE_UNIT}'.")
endif()

# =============================================================================
# Host source files - Op definition and registration
# =============================================================================
set(host_ops_def_srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/tril_def.cpp
)

# =============================================================================
# Host source files - Shape and dtype inference
# =============================================================================
set(host_ops_infershape_srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/tril_infershape.cpp
)

# =============================================================================
# Host source files - Tiling
#
# This draft targets arch22 only:
#   Ascend910B3 -> ascend910b -> arch22
#
# Do not reference an arch35 tiling source for this operator. The public
# qualification scope is the recorded Ascend910B3 profile only.
# =============================================================================
set(host_ops_tiling_srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/arch22/tril_tiling.cpp
)

set(host_ops_srcs
    ${host_ops_def_srcs}
    ${host_ops_infershape_srcs}
    ${host_ops_tiling_srcs}
)

# =============================================================================
# Generate framework registration artifacts from the hand-written definition.
# =============================================================================
npu_op_code_gen(
    SRC ${host_ops_srcs}
    PACKAGE ${package_name}
    OUT_DIR ${ASCEND_AUTOGEN_PATH}
    COMPILE_OPTIONS
        -I${CMAKE_CURRENT_SOURCE_DIR}
        -I${CMAKE_CURRENT_SOURCE_DIR}/arch22
        -I${CMAKE_CURRENT_SOURCE_DIR}/../op_graph
        -I$ENV{ASCEND_HOME_PATH}/aarch64-linux/include
        -I$ENV{ASCEND_HOME_PATH}/aarch64-linux/asc/include
        -I$ENV{ASCEND_HOME_PATH}/aarch64-linux/asc/include/tiling
        -I$ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc
        -I$ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/op_common
        -I$ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/base
        -I$ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/exe_graph
        -I$ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/graph
)

# =============================================================================
# Tiling library consumed by the kernel build.
# =============================================================================
npu_op_library(cust_optiling TILING
    ${host_ops_srcs}
)

target_include_directories(cust_optiling PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/arch22
    ${CMAKE_CURRENT_SOURCE_DIR}/../op_graph
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/include
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/asc/include
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
# ACLNN API library
#
# L2: aclnnTril / aclnnTrilGetWorkspaceSize
# L0: tril / trilGetWorkspaceSize
# =============================================================================
set(op_api_dir ${CMAKE_CURRENT_SOURCE_DIR}/../op_api)
set(op_api_srcs
    ${op_api_dir}/aclnn_tril.cpp
    ${op_api_dir}/tril.cpp
)

npu_op_library(cust_opapi ACLNN
    ${op_api_srcs}
)

target_include_directories(cust_opapi PRIVATE
    ${op_api_dir}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/arch22
    ${CMAKE_CURRENT_SOURCE_DIR}/../op_graph
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/include
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/include/aclnn
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/asc/include
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/asc/include/tiling
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
# Graph proto library
# =============================================================================
file(GLOB proto_src ${ASCEND_AUTOGEN_PATH}/op_proto.cc)
set_source_files_properties(${proto_src} PROPERTIES GENERATED TRUE)

npu_op_library(cust_op_proto GRAPH
    ${host_ops_srcs}
    ${proto_src}
)

target_include_directories(cust_op_proto PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/arch22
    ${CMAKE_CURRENT_SOURCE_DIR}/../op_graph
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/include
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/asc/include
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/asc/include/tiling
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/op_common
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/base
)

target_compile_options(cust_op_proto PRIVATE ${SECURE_COMPILE_OPTIONS})
target_link_options(cust_op_proto PRIVATE ${SECURE_LINK_OPTIONS})

# =============================================================================
# Add all host libraries to the custom operator package.
# =============================================================================
npu_op_package_add(${package_name}
    LIBRARY
        cust_optiling
        cust_opapi
        cust_op_proto
)