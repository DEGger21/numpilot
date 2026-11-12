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
# Host-side build for the Ediff1d operator (OpType Ediff1d, file/API stem ediff1d).
#
# This file is deliberately self-contained and consistent with the other deterministic
# generation stages:
#   op_host/ediff1d_def.cpp                 - OpDef / schema (1 input, 1 output, float32)
#   op_host/ediff1d_infershape.cpp          - shape/dtype inference for ediff1d
#   op_host/arch22/ediff1d_tiling.cpp       - tiling for arch22 (Ascend910B3)
#   op_host/arch22/ediff1d_tiling_data.h    - tiling data shared with the kernel
#   op_host/arch22/ediff1d_tiling_key.h     - tiling key definitions
#   op_api/aclnn_ediff1d.cpp                - L2 ACLNN two-stage API
#   op_api/ediff1d.cpp                      - L0 ACLNN API
#
# Target SoC for this generation is Ascend910B3 (arch22, ASCEND_COMPUTE_UNIT=ascend910b).
# No additional architecture is claimed as supported here.
# ----------------------------------------------------------------------------------------------------------

# =============================================================================
# Host source files - Op Definition (schema)
# =============================================================================
set(host_ops_def_srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/ediff1d_def.cpp
)

# =============================================================================
# Host source files - InferShape / InferDataType
# =============================================================================
set(host_ops_infershape_srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/ediff1d_infershape.cpp
)

# =============================================================================
# Host source files - Tiling (architecture-specific)
#
# Ascend910B3 belongs to arch22 (ascend910b, ascend910_93).
# The arch35 branch is kept only as an explicit, existence-checked hook so that
# this file never references sources that were not generated for this project.
# =============================================================================
set(host_ops_tiling_srcs)

if(ASCEND_COMPUTE_UNIT STREQUAL "ascend950")
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/arch35/ediff1d_tiling.cpp")
        list(APPEND host_ops_tiling_srcs
            ${CMAKE_CURRENT_SOURCE_DIR}/arch35/ediff1d_tiling.cpp
        )
    endif()
endif()

if(NOT host_ops_tiling_srcs)
    # Default and the only qualified path for this generation: arch22 / Ascend910B3.
    list(APPEND host_ops_tiling_srcs
        ${CMAKE_CURRENT_SOURCE_DIR}/arch22/ediff1d_tiling.cpp
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
# Tiling library (cust_optiling)
# =============================================================================
npu_op_library(cust_optiling TILING
    ${host_ops_srcs}
)

target_include_directories(cust_optiling PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
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
# ACLNN library (hand-written L0 + L2 two-stage interface)
#
# Public API mapping enforced by these sources:
#   asnumpy.ediff1d  -> Python wrapper -> pybind -> native adapter
#                    -> aclnnEdiff1dGetWorkspaceSize / aclnnEdiff1d (acl_ediff1d)
#                    -> l0op Ediff1d entry -> Ascend C kernel Ediff1d
# =============================================================================
set(op_api_dir ${CMAKE_CURRENT_SOURCE_DIR}/../op_api)
set(op_api_srcs
    ${op_api_dir}/aclnn_ediff1d.cpp
    ${op_api_dir}/ediff1d.cpp
)

npu_op_library(cust_opapi ACLNN
    ${op_api_srcs}
)

target_include_directories(cust_opapi PRIVATE
    ${op_api_dir}
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

# The L0/L2 implementation calls into the tiling library symbols.
target_link_libraries(cust_opapi PRIVATE cust_optiling)

# The L2 layer calls l0op::Contiguous / l0op::ViewCopy from CANN's libopapi.so.
target_link_directories(cust_opapi PRIVATE
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/lib64
)
target_link_libraries(cust_opapi PRIVATE opapi)

# =============================================================================
# Graph proto library (auto-generated from the Ediff1d OpDef)
# =============================================================================
file(GLOB proto_src ${ASCEND_AUTOGEN_PATH}/op_proto.cc)
set_source_files_properties(${proto_src} PROPERTIES GENERATED TRUE)

npu_op_library(cust_op_proto GRAPH
    ${host_ops_srcs}
    ${proto_src}
)

target_include_directories(cust_op_proto PRIVATE
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/include
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/asc/include/tiling
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/op_common
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/base
)

target_compile_options(cust_op_proto PRIVATE ${SECURE_COMPILE_OPTIONS})
target_link_options(cust_op_proto PRIVATE ${SECURE_LINK_OPTIONS})

# =============================================================================
# Add libraries to the operator package
# =============================================================================
npu_op_package_add(${package_name}
    LIBRARY
        cust_optiling
        cust_opapi
        cust_op_proto
)