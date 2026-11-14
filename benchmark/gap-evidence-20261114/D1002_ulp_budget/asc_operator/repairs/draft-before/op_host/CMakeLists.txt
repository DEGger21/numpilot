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
# Spacing host build
#
# Operator: Spacing
# Public API: asnumpy.spacing
# Target SoC: Ascend910B3 (ascend910b, arch22)
# Target CANN: 9.2.0-beta.2
#
# This is an unverified generated draft. It must not be installed or activated
# until all required build, unit, numerical, native-execution, whole-program,
# performance, and activation gates have passed.
# =============================================================================

if(NOT DEFINED package_name OR package_name STREQUAL "")
    set(package_name spacing_custom)
endif()

# =============================================================================
# Architecture selection
#
# The generated Spacing draft is specialized for Ascend910B3 / arch22.
# Ascend910_93 also uses the arch22 source layout. Ascend950 / arch35 is not
# part of this generated support scope and must be rejected explicitly rather
# than silently compiled against a missing or unrelated kernel.
# =============================================================================
list(FIND ASCEND_COMPUTE_UNIT "ascend950" spacing_arch35_index)
if(NOT spacing_arch35_index EQUAL -1)
    message(FATAL_ERROR
        "Spacing generated draft supports arch22 targets only "
        "(ascend910b / ascend910_93). Received ASCEND_COMPUTE_UNIT='${ASCEND_COMPUTE_UNIT}'."
    )
endif()

# =============================================================================
# Host source files - Op definition
# =============================================================================
set(host_ops_def_srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/spacing_def.cpp
)

# =============================================================================
# Host source files - Shape and dtype inference
# =============================================================================
set(host_ops_infershape_srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/spacing_infershape.cpp
)

# =============================================================================
# Host source files - Tiling
#
# Ascend910B3 is an arch22 target. Tiling data and tiling key headers are
# included by the tiling implementation and kernel translation units; only the
# tiling implementation is compiled as a host source here.
# =============================================================================
set(host_ops_tiling_srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/arch22/spacing_tiling.cpp
)

# =============================================================================
# Combine all host sources
# =============================================================================
set(host_ops_srcs
    ${host_ops_def_srcs}
    ${host_ops_infershape_srcs}
    ${host_ops_tiling_srcs}
)

foreach(spacing_host_src IN LISTS host_ops_srcs)
    if(NOT EXISTS "${spacing_host_src}")
        message(FATAL_ERROR "Required Spacing host source is missing: ${spacing_host_src}")
    endif()
endforeach()

# =============================================================================
# Generate registration/proto sources from the hand-written OpDef.
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
# Tiling host library
# =============================================================================
npu_op_library(cust_optiling TILING
    ${host_ops_srcs}
)

target_include_directories(cust_optiling PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/arch22
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
# ACLNN host library
#
# L2 entry: aclnn_spacing.cpp / aclnn_spacing.h
# L0 entry: spacing.cpp / spacing.h
# =============================================================================
set(op_api_dir ${CMAKE_CURRENT_SOURCE_DIR}/../op_api)
set(op_api_srcs
    ${op_api_dir}/aclnn_spacing.cpp
    ${op_api_dir}/spacing.cpp
)

foreach(spacing_op_api_src IN LISTS op_api_srcs)
    if(NOT EXISTS "${spacing_op_api_src}")
        message(FATAL_ERROR "Required Spacing ACLNN source is missing: ${spacing_op_api_src}")
    endif()
endforeach()

npu_op_library(cust_opapi ACLNN
    ${op_api_srcs}
)

target_include_directories(cust_opapi PRIVATE
    ${op_api_dir}
    ${CMAKE_CURRENT_SOURCE_DIR}
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

# =============================================================================
# Graph proto library
#
# op_proto.cc is generated by npu_op_code_gen above from spacing_def.cpp.
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
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/include
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/asc/include/tiling
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/op_common
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/base
)

target_compile_options(cust_op_proto PRIVATE ${SECURE_COMPILE_OPTIONS})
target_link_options(cust_op_proto PRIVATE ${SECURE_LINK_OPTIONS})

# =============================================================================
# Add host libraries to the operator package
# =============================================================================
npu_op_package_add(${package_name}
    LIBRARY
        cust_optiling
        cust_opapi
        cust_op_proto
)