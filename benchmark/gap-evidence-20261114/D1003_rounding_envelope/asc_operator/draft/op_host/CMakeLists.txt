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
# Nextafter host build
#
# Target scope: Ascend910B3 / CANN 9.2.0-beta.2, with float32 inputs and
# float32 outputs for the observed nextafter profiles.  This file deliberately
# keeps one authoritative host source set: operator definition, infer-shape,
# tiling, ACLNN L0/L2, and graph proto.
# =============================================================================

if(NOT DEFINED ENV{ASCEND_HOME_PATH} OR "$ENV{ASCEND_HOME_PATH}" STREQUAL "")
    message(FATAL_ERROR "ASCEND_HOME_PATH must be set before building the Nextafter operator")
endif()

set(NEXTAFTER_CANN_HOME "$ENV{ASCEND_HOME_PATH}")
set(NEXTAFTER_CANN_INC "${NEXTAFTER_CANN_HOME}/aarch64-linux/include")
set(NEXTAFTER_CANN_ASC_INC "${NEXTAFTER_CANN_HOME}/aarch64-linux/asc/include")
set(NEXTAFTER_CANN_TILING_INC "${NEXTAFTER_CANN_HOME}/aarch64-linux/asc/include/tiling")
set(NEXTAFTER_CANN_ACLNN_INC "${NEXTAFTER_CANN_INC}/aclnn")
set(NEXTAFTER_CANN_PKG_INC "${NEXTAFTER_CANN_HOME}/aarch64-linux/pkg_inc")
set(NEXTAFTER_CANN_OP_COMMON_INC "${NEXTAFTER_CANN_PKG_INC}/op_common")
set(NEXTAFTER_CANN_BASE_INC "${NEXTAFTER_CANN_PKG_INC}/base")
set(NEXTAFTER_CANN_EXE_GRAPH_INC "${NEXTAFTER_CANN_PKG_INC}/exe_graph")
set(NEXTAFTER_CANN_GRAPH_INC "${NEXTAFTER_CANN_PKG_INC}/graph")
set(NEXTAFTER_CANN_AICPU_INC "${NEXTAFTER_CANN_PKG_INC}/aicpu")

if(NOT DEFINED package_name OR "${package_name}" STREQUAL "")
    set(package_name nextafter_custom)
endif()

# =============================================================================
# Operator definition and shape/dtype inference
# =============================================================================
set(host_ops_def_srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/nextafter_def.cpp
)

set(host_ops_infershape_srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/nextafter_infershape.cpp
)

foreach(_required_host_file IN LISTS host_ops_def_srcs host_ops_infershape_srcs)
    if(NOT EXISTS "${_required_host_file}")
        message(FATAL_ERROR "Required Nextafter host source is missing: ${_required_host_file}")
    endif()
endforeach()

# =============================================================================
# Tiling source
#
# arch22 covers Ascend910B / Ascend910B3.  The flat fallback keeps this build
# file usable if a later deterministic stage places the single tiling source
# directly under op_host instead of arch22.
# =============================================================================
set(host_ops_tiling_srcs)

if(ASCEND_COMPUTE_UNIT STREQUAL "ascend950")
    set(_nextafter_tiling_candidate
        ${CMAKE_CURRENT_SOURCE_DIR}/arch35/nextafter_tiling.cpp
    )
else()
    set(_nextafter_tiling_candidate
        ${CMAKE_CURRENT_SOURCE_DIR}/arch22/nextafter_tiling.cpp
    )
endif()

if(NOT EXISTS "${_nextafter_tiling_candidate}")
    set(_nextafter_tiling_candidate
        ${CMAKE_CURRENT_SOURCE_DIR}/nextafter_tiling.cpp
    )
endif()

if(NOT EXISTS "${_nextafter_tiling_candidate}")
    message(FATAL_ERROR
        "Required Nextafter tiling source is missing. Expected one of: "
        "${CMAKE_CURRENT_SOURCE_DIR}/arch22/nextafter_tiling.cpp, "
        "${CMAKE_CURRENT_SOURCE_DIR}/arch35/nextafter_tiling.cpp, or "
        "${CMAKE_CURRENT_SOURCE_DIR}/nextafter_tiling.cpp"
    )
endif()

get_filename_component(_nextafter_tiling_dir "${_nextafter_tiling_candidate}" DIRECTORY)
list(APPEND host_ops_tiling_srcs "${_nextafter_tiling_candidate}")

set(host_ops_srcs
    ${host_ops_def_srcs}
    ${host_ops_infershape_srcs}
    ${host_ops_tiling_srcs}
)

# =============================================================================
# Host autogen / registration
# =============================================================================
npu_op_code_gen(
    SRC ${host_ops_srcs}
    PACKAGE ${package_name}
    OUT_DIR ${ASCEND_AUTOGEN_PATH}
    COMPILE_OPTIONS
        -I${NEXTAFTER_CANN_INC}
        -I${NEXTAFTER_CANN_TILING_INC}
        -I${NEXTAFTER_CANN_PKG_INC}
        -I${NEXTAFTER_CANN_OP_COMMON_INC}
        -I${NEXTAFTER_CANN_BASE_INC}
        -I${NEXTAFTER_CANN_EXE_GRAPH_INC}
        -I${NEXTAFTER_CANN_GRAPH_INC}
        -I${CMAKE_CURRENT_SOURCE_DIR}
        -I${_nextafter_tiling_dir}
)

npu_op_library(cust_optiling TILING
    ${host_ops_srcs}
)

target_include_directories(cust_optiling PRIVATE
    ${NEXTAFTER_CANN_INC}
    ${NEXTAFTER_CANN_TILING_INC}
    ${NEXTAFTER_CANN_PKG_INC}
    ${NEXTAFTER_CANN_OP_COMMON_INC}
    ${NEXTAFTER_CANN_BASE_INC}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${_nextafter_tiling_dir}
)

set(SECURE_COMPILE_OPTIONS -fPIC -fstack-protector-all)
set(SECURE_LINK_OPTIONS -Wl,-z,relro -Wl,-z,now -Wl,-z,noexecstack)

target_compile_options(cust_optiling PRIVATE ${SECURE_COMPILE_OPTIONS})
target_link_options(cust_optiling PRIVATE ${SECURE_LINK_OPTIONS})

# =============================================================================
# ACLNN library
#
# L2: aclnn_nextafter.cpp / aclnn_nextafter.h
# L0: nextafter.cpp / nextafter.h
#
# The L0 source is expected to use l0op:: symbols.  CANN's opapi library is
# therefore linked explicitly here, as required for L0 ACLNN implementations.
# =============================================================================
set(op_api_dir ${CMAKE_CURRENT_SOURCE_DIR}/../op_api)

set(op_api_srcs
    ${op_api_dir}/aclnn_nextafter.cpp
    ${op_api_dir}/nextafter.cpp
)

foreach(_required_op_api_file IN LISTS op_api_srcs)
    if(NOT EXISTS "${_required_op_api_file}")
        message(FATAL_ERROR "Required Nextafter op_api source is missing: ${_required_op_api_file}")
    endif()
endforeach()

npu_op_library(cust_opapi ACLNN
    ${op_api_srcs}
)

target_include_directories(cust_opapi PRIVATE
    ${op_api_dir}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/arch22
    ${CMAKE_CURRENT_SOURCE_DIR}/arch35
    ${NEXTAFTER_CANN_INC}
    ${NEXTAFTER_CANN_ACLNN_INC}
    ${NEXTAFTER_CANN_ASC_INC}
    ${NEXTAFTER_CANN_PKG_INC}
    ${NEXTAFTER_CANN_OP_COMMON_INC}
    ${NEXTAFTER_CANN_BASE_INC}
    ${NEXTAFTER_CANN_AICPU_INC}
)

target_compile_options(cust_opapi PRIVATE ${SECURE_COMPILE_OPTIONS})
target_link_options(cust_opapi PRIVATE ${SECURE_LINK_OPTIONS})

# Required when L0 op_api references l0op::Nextafter or other l0op:: symbols.
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
    ${NEXTAFTER_CANN_INC}
    ${NEXTAFTER_CANN_TILING_INC}
    ${NEXTAFTER_CANN_PKG_INC}
    ${NEXTAFTER_CANN_OP_COMMON_INC}
    ${NEXTAFTER_CANN_BASE_INC}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${_nextafter_tiling_dir}
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