# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

if(NOT DEFINED package_name OR package_name STREQUAL "")
    message(FATAL_ERROR "package_name must be defined by the top-level CMakeLists.txt before op_host is configured")
endif()

if("$ENV{ASCEND_HOME_PATH}" STREQUAL "")
    message(FATAL_ERROR "ASCEND_HOME_PATH is not set; source the target CANN environment before configuring Repeat")
endif()

set(REPEAT_CANN_HOME "$ENV{ASCEND_HOME_PATH}")
set(REPEAT_HOST_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(REPEAT_OP_API_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../op_api")

set(REPEAT_CANN_INCLUDE_DIRS
    "${REPEAT_CANN_HOME}/aarch64-linux/include"
    "${REPEAT_CANN_HOME}/aarch64-linux/include/aclnn"
    "${REPEAT_CANN_HOME}/aarch64-linux/asc/include"
    "${REPEAT_CANN_HOME}/aarch64-linux/asc/include/tiling"
    "${REPEAT_CANN_HOME}/aarch64-linux/pkg_inc"
    "${REPEAT_CANN_HOME}/aarch64-linux/pkg_inc/op_common"
    "${REPEAT_CANN_HOME}/aarch64-linux/pkg_inc/base"
    "${REPEAT_CANN_HOME}/aarch64-linux/pkg_inc/exe_graph"
    "${REPEAT_CANN_HOME}/aarch64-linux/pkg_inc/graph"
    "${REPEAT_CANN_HOME}/aarch64-linux/pkg_inc/aicpu"
)

set(REPEAT_CANN_LIBRARY_DIRS
    "${REPEAT_CANN_HOME}/aarch64-linux/lib64"
    "${REPEAT_CANN_HOME}/aarch64-linux/lib"
    "${REPEAT_CANN_HOME}/lib64"
    "${REPEAT_CANN_HOME}/lib"
)

# =============================================================================
# Host source files - Op definition and shape/dtype inference
# =============================================================================
set(host_ops_def_srcs
    "${REPEAT_HOST_DIR}/repeat_def.cpp"
)

set(host_ops_infershape_srcs
    "${REPEAT_HOST_DIR}/repeat_infershape.cpp"
)

foreach(_repeat_required_host_source IN LISTS host_ops_def_srcs host_ops_infershape_srcs)
    if(NOT EXISTS "${_repeat_required_host_source}")
        message(FATAL_ERROR "Required Repeat host source is missing: ${_repeat_required_host_source}")
    endif()
endforeach()

# =============================================================================
# Host source files - Tiling
# The target is Ascend910B3 (ascend910b / arch22). Keep the direct layout as a
# fallback so the authoritative tiling definition may live in either the
# conventional arch directory or directly under op_host.
# =============================================================================
set(host_ops_tiling_srcs)
set(REPEAT_TILING_INCLUDE_DIRS)

if(ASCEND_COMPUTE_UNIT MATCHES "ascend950")
    set(_repeat_tiling_candidates
        "${REPEAT_HOST_DIR}/arch35/repeat_tiling.cpp"
        "${REPEAT_HOST_DIR}/repeat_tiling.cpp"
    )
    list(APPEND REPEAT_TILING_INCLUDE_DIRS
        "${REPEAT_HOST_DIR}/arch35"
    )
else()
    set(_repeat_tiling_candidates
        "${REPEAT_HOST_DIR}/arch22/repeat_tiling.cpp"
        "${REPEAT_HOST_DIR}/repeat_tiling.cpp"
    )
    list(APPEND REPEAT_TILING_INCLUDE_DIRS
        "${REPEAT_HOST_DIR}/arch22"
    )
endif()

foreach(_repeat_tiling_candidate IN LISTS _repeat_tiling_candidates)
    if(EXISTS "${_repeat_tiling_candidate}")
        list(APPEND host_ops_tiling_srcs "${_repeat_tiling_candidate}")
        break()
    endif()
endforeach()

if(NOT host_ops_tiling_srcs)
    message(FATAL_ERROR
        "Repeat tiling source was not found. Expected one of: "
        "${_repeat_tiling_candidates}")
endif()

set(REPEAT_HOST_INCLUDE_DIRS
    "${REPEAT_HOST_DIR}"
    "${REPEAT_HOST_DIR}/arch22"
    "${REPEAT_HOST_DIR}/arch35"
    ${REPEAT_TILING_INCLUDE_DIRS}
)
list(REMOVE_DUPLICATES REPEAT_HOST_INCLUDE_DIRS)

set(host_ops_srcs
    ${host_ops_def_srcs}
    ${host_ops_infershape_srcs}
    ${host_ops_tiling_srcs}
)

# =============================================================================
# Generate framework metadata and the graph proto source.
# =============================================================================
set(_repeat_codegen_compile_options)
foreach(_repeat_include_dir IN LISTS REPEAT_CANN_INCLUDE_DIRS REPEAT_HOST_INCLUDE_DIRS)
    list(APPEND _repeat_codegen_compile_options "-I${_repeat_include_dir}")
endforeach()

npu_op_code_gen(
    SRC ${host_ops_srcs}
    PACKAGE ${package_name}
    OUT_DIR ${ASCEND_AUTOGEN_PATH}
    COMPILE_OPTIONS
        ${_repeat_codegen_compile_options}
)

# =============================================================================
# Tiling library: operator definition, infer-shape, and tiling.
# =============================================================================
npu_op_library(cust_optiling TILING
    ${host_ops_srcs}
)

target_include_directories(cust_optiling PRIVATE
    ${REPEAT_HOST_INCLUDE_DIRS}
    ${REPEAT_CANN_INCLUDE_DIRS}
)

set(SECURE_COMPILE_OPTIONS -fPIC -fstack-protector-all)
set(SECURE_LINK_OPTIONS -Wl,-z,relro -Wl,-z,now -Wl,-z,noexecstack)

target_compile_options(cust_optiling PRIVATE ${SECURE_COMPILE_OPTIONS})
target_link_options(cust_optiling PRIVATE ${SECURE_LINK_OPTIONS})

# =============================================================================
# ACLNN library: L2 public ACLNN entry and L0 internal entry.
# =============================================================================
set(op_api_srcs
    "${REPEAT_OP_API_DIR}/aclnn_repeat.cpp"
    "${REPEAT_OP_API_DIR}/repeat.cpp"
)

foreach(_repeat_op_api_source IN LISTS op_api_srcs)
    if(NOT EXISTS "${_repeat_op_api_source}")
        message(FATAL_ERROR "Required Repeat ACLNN source is missing: ${_repeat_op_api_source}")
    endif()
endforeach()

npu_op_library(cust_opapi ACLNN
    ${op_api_srcs}
)

target_include_directories(cust_opapi PRIVATE
    "${REPEAT_OP_API_DIR}"
    ${REPEAT_CANN_INCLUDE_DIRS}
)

target_compile_options(cust_opapi PRIVATE ${SECURE_COMPILE_OPTIONS})
target_link_options(cust_opapi PRIVATE ${SECURE_LINK_OPTIONS})

target_link_directories(cust_opapi PRIVATE
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/lib64
)
target_link_libraries(cust_opapi PRIVATE opapi)

# The L0 implementation may call CANN registry L0 operators through l0op::.
# Link libopapi only when such a symbol is actually referenced by the generated
# Repeat ACLNN sources. Do not add an unconditional dependency.
set(_repeat_opapi_required FALSE)
foreach(_repeat_op_api_source IN LISTS op_api_srcs)
    file(READ "${_repeat_op_api_source}" _repeat_op_api_content)
    string(FIND "${_repeat_op_api_content}" "l0op::" _repeat_l0op_pos)
    if(NOT "${_repeat_l0op_pos}" EQUAL "-1")
        set(_repeat_opapi_required TRUE)
    endif()
endforeach()

if(_repeat_opapi_required)
    find_library(REPEAT_CANN_OPAPI_LIBRARY
        NAMES opapi
        PATHS ${REPEAT_CANN_LIBRARY_DIRS}
        NO_DEFAULT_PATH
    )
    if(NOT REPEAT_CANN_OPAPI_LIBRARY)
        find_library(REPEAT_CANN_OPAPI_LIBRARY NAMES opapi)
    endif()
    if(NOT REPEAT_CANN_OPAPI_LIBRARY)
        message(FATAL_ERROR
            "Repeat L0 ACLNN sources reference l0op:: symbols, but CANN libopapi was not found "
            "under ASCEND_HOME_PATH=${REPEAT_CANN_HOME}")
    endif()

    target_link_directories(cust_opapi PRIVATE ${REPEAT_CANN_LIBRARY_DIRS})
    target_link_libraries(cust_opapi PRIVATE "${REPEAT_CANN_OPAPI_LIBRARY}")
endif()

# =============================================================================
# Graph proto library.
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
    ${REPEAT_HOST_INCLUDE_DIRS}
    ${REPEAT_CANN_INCLUDE_DIRS}
)

target_compile_options(cust_op_proto PRIVATE ${SECURE_COMPILE_OPTIONS})
target_link_options(cust_op_proto PRIVATE ${SECURE_LINK_OPTIONS})

# =============================================================================
# Add host libraries to the custom operator package.
# =============================================================================
npu_op_package_add(${package_name}
    LIBRARY
        cust_optiling
        cust_opapi
        cust_op_proto
)