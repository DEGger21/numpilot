# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# Host-side build for the Flip custom operator.
#
# Target scope:
#   - NPU: Ascend910B3 (ASCEND_COMPUTE_UNIT=ascend910b)
#   - CANN: 9.2.0-beta.2
#   - Public dtype admitted by this candidate: float32 input, float32 output
#
# The NumPy oracle semantics live in op_api. op_host owns the operator schema,
# infer-shape, and tiling. Kernel compilation is handled by op_kernel/CMakeLists.txt.

if(NOT DEFINED package_name OR package_name STREQUAL "")
    set(package_name flip_custom)
endif()

if(NOT DEFINED ASCEND_COMPUTE_UNIT OR ASCEND_COMPUTE_UNIT STREQUAL "")
    set(ASCEND_COMPUTE_UNIT "ascend910b")
endif()

# The build entry point normally passes one compute unit. Handle a list defensively
# by selecting its first entry, and reject unsupported generations explicitly.
set(FLIP_COMPUTE_UNIT "${ASCEND_COMPUTE_UNIT}")
if(FLIP_COMPUTE_UNIT MATCHES ";")
    list(GET FLIP_COMPUTE_UNIT 0 FLIP_COMPUTE_UNIT_FIRST)
    set(FLIP_COMPUTE_UNIT "${FLIP_COMPUTE_UNIT_FIRST}")
endif()
string(TOLOWER "${FLIP_COMPUTE_UNIT}" FLIP_COMPUTE_UNIT)

if(FLIP_COMPUTE_UNIT STREQUAL "ascend910b" OR FLIP_COMPUTE_UNIT STREQUAL "ascend910_93")
    # arch22: Ascend910B family. Flip is scoped to Ascend910B3.
    set(FLIP_TILING_ARCH_DIR "${CMAKE_CURRENT_SOURCE_DIR}/arch22")
    set(FLIP_TILING_SRC "${FLIP_TILING_ARCH_DIR}/flip_tiling.cpp")
elseif(FLIP_COMPUTE_UNIT STREQUAL "ascend950")
    message(FATAL_ERROR
        "flip is scoped to Ascend910B3 (ascend910b). "
        "ascend950 is outside this candidate's declared support scope."
    )
else()
    message(FATAL_ERROR
        "Unsupported ASCEND_COMPUTE_UNIT='${ASCEND_COMPUTE_UNIT}' for flip. "
        "Supported target: Ascend910B3 (ascend910b)."
    )
endif()

# =============================================================================
# Host source files - operator definition and shape inference
# =============================================================================
set(host_ops_def_srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/flip_def.cpp
)

set(host_ops_infershape_srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/flip_infershape.cpp
)

# =============================================================================
# Host source files - tiling
# =============================================================================
set(host_ops_tiling_srcs
    ${FLIP_TILING_SRC}
)

set(host_ops_srcs
    ${host_ops_def_srcs}
    ${host_ops_infershape_srcs}
    ${host_ops_tiling_srcs}
)

# =============================================================================
# ACLNN source files - L2 public ACLNN and L0 native implementation
# =============================================================================
set(op_api_dir ${CMAKE_CURRENT_SOURCE_DIR}/../op_api)
set(op_api_srcs
    ${op_api_dir}/aclnn_flip.cpp
    ${op_api_dir}/flip.cpp
)

set(flip_required_srcs
    ${host_ops_srcs}
    ${op_api_srcs}
)

foreach(required_file IN LISTS flip_required_srcs)
    if(NOT EXISTS "${required_file}")
        message(FATAL_ERROR "Flip build requires source file: ${required_file}")
    endif()
endforeach()

if(NOT EXISTS "${FLIP_TILING_ARCH_DIR}/flip_tiling_data.h")
    message(FATAL_ERROR
        "Flip build requires tiling data header: "
        "${FLIP_TILING_ARCH_DIR}/flip_tiling_data.h"
    )
endif()

# =============================================================================
# Run autogen to set framework properties for the operator definition.
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
        -I${FLIP_TILING_ARCH_DIR}
)

# =============================================================================
# Tiling library
# =============================================================================
npu_op_library(cust_optiling TILING
    ${host_ops_srcs}
)

target_include_directories(cust_optiling PRIVATE
    ${FLIP_TILING_ARCH_DIR}
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
# ACLNN library
# =============================================================================
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

# The L0 adapter calls l0op::Flip / l0op::Contiguous / l0op::ViewCopy from
# CANN's libopapi.so.  Link it explicitly so the package's shared-library
# validation and the runtime loader resolve those symbols without relying on
# process-global state.
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
    ${FLIP_TILING_ARCH_DIR}
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