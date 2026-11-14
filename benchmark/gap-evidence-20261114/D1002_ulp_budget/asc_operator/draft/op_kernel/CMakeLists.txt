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
# Spacing kernel sources
#
# The operator is currently qualified only for Ascend910B3, which is selected by
# the CANN build system as the ascend910b compute unit (arch22).
# =============================================================================

if(DEFINED ASCEND_COMPUTE_UNIT)
    if(NOT "${ASCEND_COMPUTE_UNIT}" STREQUAL "")
        if(NOT "${ASCEND_COMPUTE_UNIT}" STREQUAL "ascend910b")
            message(FATAL_ERROR
                "Spacing is qualified only for Ascend910B3 (ASCEND_COMPUTE_UNIT=ascend910b); "
                "got '${ASCEND_COMPUTE_UNIT}'.")
        endif()
    endif()
endif()

set(spacing_kernel_file "")
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/spacing_arch22.cpp")
    set(spacing_kernel_file "spacing_arch22.cpp")
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/spacing.cpp")
    set(spacing_kernel_file "spacing.cpp")
else()
    message(FATAL_ERROR
        "No Spacing kernel source found in ${CMAKE_CURRENT_SOURCE_DIR}. "
        "Expected spacing_arch22.cpp or spacing.cpp.")
endif()

message(STATUS "Spacing kernel source: ${spacing_kernel_file}")

npu_op_kernel_sources(ascendc_kernels
    OP_TYPE Spacing
    KERNEL_DIR .
    KERNEL_FILE ${spacing_kernel_file}
)

# =============================================================================
# Build kernel library
# =============================================================================
npu_op_kernel_library(ascendc_kernels
    SRC_BASE ${CMAKE_CURRENT_SOURCE_DIR}
    TILING_LIBRARY cust_optiling
)

# =============================================================================
# Add to package
# =============================================================================
npu_op_package_add(${package_name}
    LIBRARY
        ascendc_kernels
)