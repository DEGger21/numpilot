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
# Roll kernel build (Ascend910B3 / arch22 only)
#
# The operator support scope for this draft is fixed to Ascend910B3, which is
# built under the CANN compute-unit name "ascend910b". Other compute units are
# rejected explicitly so an accidental package cannot silently omit or replace
# the Roll kernel.
# =============================================================================

if(NOT DEFINED ASCEND_COMPUTE_UNIT OR "${ASCEND_COMPUTE_UNIT}" STREQUAL "")
    message(FATAL_ERROR
        "ASCEND_COMPUTE_UNIT must be set to ascend910b when building the Roll operator.")
endif()

if(NOT "${ASCEND_COMPUTE_UNIT}" STREQUAL "ascend910b")
    message(FATAL_ERROR
        "Roll supports only Ascend910B3 (ASCEND_COMPUTE_UNIT=ascend910b); "
        "received '${ASCEND_COMPUTE_UNIT}'.")
endif()

# Keep the arch22 kernel filename aligned with the official registry-invoke
# template. A plain roll.cpp fallback is accepted for the single-architecture
# project layout, but the selected source must still be the Roll kernel.
set(ROLL_KERNEL_FILE "")
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/roll_arch22.cpp")
    set(ROLL_KERNEL_FILE "roll_arch22.cpp")
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/roll.cpp")
    set(ROLL_KERNEL_FILE "roll.cpp")
else()
    message(FATAL_ERROR
        "Roll kernel source not found in ${CMAKE_CURRENT_SOURCE_DIR}; "
        "expected roll_arch22.cpp or roll.cpp.")
endif()

# =============================================================================
# Kernel sources
# OP_TYPE must match the registered C++ OpType exactly: Roll
# =============================================================================
npu_op_kernel_sources(ascendc_kernels
    OP_TYPE Roll
    KERNEL_DIR .
    KERNEL_FILE ${ROLL_KERNEL_FILE}
)

# =============================================================================
# Build kernel library, linking against the host tiling implementation
# =============================================================================
npu_op_kernel_library(ascendc_kernels
    SRC_BASE ${CMAKE_CURRENT_SOURCE_DIR}
    TILING_LIBRARY cust_optiling
)

# =============================================================================
# Add the Roll kernel library to the operator package
# =============================================================================
npu_op_package_add(${package_name}
    LIBRARY
        ascendc_kernels
)