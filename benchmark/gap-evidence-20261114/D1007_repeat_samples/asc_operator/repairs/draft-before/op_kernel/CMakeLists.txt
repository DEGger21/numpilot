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
# Kernel sources - architecture-specific
#
# Target: Ascend910B3
# SoC build selector: ascend910b (arch22)
#
# The Repeat operator is generated and qualified only for Ascend910B3. Reject
# other compute units explicitly instead of silently building an unsupported
# kernel variant.
# =============================================================================

if(NOT DEFINED ASCEND_COMPUTE_UNIT OR ASCEND_COMPUTE_UNIT STREQUAL "")
    message(FATAL_ERROR
        "ASCEND_COMPUTE_UNIT must be specified for the repeat operator "
        "(expected: ascend910b for Ascend910B3)."
    )
endif()

if(NOT ASCEND_COMPUTE_UNIT STREQUAL "ascend910b")
    message(FATAL_ERROR
        "The repeat operator supports only ASCEND_COMPUTE_UNIT=ascend910b "
        "(Ascend910B3). Received: '${ASCEND_COMPUTE_UNIT}'."
    )
endif()

npu_op_kernel_sources(ascendc_kernels
    OP_TYPE Repeat
    KERNEL_DIR .
    KERNEL_FILE repeat_arch22.cpp
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