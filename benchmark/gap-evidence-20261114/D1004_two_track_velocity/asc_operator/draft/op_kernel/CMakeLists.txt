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
# Diff kernel build definition
#
# The generated operator is deliberately scoped to the observed target:
#   SoC: Ascend910B3
#   CANN compute unit: ascend910b
#   C++ OpType: Diff
#   Kernel stem: diff
#
# Keeping the compute-unit check explicit prevents an unsupported architecture
# from being built silently when this operator has not been qualified for it.
# =============================================================================

if(NOT DEFINED ASCEND_COMPUTE_UNIT OR ASCEND_COMPUTE_UNIT STREQUAL "")
    set(ASCEND_COMPUTE_UNIT ascend910b)
endif()

if(NOT ASCEND_COMPUTE_UNIT STREQUAL "ascend910b")
    message(FATAL_ERROR
        "Diff is defined only for Ascend910B3 (ASCEND_COMPUTE_UNIT=ascend910b); "
        "received ASCEND_COMPUTE_UNIT='${ASCEND_COMPUTE_UNIT}'.")
endif()

set(diff_kernel_file "diff_arch22.cpp")
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${diff_kernel_file}")
    message(FATAL_ERROR
        "Required Diff kernel source was not found: "
        "${CMAKE_CURRENT_SOURCE_DIR}/${diff_kernel_file}")
endif()

# =============================================================================
# Kernel sources
# =============================================================================
npu_op_kernel_sources(ascendc_kernels
    OP_TYPE Diff
    KERNEL_DIR .
    KERNEL_FILE ${diff_kernel_file}
)

# =============================================================================
# Build kernel library
# =============================================================================
npu_op_kernel_library(ascendc_kernels
    SRC_BASE ${CMAKE_CURRENT_SOURCE_DIR}
    TILING_LIBRARY cust_optiling
)

# =============================================================================
# Add kernel library to the operator package
# =============================================================================
npu_op_package_add(${package_name}
    LIBRARY
        ascendc_kernels
)