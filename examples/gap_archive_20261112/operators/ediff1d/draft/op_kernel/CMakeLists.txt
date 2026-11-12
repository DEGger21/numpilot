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
# Ediff1d kernel sources
#
# Target SoC for this draft: Ascend910B3 -> CANN compute unit "ascend910b"
# arch22 kernel entry: ediff1d_arch22.cpp
# arch35 kernel entry: ediff1d_arch35.cpp
# =============================================================================

if(ASCEND_COMPUTE_UNIT STREQUAL "ascend950")
    set(EDIFF1D_KERNEL_FILE "ediff1d_arch35.cpp")
else()
    # Ascend910B3 and the arch22 family (ascend910b/ascend910_93).
    set(EDIFF1D_KERNEL_FILE "ediff1d_arch22.cpp")
endif()

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${EDIFF1D_KERNEL_FILE}")
    message(FATAL_ERROR
        "Ediff1d kernel source not found: ${CMAKE_CURRENT_SOURCE_DIR}/${EDIFF1D_KERNEL_FILE}. "
        "The selected compute unit is '${ASCEND_COMPUTE_UNIT}'."
    )
endif()

npu_op_kernel_sources(ascendc_kernels
    OP_TYPE Ediff1d
    KERNEL_DIR .
    KERNEL_FILE ${EDIFF1D_KERNEL_FILE}
)

npu_op_kernel_library(ascendc_kernels
    SRC_BASE ${CMAKE_CURRENT_SOURCE_DIR}
    TILING_LIBRARY cust_optiling
)

npu_op_package_add(${package_name}
    LIBRARY
        ascendc_kernels
)