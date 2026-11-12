# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# Tril is generated for Ascend910B3, which maps to the ascend910b/arch22 kernel build.
if(NOT DEFINED ASCEND_COMPUTE_UNIT OR ASCEND_COMPUTE_UNIT STREQUAL "")
    message(FATAL_ERROR "ASCEND_COMPUTE_UNIT must be set by the top-level build for tril")
endif()

if(NOT ASCEND_COMPUTE_UNIT STREQUAL "ascend910b")
    message(FATAL_ERROR
        "tril draft supports only Ascend910B3 (ASCEND_COMPUTE_UNIT=ascend910b); "
        "received '${ASCEND_COMPUTE_UNIT}'"
    )
endif()

set(tril_kernel_file "tril_arch22.cpp")
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${tril_kernel_file}")
    message(FATAL_ERROR
        "Missing tril kernel source: ${CMAKE_CURRENT_SOURCE_DIR}/${tril_kernel_file}"
    )
endif()

npu_op_kernel_sources(ascendc_kernels
    OP_TYPE Tril
    KERNEL_DIR .
    KERNEL_FILE ${tril_kernel_file}
)

npu_op_kernel_library(ascendc_kernels
    SRC_BASE ${CMAKE_CURRENT_SOURCE_DIR}
    TILING_LIBRARY cust_optiling
)

npu_op_package_add(${package_name}
    LIBRARY
        ascendc_kernels
)