# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# Ascend910B3 is an arch22 target (ascend910b). This generated candidate keeps the
# kernel source selection explicit so the kernel entry and package stay aligned.
npu_op_kernel_sources(ascendc_kernels
    OP_TYPE Argmax
    KERNEL_DIR .
    KERNEL_FILE argmax_arch22.cpp
)

# Build the kernel library and link it against the host tiling implementation.
npu_op_kernel_library(ascendc_kernels
    SRC_BASE ${CMAKE_CURRENT_SOURCE_DIR}
    TILING_LIBRARY cust_optiling
)

# Include the generated Argmax kernel binary in the custom operator package.
npu_op_package_add(${package_name}
    LIBRARY
        ascendc_kernels
)