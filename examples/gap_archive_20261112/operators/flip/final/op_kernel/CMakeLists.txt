# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
#
# Flip kernel build configuration.
#
# This candidate is scoped to Ascend910B3, which uses the arch22 kernel path.
# The C++ OpType is Flip and the kernel source stem is flip.
# Do not add a general arch35 fallback here until that profile is explicitly
# implemented and qualified.
# ----------------------------------------------------------------------------------------------------------

# The generated kernel is arch22-only for the qualified target (Ascend910B3).
# Fail explicitly if the build is configured for an unrelated architecture.
if(ASCEND_COMPUTE_UNIT MATCHES "ascend950")
    message(FATAL_ERROR
        "flip is implemented for Ascend910B3 (ascend910b/arch22) only; "
        "ASCEND_COMPUTE_UNIT='${ASCEND_COMPUTE_UNIT}' is unsupported")
endif()

npu_op_kernel_sources(ascendc_kernels
    OP_TYPE Flip
    KERNEL_DIR .
    KERNEL_FILE flip_arch22.cpp
)

# Link the kernel against the host tiling implementation so the generated
# Flip kernel and tiling data stay consistent.
npu_op_kernel_library(ascendc_kernels
    SRC_BASE ${CMAKE_CURRENT_SOURCE_DIR}
    TILING_LIBRARY cust_optiling
)

# Add the Flip kernel library to the custom operator package.
npu_op_package_add(${package_name}
    LIBRARY
        ascendc_kernels
)