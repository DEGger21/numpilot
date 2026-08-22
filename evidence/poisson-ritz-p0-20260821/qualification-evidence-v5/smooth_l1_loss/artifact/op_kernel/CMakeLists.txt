# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms of the
# CANN Open Software License Agreement Version 2.0 (the "License").
# ----------------------------------------------------------------------------------------------------------

# =============================================================================
# Kernel sources - Architecture-specific
# Target SoC: ascend910b (arch22 generation)
# NOTE: internal OpType is digit-free (NumPilotSmoothMeanLoss) so that CANN 9.0
# build-time and runtime CamelCase-to-file-name conversions agree on the
# generated dynamic-kernel config JSON name.
# =============================================================================

if(ASCEND_COMPUTE_UNIT STREQUAL "ascend950")
    npu_op_kernel_sources(ascendc_kernels
        OP_TYPE NumPilotSmoothMeanLoss
        KERNEL_DIR .
        KERNEL_FILE smooth_l1_loss_arch35.cpp
    )
else()
    # ascend910b / ascend910_93 (arch22)
    npu_op_kernel_sources(ascendc_kernels
        OP_TYPE NumPilotSmoothMeanLoss
        KERNEL_DIR .
        KERNEL_FILE smooth_l1_loss_arch22.cpp
    )
endif()

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