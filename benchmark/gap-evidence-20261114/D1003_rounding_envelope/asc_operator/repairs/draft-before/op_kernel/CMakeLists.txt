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
# Nextafter kernel sources - architecture-specific
#
# Ascend910B3 (ascend910b) and other arch22 targets use nextafter_arch22.cpp.
# ascend950 uses nextafter_arch35.cpp.  A canonical nextafter.cpp fallback is
# accepted so this build file remains consistent with either generated layout.
# =============================================================================

set(nextafter_kernel_file "")

if(ASCEND_COMPUTE_UNIT STREQUAL "ascend950")
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/nextafter_arch35.cpp")
        set(nextafter_kernel_file "nextafter_arch35.cpp")
    endif()
else()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/nextafter_arch22.cpp")
        set(nextafter_kernel_file "nextafter_arch22.cpp")
    endif()
endif()

if(nextafter_kernel_file STREQUAL "" AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/nextafter.cpp")
    set(nextafter_kernel_file "nextafter.cpp")
endif()

if(nextafter_kernel_file STREQUAL "")
    message(FATAL_ERROR
        "Nextafter kernel source not found under ${CMAKE_CURRENT_SOURCE_DIR}. "
        "Expected nextafter_arch22.cpp for ascend910b/ascend910_93, "
        "nextafter_arch35.cpp for ascend950, or nextafter.cpp.")
endif()

npu_op_kernel_sources(ascendc_kernels
    OP_TYPE Nextafter
    KERNEL_DIR .
    KERNEL_FILE ${nextafter_kernel_file}
)

# =============================================================================
# Build kernel library
# =============================================================================
npu_op_kernel_library(ascendc_kernels
    SRC_BASE ${CMAKE_CURRENT_SOURCE_DIR}
    TILING_LIBRARY cust_optiling
)

# =============================================================================
# Add the Nextafter kernel library to the operator package
# =============================================================================
npu_op_package_add(${package_name}
    LIBRARY
        ascendc_kernels
)