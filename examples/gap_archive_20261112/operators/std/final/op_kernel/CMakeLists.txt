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
# op_kernel/CMakeLists.txt — Ascend C kernel build for the `std` operator
#
#   C++ OpType     : NumpyStd (kernel entry symbol: numpy_std)
#   File/API stem  : std
#   Public API     : asnumpy.std
#   ACLNN API      : aclnnStd / aclnnStdGetWorkspaceSize
#   Target SoC     : Ascend910B3  (ASCEND_COMPUTE_UNIT=ascend910b, arch22)
#   Target CANN    : 9.2.0-beta.2
#
# Kernel entry argument order (fixed, implemented in the kernel source):
#     inputs..., outputs..., workspace, tiling
# i.e. for this operator:
#     __global__ __aicore__ void std(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
#
# Tiling data structures are owned by op_host/arch22 and consumed through the
# `cust_optiling` library linked below; the kernel side must not redefine them.
# =============================================================================

# -----------------------------------------------------------------------------
# Scope guard.
#
# The draft candidate is specialised for the single observed profile
# (float32 [2048, 2048], axis=1, C-contiguous) on Ascend910B3. Other SoCs are
# rejected explicitly instead of silently reusing the Arch22 kernel, so that no
# untested qualification scope is implied by the build system.
# -----------------------------------------------------------------------------
if(NOT DEFINED ASCEND_COMPUTE_UNIT OR ASCEND_COMPUTE_UNIT STREQUAL "")
    message(FATAL_ERROR
        "ASCEND_COMPUTE_UNIT is not set. The `std` operator draft supports only "
        "Ascend910B3; configure with -DASCEND_COMPUTE_UNIT=ascend910b "
        "(e.g. bash build.sh --soc=ascend910b).")
endif()

if(NOT ASCEND_COMPUTE_UNIT STREQUAL "ascend910b")
    message(FATAL_ERROR
        "The `std` operator draft supports only Ascend910B3 "
        "(ASCEND_COMPUTE_UNIT=ascend910b), but '${ASCEND_COMPUTE_UNIT}' was requested. "
        "Out-of-scope SoCs are rejected explicitly; they are not covered by any "
        "measured profile and must not inherit the Arch22 kernel.")
endif()

# -----------------------------------------------------------------------------
# Kernel sources — Arch22 (ascend910b / Ascend910B3).
#
# The kernel implementation file name is resolved from the source tree so that
# the build stays consistent with the kernel delivery stage. Exactly one kernel
# implementation must exist; anything else is a hard configuration error.
# -----------------------------------------------------------------------------
set(std_kernel_file "")
foreach(std_kernel_candidate std_arch22.cpp std.cpp)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${std_kernel_candidate}")
        set(std_kernel_file "${std_kernel_candidate}")
        break()
    endif()
endforeach()

if(std_kernel_file STREQUAL "")
    message(FATAL_ERROR
        "No kernel implementation found in ${CMAKE_CURRENT_SOURCE_DIR}. "
        "Expected `std_arch22.cpp` (arch22 entry for Ascend910B3) or `std.cpp`.")
endif()

message(STATUS "[std] kernel source: ${std_kernel_file} (OP_TYPE NumpyStd, soc ascend910b)")

npu_op_kernel_sources(ascendc_kernels
    OP_TYPE NumpyStd
    KERNEL_DIR .
    KERNEL_FILE ${std_kernel_file}
)

# -----------------------------------------------------------------------------
# Build the kernel library.
#
# `cust_optiling` is produced by op_host/CMakeLists.txt and carries the tiling
# registration plus the StdTilingData definition shared with the kernel; it must
# stay linked so that the host tiling data layout and the kernel parsing agree.
# -----------------------------------------------------------------------------
npu_op_kernel_library(ascendc_kernels
    SRC_BASE ${CMAKE_CURRENT_SOURCE_DIR}
    TILING_LIBRARY cust_optiling
)

# -----------------------------------------------------------------------------
# Register the kernel binary in the operator package.
# `package_name` is defined by the project root CMakeLists.txt.
# -----------------------------------------------------------------------------
npu_op_package_add(${package_name}
    LIBRARY
        ascendc_kernels
)