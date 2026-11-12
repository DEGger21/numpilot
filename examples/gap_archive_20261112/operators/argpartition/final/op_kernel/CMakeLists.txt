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
# op_kernel/CMakeLists.txt  --  kernel build description for the Argpartition custom operator.
#
# Operator identity (must stay in sync with op_host/argpartition_def.cpp):
#   * C++ OpType / registry key ............ Argpartition
#   * File / API stem ...................... argpartition
#   * Public AsNumpy API ................... asnumpy.argpartition
#   * Public ACLNN two-stage symbols ....... aclnnArgpartitionGetWorkspaceSize / aclnnArgpartition
#   * Kernel entry function ................ argpartition(...) defined in argpartition_arch22.cpp
#
# Kernel ABI (ordered exactly as mandated by the registry-invoke template):
#   __global__ __aicore__ void argpartition(GM_ADDR x, GM_ADDR k, GM_ADDR y,
#                                           GM_ADDR workspace, GM_ADDR tiling)
# i.e. inputs (x, k) -> output (y) -> workspace -> tiling.
#
# Qualified hardware scope for this draft generation stage:
#   * CANN 9.2.0-beta.2, SoC Ascend910B3  ==  build SoC token "ascend910b" (arch22 generation)
#   * Only the observed profile is admitted: float32 [32, 16] input, int64 [32, 16] output,
#     C-contiguous, signature (a, kth) with keyword axis=1.
# Any other SoC token is rejected here, loudly, instead of silently producing a kernel binary
# that was never qualified. There is no host / CPU fallback anywhere in this build path.
#
# NOTE: this file only *describes* the kernel compilation; it makes no correctness claim.
# Accuracy and device qualification are performed by the deterministic validation gates.
# ----------------------------------------------------------------------------------------------------------

# =============================================================================
# Architecture generation configuration
#   arch22: ascend910b (Ascend910B3 is the qualified target), ascend910_93
#   arch35: ascend950  -> NOT implemented / NOT qualified for this operator draft
# =============================================================================

set(ARGPARTITION_ARCH22_COMPUTE_UNITS ascend910b ascend910_93)
set(ARGPARTITION_QUALIFIED_COMPUTE_UNIT ascend910b)

# Resolve the SoC token for this build.
# If the top-level build did not pin a SoC, default to the qualified target instead of
# fanning out to every known SoC, so an unqualified architecture is never built implicitly.
if(NOT DEFINED ASCEND_COMPUTE_UNIT OR ASCEND_COMPUTE_UNIT STREQUAL "")
    set(ARGPARTITION_COMPUTE_UNIT ${ARGPARTITION_QUALIFIED_COMPUTE_UNIT})
else()
    set(ARGPARTITION_COMPUTE_UNIT ${ASCEND_COMPUTE_UNIT})
endif()

string(TOLOWER "${ARGPARTITION_COMPUTE_UNIT}" ARGPARTITION_COMPUTE_UNIT)

list(FIND ARGPARTITION_ARCH22_COMPUTE_UNITS ${ARGPARTITION_COMPUTE_UNIT} ARGPARTITION_ARCH22_INDEX)
if(ARGPARTITION_ARCH22_INDEX EQUAL -1)
    message(FATAL_ERROR
        "Argpartition: unsupported ASCEND_COMPUTE_UNIT='${ARGPARTITION_COMPUTE_UNIT}'. "
        "This unverified draft only declares a kernel for the arch22 generation "
        "(${ARGPARTITION_ARCH22_COMPUTE_UNITS}) and is qualified only for Ascend910B3 "
        "('${ARGPARTITION_QUALIFIED_COMPUTE_UNIT}') with CANN 9.2.0-beta.2 and float32 inputs. "
        "There is no implicit architecture fallback.")
endif()

if(NOT ARGPARTITION_COMPUTE_UNIT STREQUAL ARGPARTITION_QUALIFIED_COMPUTE_UNIT)
    message(WARNING
        "Argpartition: building for '${ARGPARTITION_COMPUTE_UNIT}'. Only '${ARGPARTITION_QUALIFIED_COMPUTE_UNIT}' "
        "(Ascend910B3) is in the declared qualification scope of this draft; other SoCs remain unverified.")
endif()

message(STATUS "Argpartition: kernel build for SoC '${ARGPARTITION_COMPUTE_UNIT}' (arch22)")

# =============================================================================
# Kernel sources - arch22 generation
#   argpartition_arch22.cpp implements the Init -> CopyIn -> Compute -> CopyOut pipeline
#   and the single __global__ __aicore__ entry named `argpartition`.
# =============================================================================

npu_op_kernel_sources(ascendc_kernels
    OP_TYPE Argpartition
    KERNEL_DIR .
    KERNEL_FILE argpartition_arch22.cpp
)

# =============================================================================
# Build the kernel library
#   TILING_LIBRARY must be the host tiling library created in op_host/CMakeLists.txt
#   (cust_optiling); the kernel and host tiling share the same TilingData layout, so a
#   mismatch here would silently corrupt tiling -> fail the link instead of guessing.
# =============================================================================

if(NOT TARGET cust_optiling)
    message(FATAL_ERROR
        "Argpartition: host tiling library 'cust_optiling' is not defined. "
        "op_host/CMakeLists.txt must be processed before op_kernel/CMakeLists.txt so that the "
        "kernel links against the matching TilingData definition.")
endif()

npu_op_kernel_library(ascendc_kernels
    SRC_BASE ${CMAKE_CURRENT_SOURCE_DIR}
    TILING_LIBRARY cust_optiling
)

# =============================================================================
# Add the kernel library to the run package
#   ${package_name} is defined by the top-level CMakeLists.txt of this operator project.
# =============================================================================

if(NOT DEFINED package_name OR package_name STREQUAL "")
    message(FATAL_ERROR
        "Argpartition: 'package_name' is not defined. The top-level CMakeLists.txt must set it "
        "before adding op_kernel, otherwise the kernel binary cannot be packaged.")
endif()

npu_op_package_add(${package_name}
    LIBRARY
        ascendc_kernels
)