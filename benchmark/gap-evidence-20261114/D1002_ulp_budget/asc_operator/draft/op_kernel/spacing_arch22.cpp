/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file spacing_arch22.cpp
 * \brief Spacing 算子 kernel 入口（arch22 / Ascend910B 代际）
 *
 * 算子语义（语义真值：numpy.spacing，见 GapSpec semantic_source）：
 *   y = np.spacing(x) = nextafter(x, +inf) - x
 *   - 有限值：返回 x 到「朝 +inf 方向的相邻可表示 float32 值」的距离（ulp）
 *   - x == +0.0 / -0.0：返回最小非规格化数（float32: 2^-149）
 *   - x == +inf / -inf：返回 NaN
 *   - x == NaN：返回 NaN（NaN 传播）
 *   符号位、NaN/Inf、有符号零的判定均在设备侧逐元素完成，不做任何隐式 cast，
 *   也不做 host 侧数值回退。上述数值逻辑由 NsSpacing::Spacing::Compute 实现
 *   （见 arch22/spacing.h），本文件只负责 kernel 入口的装配。
 *
 * 支持范围（与 GapSpec support_scope 一致，超出范围由 host/tiling/op_api 显式拒绝）：
 *   - 输入 dtype：float32（唯一被观测、唯一被特化的 dtype）
 *   - 输出 dtype：float32（dtype/shape 与输入一致，单 tensor 返回结构）
 *   - 输入个数：1（位置参数，keywords=[]，与原 NumPy oracle 参数顺序一致）
 *   - 输出个数：1
 *   - 布局：ND / C 连续（非连续输入由 L2 ACLNN 侧 Contiguous 处理）
 *   - 不支持的 dtype、kwarg、多输入等调用：在 op_api/tiling 层显式拒绝，不进 kernel
 *
 * Kernel 侧执行流程（Init -> CopyIn -> Compute -> CopyOut）：
 *   NsSpacing::Spacing<D_T_X, BUFFER_MODE>::Init      : 绑定 GM 地址、按 blockFactor 偏移、申请 UB
 *   NsSpacing::Spacing<D_T_X, BUFFER_MODE>::Process   : 按 ubFactor 切分，循环
 *                                                       CopyIn(GM->UB) -> Compute(UB) -> CopyOut(UB->GM)
 *
 * 符号契约（必须与本工程其他阶段保持一致，禁止单边修改）：
 *   - Kernel 入口符号：spacing（本文件），参数顺序固定为 输入 x -> 输出 y -> workspace -> tiling
 *   - TilingData 结构体名：SpacingTilingData（定义于 arch22/spacing_tiling_data.h，
 *     由 arch22/spacing.h 引入），Host 侧 tiling 写入同一份结构体
 *   - TilingKey 声明：ASCENDC_TPL_ARGS_DECL(Spacing, ...)（定义于 arch22/spacing_tiling_key.h），
 *     Host 侧使用 GET_TPL_TILING_KEY(...) 设置
 *   - Kernel 类：NsSpacing::Spacing<D_T_X, BUFFER_MODE>（定义于 arch22/spacing.h）
 *   - ACLNN 两段式：aclnnSpacingGetWorkspaceSize / aclnnSpacing（op_api 阶段）
 *
 * 模板参数说明（顺序与 arch22/spacing_tiling_key.h 的 ASCENDC_TPL_ARGS_DECL 对应）：
 *   - D_T_X      : 输入/输出数据类型，本阶段只允许 float
 *   - BUFFER_MODE: UB 缓冲模式（0=单缓冲, 1=双缓冲）
 *   两个参数均带默认值：当 TilingKey 只声明 dtype 维度（或本代际只走单一
 *   dtype/缓冲策略）时，入口仍可被实例化并调用，保持与本批「float32 定向特化、
 *   不做性能优化」的定位一致。
 */

#include "kernel_operator.h"
#include "arch22/spacing.h"

/*!
 * \brief Spacing 算子 kernel 入口（Ascend910B / arch22）
 * \param[in]  x        输入张量 GM 地址（float32，ND，C 连续）
 * \param[out] y        输出张量 GM 地址（float32，shape/dtype 与 x 完全一致）
 * \param[in]  workspace 系统预留 workspace 指针；Spacing 为逐元素算子，无需额外 workspace
 * \param[in]  tiling    TilingData（SpacingTilingData）GM 地址
 *
 * 该入口不直接做数值计算：数值语义（nextafter 形式、NaN/Inf/有符号零）全部由
 * NsSpacing::Spacing::Compute 在 UB 上完成，保证 host 侧 tiling、kernel 侧计算与
 * NumPy oracle 使用同一套 float32 语义。
 */
template <typename D_T_X = float, int BUFFER_MODE = 0>
__global__ __aicore__ void spacing(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    // 与 Host 侧 tiling 使用同一个 TilingData 结构体（SpacingTilingData），
    // 字段含义（totalNum / blockFactor / ubFactor）由 op_host/arch22/spacing_tiling.cpp 填充。
    REGISTER_TILING_DEFAULT(SpacingTilingData);
    GET_TILING_DATA_WITH_STRUCT(SpacingTilingData, tilingData, tiling);

    // 单输入、单输出、无 workspace 需求；workspace 仅按框架约定透传，不做读写。
    NsSpacing::Spacing<D_T_X, BUFFER_MODE> op;
    op.Init(x, y, &tilingData);
    op.Process();
}