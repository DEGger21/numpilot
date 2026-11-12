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
 * \file flip_arch22.cpp
 * \brief Flip 算子 kernel 入口（arch22 / Ascend910B3）
 *
 * 语义来源：NumPy `numpy.flip(a, axis=...)`（语义真值见 numpy_oracle）。
 *   输出 y 与输入 x 形状、dtype 相同；沿归一化后的 axis 维做索引反转，
 *   y[..., k, ...] = x[..., axisDim - 1 - k, ...]（其余维保持不变）。
 *
 * 本候选的支持范围（未验证草案，范围外由 op_host 显式拒绝）：
 *   - 输入/输出 dtype：float32（C_DT_FLOAT），不做任何隐式 cast；
 *   - 输入为 C 连续 ND 张量，输出与输入同 shape；
 *   - 签名固定为 1 个位置参数 + 关键字 axis（axis 由 host 侧做负轴归一化与范围校验）。
 *
 * 模板参数说明（与 op_kernel/arch22/flip_tiling_key.h 中 ASCENDC_TPL_ARGS_DECL 定义对应）：
 *   - D_T_X:       数据类型，由 ASCENDC_TPL_DATATYPE_DECL 定义（本候选仅 CGFloat32/C_DT_FLOAT 有 SEL）；
 *   - BUFFER_MODE: 缓冲模式（0 = 单缓冲，1 = 双缓冲），由 ASCENDC_TPL_UINT_DECL 定义。
 *
 * 核函数参数顺序固定：输入 -> 输出 -> workspace -> tiling。
 *   - `x`         : 输入张量 GM 地址（float32，C 连续）
 *   - `y`         : 输出张量 GM 地址（float32，与 x 同 shape）
 *   - `workspace` : 系统 workspace（本算子不额外申请）
 *   - `tiling`    : FlipTilingData 序列化数据
 *
 * 执行流程（在 NsFlip::Flip::Process 内实现，遵循 Init -> CopyIn -> Compute -> CopyOut）：
 *   按 (outer, axisDim, inner) 视图切分，逐 UB 块搬运连续 inner 段并在 UB 内完成反转，
 *   不做 host 侧计算，也不存在 CPU/NumPy 回退路径。
 */

#include "arch22/flip.h"
#include "arch22/flip_tiling_data.h"

extern "C" __global__ __aicore__ void flip(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    REGISTER_TILING_DEFAULT(FlipTilingData);
    GET_TILING_DATA_WITH_STRUCT(FlipTilingData, tilingData, tiling);

    // float32 and single buffering are the only admitted instantiation; the
    // host tiling rejects every other dtype and buffer mode.
    NsFlip::Flip<float, 0> op;
    op.Init(x, y, &tilingData);
    op.Process();
}