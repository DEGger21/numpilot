/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file repeat.h
 * @brief ACLNN L0 API 接口声明 - Repeat（numpy.repeat 定向特化算子）
 *
 * 语义来源：numpy.repeat(a, repeats, axis=...)（CPU oracle，见 gap-spec.json）。
 *
 * 分层职责（与 op_api/repeat.cpp、aclnn_repeat.cpp 严格一致）：
 * - L2 API（aclnn_repeat.cpp，两段式）：
 *     aclnnRepeatGetWorkspaceSize / aclnnRepeat
 *     负责 CheckParams、Contiguous 输入连续化、ViewCopy 输出回写、
 *     axis 归一化（负轴 -> 非负轴）以及不支持场景的显式拒绝。
 * - L0 API（本头文件 / repeat.cpp，namespace l0op）：
 *     负责输出形状推导（InferShape）、AiCore 支持性判定、输出 Tensor 分配、
 *     Kernel 调度（ADD_TO_LAUNCHER_LIST_AICORE）。
 *
 * 本候选（未验证草稿）支持范围：
 * - 输入 dtype：DT_FLOAT（float32）；其他 dtype 由 L2 显式拒绝，禁止隐式 cast。
 * - 输入 layout：ND，且已由 L2 通过 l0op::Contiguous 转为连续。
 * - rank：>= 1，且 <= ACLNN_MAX_SHAPE_RANK（由 L2 检查）。
 * - repeats：非负整数标量（int64_t），由 TilingData 下发给 Kernel；
 *   数组形式 repeats、repeats < 0 由 L2 显式拒绝。
 * - axis：已由 L2 归一化到 [0, rank-1] 的非负值；axis=None（展平语义）
 *   以及越界 axis 由 L2 显式拒绝，因此不会进入本接口。
 * - 输出 dtype 与输入 dtype 相同（DT_FLOAT），shape 为输入 shape 在 axis 维
 *   乘以 repeats，其余维保持不变（例：[1024, 1024] + repeats=3 + axis=1
 *   -> [1024, 3072]）。
 * - 数值语义：纯数据搬运（无算术运算），因此 NaN / +Inf / -Inf / +0.0 / -0.0
 *   必须按位原样复制，包括符号位；不引入任何 dtype 提升或量化。
 *
 * 调用约定：
 * - self 只读，接口不修改输入 Tensor；
 * - 返回值为本 L0 内部通过 executor->AllocTensor 分配的输出 Tensor，
 *   由 executor 管理生命周期；失败返回 nullptr，L2 必须检查空指针并转为
 *   ACLNN_ERR_INNER_NULLPTR；
 * - repeats == 0 时输出在 axis 维为 0，L2 走空 Tensor 快速返回，
 *   不会调用本接口。
 */

#ifndef OP_API_INC_LEVEL0_REPEAT_H_
#define OP_API_INC_LEVEL0_REPEAT_H_

#include <cstdint>

#include "opdev/op_executor.h"

namespace l0op {

/**
 * @brief Repeat 的 L0 入口（numpy.repeat 的定向特化实现）
 *
 * 执行流程：
 * 1. InferShape：outShape[axis] = self->GetViewShape()[axis] * repeats，
 *    其余维与 self 相同；任一维乘积溢出 int64 时返回 nullptr 并记录错误日志。
 * 2. IsAiCoreSupport：校验 NpuArch（Ascend910B3 -> DAV_2201）、
 *    dtype（仅 DT_FLOAT）、repeats >= 0、0 <= axis < rank。
 * 3. AllocTensor：按 outShape + self->GetDataType() 分配输出。
 * 4. RepeatAiCore：ADD_TO_LAUNCHER_LIST_AICORE(Repeat, OP_INPUT(self),
 *    OP_OUTPUT(out), OP_ATTR(repeats, axis))。
 *
 * @param [in] self     输入张量；要求 DT_FLOAT、ND、连续（L2 已保证）；
 *                      不得为 nullptr。
 * @param [in] repeats  沿 axis 的重复次数；必须 >= 0（标量语义）。
 * @param [in] axis     归一化后的非负轴，必须满足 0 <= axis < self 的 rank。
 * @param [in] executor 执行器，不得为 nullptr；用于分配输出并登记 Kernel。
 * @return 输出 aclTensor 指针；失败返回 nullptr（调用方必须检查）。
 */
const aclTensor* Repeat(const aclTensor* self, int64_t repeats, int64_t axis, aclOpExecutor* executor);

} // namespace l0op

#endif // OP_API_INC_LEVEL0_REPEAT_H_