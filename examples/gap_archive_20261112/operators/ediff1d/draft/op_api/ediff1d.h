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
 * @file ediff1d.h
 * @brief ACLNN L0 API 接口声明 - Ediff1d（相邻元素差分，语义对齐 numpy.ediff1d）
 *
 * 文件命名规范（registry-invoke 工程）：
 * - L2 API: aclnn_{op}.h / aclnn_{op}.cpp  -> 对外两段式接口（GetWorkspaceSize / Execute）
 * - L0 API: {op}.h / {op}.cpp              -> 内部实现（形状推导、Kernel 调度），本文件
 *
 * 语义契约（唯一真值源：numpy.ediff1d 的 CPU oracle；仅支持已观测调用签名）：
 *   y = numpy.ediff1d(a)                      # 1 个位置参数，无关键字参数
 *   等价于:
 *       flat = a.reshape(-1)                  # 按 C 连续（行主序）展平
 *       y[i] = flat[i + 1] - flat[i]          # i in [0, numel(a) - 1)
 *       y.shape == [numel(a) - 1]             # 一维输出，与 a 的 rank 无关
 *
 * 空输入 / 单元素输入：numel(a) == 0 视为非法输入（oracle 抛出异常）；
 * numel(a) == 1 时输出为空一维张量，shape == [0]。
 *
 * 支持范围（本候选只声明已观测 profile，不做泛化承诺）：
 * - 输入 dtype：DT_FLOAT（float32）。其他 dtype 必须显式拒绝，禁止静默 cast。
 * - 输出 dtype：DT_FLOAT（float32），返回结构为单个张量。
 * - 布局：要求 C 连续；非连续输入由 L2 通过 l0op::Contiguous 转换后传入。
 * - rank：<= 8（ACLNN_MAX_SHAPE_RANK）。
 * - 关键字参数（to_begin / to_end / axis 等）在本批次不受支持，由 L2 层显式拒绝。
 *
 * 数值语义：单次减法，不带累加，因此为精确结果（无归约误差）；
 * NaN / +/-Inf 按 IEEE-754 传播，+0.0 与 -0.0 的符号位保持 NumPy 行为
 * （例如 (-0.0) - (0.0) == -0.0，符号位为 1）。
 */

#ifndef OP_API_INC_LEVEL0_EDIFF1D_H_
#define OP_API_INC_LEVEL0_EDIFF1D_H_

#include "opdev/op_executor.h"

namespace l0op {

/**
 * @brief Ediff1d L0 API 入口：对输入张量按行主序展平后计算相邻元素差分
 *
 * 处理流程（实现见 ediff1d.cpp）：
 *   1. 形状推导：输出恒为 1 维 [numel(a) - 1]，与输入 rank 无关；
 *      若 numel(a) == 0，返回 nullptr 表示非法输入；numel(a) == 1 时输出 shape 为 [0]。
 *   2. IsAiCoreSupport：校验 dtype 必须为 DT_FLOAT（其他 dtype 显式拒绝、不 cast），
 *      并校验当前平台为 Ascend910B3（DAV_2201）等受支持架构。
 *   3. AllocTensor：按推导形状分配输出张量（dtype 与输入一致，DT_FLOAT）。
 *   4. Ediff1dAiCore：通过 ADD_TO_LAUNCHER_LIST_AICORE(Ediff1d, OP_INPUT(a), OP_OUTPUT(out))
 *      下发 Kernel；Kernel 以扁平下标访问 GM，因此展平不需要额外的内存搬运。
 *
 * 参数顺序与 Kernel 入口保持一致：inputs -> outputs -> workspace -> tiling。
 *
 * @param [in]  a        输入张量（float32，C 连续，rank <= 8，元素个数 >= 1）
 * @param [in]  executor 执行器，用于推导形状、分配输出张量并下发任务
 * @return 成功返回新分配的 1 维输出张量（shape == [numel(a) - 1]，dtype == DT_FLOAT）；
 *         参数不合法、平台/ dtype 不受支持或下发失败时返回 nullptr
 *         （详细错误码由内部 OP_LOGE 记录，L2 层将其转换为 ACLNN_ERR_* 返回码）。
 *
 * @note 本函数不进行任何 workspace 申请，也不做同步等待；workspace 大小由
 *       aclnnEdiff1dGetWorkspaceSize 通过 executor->GetWorkspaceSize() 汇总，
 *       真正的设备执行与错误码返回由 aclnnEdiff1d（第二段接口）负责。
 * @note 调用者（L2）负责保证输入已连续化，并负责把本函数的返回值 ViewCopy 到用户输出张量；
 *       本函数不做隐式 CPU fallback，也不做隐式 dtype 提升。
 */
const aclTensor* Ediff1d(const aclTensor* a, aclOpExecutor* executor);

} // namespace l0op

#endif // OP_API_INC_LEVEL0_EDIFF1D_H_