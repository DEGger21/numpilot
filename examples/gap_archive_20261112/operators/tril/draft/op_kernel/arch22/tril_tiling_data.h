/*!
 * \file tril_tiling_data.h
 * \brief Tiling data shared by the Tril host tiling implementation and the
 *        arch22 Ascend C kernel.
 *
 * The operator implements numpy.tril(a) for one float32 2-D C-contiguous
 * input tensor.  The diagonal offset is implicitly zero, matching the
 * single-positional-argument source signature.
 *
 * The tiling data carries both element-wise and row-wise partition values so
 * that the host tiling and kernel can agree on a bounded execution plan.
 * Consumers must use the field family that matches their launch strategy:
 *   - element-wise: totalNum, blockFactor, ubFactor
 *   - row-wise:     rows, cols, blockRows, ubRows
 *
 * Invariants:
 *   totalNum == rows * cols
 *   blockFactor and ubFactor are element counts
 *   blockRows and ubRows are row counts
 *   alignCols is the smallest multiple of 8 (32 bytes for float32) that is
 *   greater than or equal to cols
 */

#ifndef _TRIL_TILING_DATA_H_
#define _TRIL_TILING_DATA_H_

#include <cstdint>

struct TrilTilingData {
    // Element-wise partition.  totalNum is rows * cols; blockFactor is the
    // maximum number of elements assigned to one AI core; ubFactor is the
    // maximum number of elements processed in one UB loop iteration.
    int64_t totalNum = 0;
    int64_t blockFactor = 0;
    int64_t ubFactor = 0;

    // Matrix geometry for the tril mask.  The observed production profile is
    // a 2048 x 2048 float32 C-contiguous tensor, but the kernel uses these
    // values at runtime rather than assuming that shape.
    int64_t rows = 0;
    int64_t cols = 0;

    // Row-wise partition alternative.  blockRows is the maximum number of
    // rows assigned to one AI core; ubRows is the maximum number of rows
    // processed in one UB loop iteration.
    int64_t blockRows = 0;
    int64_t ubRows = 0;

    // Column count rounded up to a 32-byte boundary in float32 elements.
    // Used when a kernel copies aligned row segments into UB.
    int64_t alignCols = 0;

    // Number of AI cores actually launched for this call.
    int64_t usedCoreNum = 0;

    // Reserved for future tiling extensions.  Must be set to zero by the
    // host tiling implementation and ignored by the kernel.
    int64_t reserved = 0;
};

#endif // _TRIL_TILING_DATA_H_