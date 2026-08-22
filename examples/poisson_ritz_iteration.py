"""Poisson Ritz 的完整动态单轮计算契约。

``build_inputs`` 只负责在 CPU 构造确定性配点、初始参数和广播常量；RBF 前向、
Laplacian、鲁棒损失、解析梯度与四组参数更新全部位于唯一的 ``forward`` 内。
NumPilot 因而可以先扫描整轮所需接口，只对真正缺失的两个领域算子做定向生成并
将合格产物保存在本地。首轮生成不拉取 ``ops-nn``。
"""

import numpy as np

from numpilot import operator_spec


@operator_spec(
    name="smooth_l1_loss",
    version="2.mean.beta1",
    numpy_oracle="smooth_l1_loss_v2_mean_beta1",
    target_api="asnumpy.smooth_l1_loss",
    provenance="custom_generation_candidate",
    route="ascendc_registry_candidate",
    signature_variants=(
        {"positional_args": 2, "keyword_args": ()},
    ),
    dtype_rules=("float32",),
    shape_rules=("rank>=1", "contiguous"),
    source="numpilot://local-llm-generation/operator/smooth_l1_loss",
    license="CANN-Open-Software-License-Agreement-2.0",
    target_soc="ascend910b",
    cann_version="9.0.0",
)
def smooth_l1_loss_v2_mean_beta1(predict, label):
    """SmoothL1V2 的固定 mean/beta=1 NumPy oracle。"""

    error = np.subtract(predict, label)
    absolute = np.abs(error)
    bounded = np.minimum(absolute, np.float32(1.0))
    quadratic = np.divide(
        np.multiply(bounded, bounded),
        np.float32(2.0),
    )
    linear = np.subtract(absolute, bounded)
    return np.mean(np.add(quadratic, linear))


@operator_spec(
    name="apply_gradient_descent",
    version="1",
    numpy_oracle="apply_gradient_descent_oracle",
    target_api="asnumpy.apply_gradient_descent",
    provenance="custom_generation_candidate",
    route="ascendc_registry_candidate",
    signature_variants=(
        {"positional_args": 3, "keyword_args": ()},
    ),
    dtype_rules=("float32",),
    shape_rules=("rank>=1", "contiguous"),
    source="numpilot://local-llm-generation/operator/apply_gradient_descent",
    license="CANN-Open-Software-License-Agreement-2.0",
    target_soc="ascend910b",
    cann_version="9.0.0",
)
def apply_gradient_descent_oracle(var, alpha, delta):
    """ApplyGradientDescent 的非原地 NumPy oracle。"""

    return np.subtract(var, np.multiply(alpha, delta))


def build_inputs():
    """构造 8×8 内部点、四边各 8 点和 3×3 RBF 的确定性状态。"""

    rng = np.random.default_rng(42)
    interior_axis = np.linspace(0.04, 0.96, 8, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(interior_axis, interior_axis)
    x_in = np.ascontiguousarray(grid_x.ravel(), dtype=np.float32)
    y_in = np.ascontiguousarray(grid_y.ravel(), dtype=np.float32)

    boundary_axis = np.linspace(0.0, 1.0, 8, dtype=np.float32)
    x_bd = np.ascontiguousarray(np.concatenate((
        np.zeros_like(boundary_axis),
        np.ones_like(boundary_axis),
        boundary_axis,
        boundary_axis,
    )), dtype=np.float32)
    y_bd = np.ascontiguousarray(np.concatenate((
        boundary_axis,
        boundary_axis,
        np.zeros_like(boundary_axis),
        np.ones_like(boundary_axis),
    )), dtype=np.float32)

    centre_axis = np.linspace(0.15, 0.85, 3, dtype=np.float32)
    centre_x, centre_y = np.meshgrid(centre_axis, centre_axis)
    cx = np.ascontiguousarray(centre_x.ravel(), dtype=np.float32)
    cy = np.ascontiguousarray(centre_y.ravel(), dtype=np.float32)
    width = np.full(9, np.float32(0.18), dtype=np.float32)
    weight = rng.normal(0.0, 0.05, 9).astype(np.float32)

    source = np.multiply(
        np.float32(2.0 * np.pi * np.pi),
        np.multiply(
            np.sin(np.multiply(np.float32(np.pi), x_in)),
            np.sin(np.multiply(np.float32(np.pi), y_in)),
        ),
    ).astype(np.float32)
    boundary_target = np.zeros_like(x_bd)
    source[[5, 37]] += np.array([10.0, -10.0], dtype=np.float32)
    boundary_target[[3, 22]] += np.array([-10.0, 10.0], dtype=np.float32)

    count_in = x_in.size
    count_bd = x_bd.size
    count_rbf = weight.size
    ones_rbf = np.ones(count_rbf, dtype=np.float32)
    ones_in = np.ones(count_in, dtype=np.float32)
    ones_bd = np.ones(count_bd, dtype=np.float32)

    return (
        x_in,
        y_in,
        np.ascontiguousarray(source, dtype=np.float32),
        x_bd,
        y_bd,
        np.ascontiguousarray(boundary_target, dtype=np.float32),
        weight,
        cx,
        cy,
        width,
        ones_rbf,
        ones_in,
        ones_bd,
        np.full(count_in, np.float32(1.0 / count_in), dtype=np.float32),
        np.full(count_bd, np.float32(1.0 / count_bd), dtype=np.float32),
        np.full((count_in, count_rbf), np.float32(-0.5), dtype=np.float32),
        np.full((count_bd, count_rbf), np.float32(-0.5), dtype=np.float32),
        np.full((count_in, count_rbf), np.float32(2.0), dtype=np.float32),
        np.full((count_in, count_rbf), np.float32(-4.0), dtype=np.float32),
        np.full((count_in, count_rbf), np.float32(4.0), dtype=np.float32),
        np.full(count_in, np.float32(-1.0), dtype=np.float32),
        np.full(count_in, np.float32(1.0), dtype=np.float32),
        np.full(count_bd, np.float32(-1.0), dtype=np.float32),
        np.full(count_bd, np.float32(1.0), dtype=np.float32),
        np.array([np.float32(5.0)], dtype=np.float32),
        np.array([np.float32(3e-3)], dtype=np.float32),
        np.array([np.float32(0.0)], dtype=np.float32),
        np.array([np.float32(0.0)], dtype=np.float32),
        np.full(count_rbf, np.float32(0.07), dtype=np.float32),
        np.full(count_rbf, np.float32(0.30), dtype=np.float32),
    )


def forward(
    x_in,
    y_in,
    source,
    x_bd,
    y_bd,
    boundary_target,
    weight,
    centre_x,
    centre_y,
    width,
    ones_rbf,
    ones_in,
    ones_bd,
    mean_rows_in,
    mean_rows_bd,
    neg_half_in,
    neg_half_bd,
    two_in,
    neg_four_in,
    four_in,
    neg_one_in,
    one_in,
    neg_one_bd,
    one_bd,
    boundary_loss_weight,
    alpha_weight,
    alpha_centre,
    alpha_width,
    width_min,
    width_max,
):
    """在一个可迁移的 straight-line NPU 岛内完成完整训练迭代。"""

    width_squared_vector = np.multiply(width, width)

    x_in_matrix = np.outer(x_in, ones_rbf)
    cx_in_matrix = np.outer(ones_in, centre_x)
    y_in_matrix = np.outer(y_in, ones_rbf)
    cy_in_matrix = np.outer(ones_in, centre_y)
    dx_in = np.subtract(x_in_matrix, cx_in_matrix)
    dy_in = np.subtract(y_in_matrix, cy_in_matrix)
    radius_squared_in = np.add(
        np.multiply(dx_in, dx_in),
        np.multiply(dy_in, dy_in),
    )
    width_squared_in = np.outer(ones_in, width_squared_vector)
    phi_in = np.exp(np.multiply(
        np.divide(radius_squared_in, width_squared_in),
        neg_half_in,
    ))
    width_fourth_in = np.multiply(width_squared_in, width_squared_in)
    laplace_factor_in = np.subtract(
        np.divide(radius_squared_in, width_fourth_in),
        np.divide(two_in, width_squared_in),
    )
    laplacian_in = np.matmul(
        np.multiply(phi_in, laplace_factor_in),
        weight,
    )
    negative_laplacian = np.negative(laplacian_in)

    x_bd_matrix = np.outer(x_bd, ones_rbf)
    cx_bd_matrix = np.outer(ones_bd, centre_x)
    y_bd_matrix = np.outer(y_bd, ones_rbf)
    cy_bd_matrix = np.outer(ones_bd, centre_y)
    dx_bd = np.subtract(x_bd_matrix, cx_bd_matrix)
    dy_bd = np.subtract(y_bd_matrix, cy_bd_matrix)
    radius_squared_bd = np.add(
        np.multiply(dx_bd, dx_bd),
        np.multiply(dy_bd, dy_bd),
    )
    width_squared_bd = np.outer(ones_bd, width_squared_vector)
    phi_bd = np.exp(np.multiply(
        np.divide(radius_squared_bd, width_squared_bd),
        neg_half_bd,
    ))
    boundary_prediction = np.matmul(phi_bd, weight)

    loss_pde = smooth_l1_loss_v2_mean_beta1(negative_laplacian, source)
    loss_boundary = smooth_l1_loss_v2_mean_beta1(
        boundary_prediction,
        boundary_target,
    )
    weighted_boundary_loss = np.multiply(boundary_loss_weight, loss_boundary)
    total_loss = np.add(loss_pde, weighted_boundary_loss)

    residual_pde = np.subtract(negative_laplacian, source)
    residual_bd = np.subtract(boundary_prediction, boundary_target)
    residual_gradient_pde = np.clip(residual_pde, neg_one_in, one_in)
    residual_gradient_bd = np.clip(residual_bd, neg_one_bd, one_bd)
    gradient_matrix_pde = np.outer(residual_gradient_pde, ones_rbf)
    gradient_matrix_bd = np.outer(residual_gradient_bd, ones_rbf)

    pde_weight_integrand = np.multiply(
        np.multiply(phi_in, laplace_factor_in),
        gradient_matrix_pde,
    )
    gradient_weight_pde = np.negative(
        np.matmul(mean_rows_in, pde_weight_integrand)
    )

    width_root_in = np.sqrt(width_squared_in)
    width_cubed_in = np.multiply(width_squared_in, width_root_in)
    width_fifth_in = np.multiply(width_fourth_in, width_root_in)
    dphi_centre_x_in = np.multiply(
        np.divide(dx_in, width_squared_in),
        phi_in,
    )
    dphi_centre_y_in = np.multiply(
        np.divide(dy_in, width_squared_in),
        phi_in,
    )
    dfactor_centre_x_in = np.divide(
        np.multiply(two_in, dx_in),
        width_fourth_in,
    )
    dfactor_centre_y_in = np.divide(
        np.multiply(two_in, dy_in),
        width_fourth_in,
    )
    dphi_width_in = np.multiply(
        np.divide(radius_squared_in, width_cubed_in),
        phi_in,
    )
    dfactor_width_in = np.add(
        np.divide(np.multiply(neg_four_in, radius_squared_in), width_fifth_in),
        np.divide(four_in, width_cubed_in),
    )
    weight_matrix_in = np.outer(ones_in, weight)
    gradient_centre_x_integrand_pde = np.multiply(
        np.multiply(
            weight_matrix_in,
            np.add(
                np.multiply(dfactor_centre_x_in, phi_in),
                np.multiply(laplace_factor_in, dphi_centre_x_in),
            ),
        ),
        gradient_matrix_pde,
    )
    gradient_centre_y_integrand_pde = np.multiply(
        np.multiply(
            weight_matrix_in,
            np.add(
                np.multiply(dfactor_centre_y_in, phi_in),
                np.multiply(laplace_factor_in, dphi_centre_y_in),
            ),
        ),
        gradient_matrix_pde,
    )
    gradient_width_integrand_pde = np.multiply(
        np.multiply(
            weight_matrix_in,
            np.add(
                np.multiply(dfactor_width_in, phi_in),
                np.multiply(laplace_factor_in, dphi_width_in),
            ),
        ),
        gradient_matrix_pde,
    )
    gradient_centre_x_pde = np.negative(
        np.matmul(mean_rows_in, gradient_centre_x_integrand_pde)
    )
    gradient_centre_y_pde = np.negative(
        np.matmul(mean_rows_in, gradient_centre_y_integrand_pde)
    )
    gradient_width_pde = np.negative(
        np.matmul(mean_rows_in, gradient_width_integrand_pde)
    )

    gradient_weight_bd = np.multiply(
        boundary_loss_weight,
        np.matmul(mean_rows_bd, np.multiply(phi_bd, gradient_matrix_bd)),
    )
    width_root_bd = np.sqrt(width_squared_bd)
    width_cubed_bd = np.multiply(width_squared_bd, width_root_bd)
    weight_matrix_bd = np.outer(ones_bd, weight)
    common_bd = np.multiply(
        np.multiply(weight_matrix_bd, phi_bd),
        gradient_matrix_bd,
    )
    gradient_centre_x_bd = np.multiply(
        boundary_loss_weight,
        np.matmul(
            mean_rows_bd,
            np.multiply(np.divide(dx_bd, width_squared_bd), common_bd),
        ),
    )
    gradient_centre_y_bd = np.multiply(
        boundary_loss_weight,
        np.matmul(
            mean_rows_bd,
            np.multiply(np.divide(dy_bd, width_squared_bd), common_bd),
        ),
    )
    gradient_width_bd = np.multiply(
        boundary_loss_weight,
        np.matmul(
            mean_rows_bd,
            np.multiply(np.divide(radius_squared_bd, width_cubed_bd), common_bd),
        ),
    )

    gradient_weight = np.add(gradient_weight_pde, gradient_weight_bd)
    gradient_centre_x = np.add(gradient_centre_x_pde, gradient_centre_x_bd)
    gradient_centre_y = np.add(gradient_centre_y_pde, gradient_centre_y_bd)
    gradient_width = np.add(gradient_width_pde, gradient_width_bd)

    next_weight = apply_gradient_descent_oracle(
        weight,
        alpha_weight,
        gradient_weight,
    )
    next_centre_x = apply_gradient_descent_oracle(
        centre_x,
        alpha_centre,
        gradient_centre_x,
    )
    next_centre_y = apply_gradient_descent_oracle(
        centre_y,
        alpha_centre,
        gradient_centre_y,
    )
    raw_width = apply_gradient_descent_oracle(
        width,
        alpha_width,
        gradient_width,
    )
    next_width = np.clip(raw_width, width_min, width_max)
    return total_loss, next_weight, next_centre_x, next_centre_y, next_width


if __name__ == "__main__":
    result = forward(*build_inputs())
    print("loss:", float(np.asarray(result[0]).reshape(-1)[0]))
    print("weight norm:", float(np.linalg.norm(result[1])))
