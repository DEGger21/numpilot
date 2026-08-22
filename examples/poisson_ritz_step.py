"""Poisson Ritz 的确定性单步计算契约。

外层求解器负责配点、RBF 前向和解析梯度编排；``build_inputs`` 在 CPU 上构造
一个真实的确定性 Poisson/RBF 状态，``forward`` 则表达损失计算和一次参数更新。
两个领域 helper 通过 OperatorSpec 显式映射到待补充的 AsNumpy 公共 API。NumPilot
先根据缺口契约和固定版本的官方 CANNBot Skill 直接生成本地 Ascend C 候选工程；
外部算子仓库只允许作为后续人工审核的可选参考，不是首轮生成依赖。
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
    """构造一个 8×8 内部配点、每边 8 点、9 个 RBF 的真实单步状态。"""

    rng = np.random.default_rng(42)
    interior_axis = np.linspace(0.04, 0.96, 8, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(interior_axis, interior_axis)
    x_in = np.ascontiguousarray(grid_x.ravel(), dtype=np.float32)
    y_in = np.ascontiguousarray(grid_y.ravel(), dtype=np.float32)

    boundary_axis = np.linspace(0.0, 1.0, 8, dtype=np.float32)
    x_bd = np.concatenate((
        np.zeros_like(boundary_axis),
        np.ones_like(boundary_axis),
        boundary_axis,
        boundary_axis,
    ))
    y_bd = np.concatenate((
        boundary_axis,
        boundary_axis,
        np.zeros_like(boundary_axis),
        np.ones_like(boundary_axis),
    ))

    centre_axis = np.linspace(0.15, 0.85, 3, dtype=np.float32)
    centre_x, centre_y = np.meshgrid(centre_axis, centre_axis)
    cx = np.ascontiguousarray(centre_x.ravel(), dtype=np.float32)
    cy = np.ascontiguousarray(centre_y.ravel(), dtype=np.float32)
    width = np.full(9, np.float32(0.18), dtype=np.float32)
    weight = rng.normal(0.0, 0.05, 9).astype(np.float32)

    def basis_and_laplacian(x, y):
        dx = np.subtract(x[:, None], cx[None, :])
        dy = np.subtract(y[:, None], cy[None, :])
        radius2 = np.add(np.multiply(dx, dx), np.multiply(dy, dy))
        width2 = np.multiply(width[None, :], width[None, :])
        phi = np.exp(
            np.divide(np.negative(radius2), np.multiply(np.float32(2.0), width2))
        )
        laplace_factor = np.subtract(
            np.divide(radius2, np.multiply(width2, width2)),
            np.divide(np.float32(2.0), width2),
        )
        prediction = np.matmul(phi, weight)
        laplacian = np.matmul(np.multiply(phi, laplace_factor), weight)
        return dx, dy, radius2, width2, phi, laplace_factor, prediction, laplacian

    in_state = basis_and_laplacian(x_in, y_in)
    bd_state = basis_and_laplacian(x_bd, y_bd)
    source = np.multiply(
        np.float32(2.0 * np.pi * np.pi),
        np.multiply(np.sin(np.float32(np.pi) * x_in), np.sin(np.float32(np.pi) * y_in)),
    ).astype(np.float32)
    boundary_target = np.zeros_like(x_bd)

    # 固定异常值，保留 SmoothL1 鲁棒损失的实际场景。
    source = source.copy()
    boundary_target = boundary_target.copy()
    source[[5, 37]] += np.array([10.0, -10.0], dtype=np.float32)
    boundary_target[[3, 22]] += np.array([-10.0, 10.0], dtype=np.float32)

    residual_pde = np.subtract(np.negative(in_state[7]), source)
    residual_bd = np.subtract(bd_state[6], boundary_target)
    residual_grad_pde = np.clip(residual_pde, -1.0, 1.0)
    residual_grad_bd = np.clip(residual_bd, -1.0, 1.0)

    def analytic_gradients(state, residual_grad, *, pde):
        dx, dy, radius2, width2, phi, factor = state[:6]
        count = np.float32(residual_grad.size)
        gradient_matrix = residual_grad[:, None]
        width3 = np.multiply(width2, np.sqrt(width2))
        if pde:
            width4 = np.multiply(width2, width2)
            width5 = np.multiply(width4, np.sqrt(width2))
            grad_w = np.negative(
                np.sum(np.multiply(np.multiply(phi, factor), gradient_matrix), axis=0)
                / count
            )
            dphi_x = np.multiply(np.divide(dx, width2), phi)
            dphi_y = np.multiply(np.divide(dy, width2), phi)
            dfactor_x = np.divide(np.multiply(np.float32(2.0), dx), width4)
            dfactor_y = np.divide(np.multiply(np.float32(2.0), dy), width4)
            dfactor_s = np.add(
                np.multiply(np.float32(-4.0), np.divide(radius2, width5)),
                np.divide(np.float32(4.0), width3),
            )
            dphi_s = np.multiply(np.divide(radius2, width3), phi)
            grad_x = np.negative(np.sum(
                np.multiply(
                    np.multiply(
                        weight,
                        np.add(np.multiply(dfactor_x, phi), np.multiply(factor, dphi_x)),
                    ),
                    gradient_matrix,
                ),
                axis=0,
            ) / count)
            grad_y = np.negative(np.sum(
                np.multiply(
                    np.multiply(
                        weight,
                        np.add(np.multiply(dfactor_y, phi), np.multiply(factor, dphi_y)),
                    ),
                    gradient_matrix,
                ),
                axis=0,
            ) / count)
            grad_s = np.negative(np.sum(
                np.multiply(
                    np.multiply(
                        weight,
                        np.add(np.multiply(dfactor_s, phi), np.multiply(factor, dphi_s)),
                    ),
                    gradient_matrix,
                ),
                axis=0,
            ) / count)
            return grad_w, grad_x, grad_y, grad_s

        loss_weight = np.float32(5.0)
        grad_w = loss_weight * np.sum(np.multiply(phi, gradient_matrix), axis=0) / count
        common = np.multiply(np.multiply(weight, phi), gradient_matrix)
        grad_x = loss_weight * np.sum(np.multiply(np.divide(dx, width2), common), axis=0) / count
        grad_y = loss_weight * np.sum(np.multiply(np.divide(dy, width2), common), axis=0) / count
        grad_s = loss_weight * np.sum(np.multiply(np.divide(radius2, width3), common), axis=0) / count
        return grad_w, grad_x, grad_y, grad_s

    pde_grads = analytic_gradients(in_state, residual_grad_pde, pde=True)
    boundary_grads = analytic_gradients(bd_state, residual_grad_bd, pde=False)
    gradients = tuple(
        np.ascontiguousarray(np.add(left, right), dtype=np.float32)
        for left, right in zip(pde_grads, boundary_grads)
    )

    return (
        np.ascontiguousarray(np.negative(in_state[7]), dtype=np.float32),
        np.ascontiguousarray(source, dtype=np.float32),
        np.ascontiguousarray(bd_state[6], dtype=np.float32),
        np.ascontiguousarray(boundary_target, dtype=np.float32),
        weight,
        cx,
        cy,
        width,
        gradients[0],
        gradients[1],
        gradients[2],
        gradients[3],
        np.array([np.float32(3e-3)], dtype=np.float32),
        np.array([np.float32(0.0)], dtype=np.float32),
        np.array([np.float32(0.0)], dtype=np.float32),
        np.full(9, np.float32(0.07), dtype=np.float32),
        np.full(9, np.float32(0.30), dtype=np.float32),
        np.full(1, np.float32(5.0), dtype=np.float32),
    )


def forward(
    negative_laplacian,
    source,
    boundary_prediction,
    boundary_target,
    weight,
    centre_x,
    centre_y,
    width,
    grad_weight,
    grad_centre_x,
    grad_centre_y,
    grad_width,
    alpha_weight,
    alpha_centre,
    alpha_width,
    width_min,
    width_max,
    boundary_loss_weight,
):
    """计算鲁棒 Poisson/边界损失，并完成四组参数的一次更新。"""

    loss_pde = smooth_l1_loss_v2_mean_beta1(negative_laplacian, source)
    loss_boundary = smooth_l1_loss_v2_mean_beta1(
        boundary_prediction,
        boundary_target,
    )
    weighted_boundary = np.multiply(boundary_loss_weight, loss_boundary)
    total_loss = np.add(loss_pde, weighted_boundary)
    next_weight = apply_gradient_descent_oracle(weight, alpha_weight, grad_weight)
    next_centre_x = apply_gradient_descent_oracle(
        centre_x,
        alpha_centre,
        grad_centre_x,
    )
    next_centre_y = apply_gradient_descent_oracle(
        centre_y,
        alpha_centre,
        grad_centre_y,
    )
    raw_width = apply_gradient_descent_oracle(width, alpha_width, grad_width)
    next_width = np.clip(raw_width, width_min, width_max)
    return total_loss, next_weight, next_centre_x, next_centre_y, next_width


if __name__ == "__main__":
    result = forward(*build_inputs())
    print("loss:", float(np.asarray(result[0]).reshape(-1)[0]))
    print("weight norm:", float(np.linalg.norm(result[1])))
