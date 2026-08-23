"""Poisson Ritz iteration with broadcast-shaped coordinates and reused terms.

This is an optimization variant of ``poisson_ritz_iteration.py``.  Its
mathematical update is unchanged, while column-shaped coordinates allow the
NPU path to broadcast trainable vectors directly instead of materialising
twelve ``outer`` products per iteration.  Two repeated products and shared
width powers are also computed once.
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
    """SmoothL1V2 fixed mean/beta=1 NumPy oracle."""

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
    """ApplyGradientDescent non-mutating NumPy oracle."""

    return np.subtract(var, np.multiply(alpha, delta))


def build_scaled_inputs(interior_size=8, boundary_size=8, centre_size=3):
    """Build the deterministic contract at a requested scale."""

    for name, value, lower, upper in (
        ("interior_size", interior_size, 2, 256),
        ("boundary_size", boundary_size, 2, 256),
        ("centre_size", centre_size, 2, 32),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
        if not lower <= value <= upper:
            raise ValueError(f"{name} must be between {lower} and {upper}")

    rng = np.random.default_rng(42)
    interior_axis = np.linspace(0.04, 0.96, interior_size, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(interior_axis, interior_axis)
    x_in = np.ascontiguousarray(grid_x.reshape(-1, 1), dtype=np.float32)
    y_in = np.ascontiguousarray(grid_y.reshape(-1, 1), dtype=np.float32)

    boundary_axis = np.linspace(0.0, 1.0, boundary_size, dtype=np.float32)
    x_bd_vector = np.concatenate((
        np.zeros_like(boundary_axis),
        np.ones_like(boundary_axis),
        boundary_axis,
        boundary_axis,
    ))
    y_bd_vector = np.concatenate((
        boundary_axis,
        boundary_axis,
        np.zeros_like(boundary_axis),
        np.ones_like(boundary_axis),
    ))
    x_bd = np.ascontiguousarray(x_bd_vector.reshape(-1, 1), dtype=np.float32)
    y_bd = np.ascontiguousarray(y_bd_vector.reshape(-1, 1), dtype=np.float32)

    centre_axis = np.linspace(0.15, 0.85, centre_size, dtype=np.float32)
    centre_x, centre_y = np.meshgrid(centre_axis, centre_axis)
    cx = np.ascontiguousarray(centre_x.ravel(), dtype=np.float32)
    cy = np.ascontiguousarray(centre_y.ravel(), dtype=np.float32)
    count_rbf = cx.size
    width = np.full(count_rbf, np.float32(0.18), dtype=np.float32)
    weight = rng.normal(0.0, 0.05, count_rbf).astype(np.float32)

    x_in_vector = x_in[:, 0]
    y_in_vector = y_in[:, 0]
    source = np.multiply(
        np.float32(2.0 * np.pi * np.pi),
        np.multiply(
            np.sin(np.multiply(np.float32(np.pi), x_in_vector)),
            np.sin(np.multiply(np.float32(np.pi), y_in_vector)),
        ),
    ).astype(np.float32)
    boundary_target = np.zeros(x_bd.shape[0], dtype=np.float32)
    source[[min(5, source.size - 1), min(37, source.size - 1)]] += np.array(
        [10.0, -10.0], dtype=np.float32
    )
    boundary_target[
        [min(3, boundary_target.size - 1), min(22, boundary_target.size - 1)]
    ] += np.array([-10.0, 10.0], dtype=np.float32)

    count_in = x_in.shape[0]
    count_bd = x_bd.shape[0]
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
        np.ones(count_rbf, dtype=np.float32),
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


def build_inputs():
    """Build the default 8×8 / four-by-8 boundary / 3×3 RBF state."""

    return build_scaled_inputs()


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
    """Execute one broadcast-optimized training iteration."""

    width_squared = np.multiply(width, width)
    width_root = np.sqrt(width_squared)
    width_cubed = np.multiply(width_squared, width_root)
    width_fourth = np.multiply(width_squared, width_squared)
    width_fifth = np.multiply(width_fourth, width_root)

    dx_in = np.subtract(x_in, centre_x)
    dy_in = np.subtract(y_in, centre_y)
    radius_squared_in = np.add(
        np.multiply(dx_in, dx_in),
        np.multiply(dy_in, dy_in),
    )
    phi_in = np.exp(np.multiply(
        np.divide(radius_squared_in, width_squared),
        neg_half_in,
    ))
    laplace_factor_in = np.subtract(
        np.divide(radius_squared_in, width_fourth),
        np.divide(two_in, width_squared),
    )
    phi_laplace_in = np.multiply(phi_in, laplace_factor_in)
    laplacian_in = np.matmul(phi_laplace_in, weight)
    negative_laplacian = np.negative(laplacian_in)

    dx_bd = np.subtract(x_bd, centre_x)
    dy_bd = np.subtract(y_bd, centre_y)
    radius_squared_bd = np.add(
        np.multiply(dx_bd, dx_bd),
        np.multiply(dy_bd, dy_bd),
    )
    phi_bd = np.exp(np.multiply(
        np.divide(radius_squared_bd, width_squared),
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

    pde_weight_integrand = np.multiply(phi_laplace_in, gradient_matrix_pde)
    gradient_weight_pde = np.negative(
        np.matmul(mean_rows_in, pde_weight_integrand)
    )

    dphi_centre_x_in = np.multiply(
        np.divide(dx_in, width_squared),
        phi_in,
    )
    dphi_centre_y_in = np.multiply(
        np.divide(dy_in, width_squared),
        phi_in,
    )
    dfactor_centre_x_in = np.divide(
        np.multiply(two_in, dx_in),
        width_fourth,
    )
    dfactor_centre_y_in = np.divide(
        np.multiply(two_in, dy_in),
        width_fourth,
    )
    dphi_width_in = np.multiply(
        np.divide(radius_squared_in, width_cubed),
        phi_in,
    )
    dfactor_width_in = np.add(
        np.divide(np.multiply(neg_four_in, radius_squared_in), width_fifth),
        np.divide(four_in, width_cubed),
    )
    gradient_centre_x_integrand_pde = np.multiply(
        np.multiply(
            weight,
            np.add(
                np.multiply(dfactor_centre_x_in, phi_in),
                np.multiply(laplace_factor_in, dphi_centre_x_in),
            ),
        ),
        gradient_matrix_pde,
    )
    gradient_centre_y_integrand_pde = np.multiply(
        np.multiply(
            weight,
            np.add(
                np.multiply(dfactor_centre_y_in, phi_in),
                np.multiply(laplace_factor_in, dphi_centre_y_in),
            ),
        ),
        gradient_matrix_pde,
    )
    gradient_width_integrand_pde = np.multiply(
        np.multiply(
            weight,
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

    phi_gradient_bd = np.multiply(phi_bd, gradient_matrix_bd)
    gradient_weight_bd = np.multiply(
        boundary_loss_weight,
        np.matmul(mean_rows_bd, phi_gradient_bd),
    )
    common_bd = np.multiply(weight, phi_gradient_bd)
    gradient_centre_x_bd = np.multiply(
        boundary_loss_weight,
        np.matmul(
            mean_rows_bd,
            np.multiply(np.divide(dx_bd, width_squared), common_bd),
        ),
    )
    gradient_centre_y_bd = np.multiply(
        boundary_loss_weight,
        np.matmul(
            mean_rows_bd,
            np.multiply(np.divide(dy_bd, width_squared), common_bd),
        ),
    )
    gradient_width_bd = np.multiply(
        boundary_loss_weight,
        np.matmul(
            mean_rows_bd,
            np.multiply(np.divide(radius_squared_bd, width_cubed), common_bd),
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
