"""Poisson Ritz iteration with broadcast-shaped coordinates and reused terms.

This is an optimization variant of ``poisson_ritz_iteration.py``.  Its
mathematical update is unchanged, while column-shaped coordinates allow the
NPU path to broadcast trainable vectors directly instead of materialising
twelve ``outer`` products per iteration.  Two repeated products and shared
width powers are also computed once.
"""
import numpy as np
from numpilot import operator_spec
import asnumpy as ap

@operator_spec(name='smooth_l1_loss', version='2.mean.beta1', numpy_oracle='smooth_l1_loss_v2_mean_beta1', target_api='asnumpy.smooth_l1_loss', provenance='custom_generation_candidate', route='ascendc_registry_candidate', signature_variants=({'positional_args': 2, 'keyword_args': ()},), dtype_rules=('float32',), shape_rules=('rank>=1', 'contiguous'), source='numpilot://local-llm-generation/operator/smooth_l1_loss', license='CANN-Open-Software-License-Agreement-2.0', target_soc='ascend910b', cann_version='9.0.0')
def smooth_l1_loss_v2_mean_beta1(predict, label):
    """SmoothL1V2 fixed mean/beta=1 NumPy oracle."""
    error = np.subtract(predict, label)
    absolute = np.abs(error)
    bounded = np.minimum(absolute, np.float32(1.0))
    quadratic = np.divide(np.multiply(bounded, bounded), np.float32(2.0))
    linear = np.subtract(absolute, bounded)
    return np.mean(np.add(quadratic, linear))

@operator_spec(name='apply_gradient_descent', version='1', numpy_oracle='apply_gradient_descent_oracle', target_api='asnumpy.apply_gradient_descent', provenance='custom_generation_candidate', route='ascendc_registry_candidate', signature_variants=({'positional_args': 3, 'keyword_args': ()},), dtype_rules=('float32',), shape_rules=('rank>=1', 'contiguous'), source='numpilot://local-llm-generation/operator/apply_gradient_descent', license='CANN-Open-Software-License-Agreement-2.0', target_soc='ascend910b', cann_version='9.0.0')
def apply_gradient_descent_oracle(var, alpha, delta):
    """ApplyGradientDescent non-mutating NumPy oracle."""
    return np.subtract(var, np.multiply(alpha, delta))

def build_scaled_inputs(interior_size=8, boundary_size=8, centre_size=3):
    """Build the deterministic contract at a requested scale."""
    for (name, value, lower, upper) in (('interior_size', interior_size, 2, 256), ('boundary_size', boundary_size, 2, 256), ('centre_size', centre_size, 2, 32)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f'{name} must be an integer')
        if not lower <= value <= upper:
            raise ValueError(f'{name} must be between {lower} and {upper}')
    rng = np.random.default_rng(42)
    interior_axis = np.linspace(0.04, 0.96, interior_size, dtype=np.float32)
    (grid_x, grid_y) = np.meshgrid(interior_axis, interior_axis)
    x_in = np.ascontiguousarray(grid_x.reshape(-1, 1), dtype=np.float32)
    y_in = np.ascontiguousarray(grid_y.reshape(-1, 1), dtype=np.float32)
    boundary_axis = np.linspace(0.0, 1.0, boundary_size, dtype=np.float32)
    x_bd_vector = np.concatenate((np.zeros_like(boundary_axis), np.ones_like(boundary_axis), boundary_axis, boundary_axis))
    y_bd_vector = np.concatenate((boundary_axis, boundary_axis, np.zeros_like(boundary_axis), np.ones_like(boundary_axis)))
    x_bd = np.ascontiguousarray(x_bd_vector.reshape(-1, 1), dtype=np.float32)
    y_bd = np.ascontiguousarray(y_bd_vector.reshape(-1, 1), dtype=np.float32)
    centre_axis = np.linspace(0.15, 0.85, centre_size, dtype=np.float32)
    (centre_x, centre_y) = np.meshgrid(centre_axis, centre_axis)
    cx = np.ascontiguousarray(centre_x.ravel(), dtype=np.float32)
    cy = np.ascontiguousarray(centre_y.ravel(), dtype=np.float32)
    count_rbf = cx.size
    width = np.full(count_rbf, np.float32(0.18), dtype=np.float32)
    weight = rng.normal(0.0, 0.05, count_rbf).astype(np.float32)
    x_in_vector = x_in[:, 0]
    y_in_vector = y_in[:, 0]
    source = np.multiply(np.float32(2.0 * np.pi * np.pi), np.multiply(np.sin(np.multiply(np.float32(np.pi), x_in_vector)), np.sin(np.multiply(np.float32(np.pi), y_in_vector)))).astype(np.float32)
    boundary_target = np.zeros(x_bd.shape[0], dtype=np.float32)
    source[[min(5, source.size - 1), min(37, source.size - 1)]] += np.array([10.0, -10.0], dtype=np.float32)
    boundary_target[[min(3, boundary_target.size - 1), min(22, boundary_target.size - 1)]] += np.array([-10.0, 10.0], dtype=np.float32)
    count_in = x_in.shape[0]
    count_bd = x_bd.shape[0]
    return (x_in, y_in, np.ascontiguousarray(source, dtype=np.float32), x_bd, y_bd, np.ascontiguousarray(boundary_target, dtype=np.float32), weight, cx, cy, width, np.ones(count_rbf, dtype=np.float32), np.full(count_in, np.float32(1.0 / count_in), dtype=np.float32), np.full(count_bd, np.float32(1.0 / count_bd), dtype=np.float32), np.full((count_in, count_rbf), np.float32(-0.5), dtype=np.float32), np.full((count_bd, count_rbf), np.float32(-0.5), dtype=np.float32), np.full((count_in, count_rbf), np.float32(2.0), dtype=np.float32), np.full((count_in, count_rbf), np.float32(-4.0), dtype=np.float32), np.full((count_in, count_rbf), np.float32(4.0), dtype=np.float32), np.full(count_in, np.float32(-1.0), dtype=np.float32), np.full(count_in, np.float32(1.0), dtype=np.float32), np.full(count_bd, np.float32(-1.0), dtype=np.float32), np.full(count_bd, np.float32(1.0), dtype=np.float32), np.array([np.float32(5.0)], dtype=np.float32), np.array([np.float32(0.003)], dtype=np.float32), np.array([np.float32(0.0)], dtype=np.float32), np.array([np.float32(0.0)], dtype=np.float32), np.full(count_rbf, np.float32(0.07), dtype=np.float32), np.full(count_rbf, np.float32(0.3), dtype=np.float32))

def build_inputs():
    """Build the default 8×8 / four-by-8 boundary / 3×3 RBF state."""
    return build_scaled_inputs()

def forward(x_in, y_in, source, x_bd, y_bd, boundary_target, weight, centre_x, centre_y, width, ones_rbf, mean_rows_in, mean_rows_bd, neg_half_in, neg_half_bd, two_in, neg_four_in, four_in, neg_one_in, one_in, neg_one_bd, one_bd, boundary_loss_weight, alpha_weight, alpha_centre, alpha_width, width_min, width_max):
    """Execute one broadcast-optimized training iteration."""
    if isinstance(x_in, np.ndarray):
        x_in = ap.ndarray.from_numpy(x_in)
    if isinstance(y_in, np.ndarray):
        y_in = ap.ndarray.from_numpy(y_in)
    if isinstance(source, np.ndarray):
        source = ap.ndarray.from_numpy(source)
    if isinstance(x_bd, np.ndarray):
        x_bd = ap.ndarray.from_numpy(x_bd)
    if isinstance(y_bd, np.ndarray):
        y_bd = ap.ndarray.from_numpy(y_bd)
    if isinstance(boundary_target, np.ndarray):
        boundary_target = ap.ndarray.from_numpy(boundary_target)
    if isinstance(weight, np.ndarray):
        weight = ap.ndarray.from_numpy(weight)
    if isinstance(centre_x, np.ndarray):
        centre_x = ap.ndarray.from_numpy(centre_x)
    if isinstance(centre_y, np.ndarray):
        centre_y = ap.ndarray.from_numpy(centre_y)
    if isinstance(width, np.ndarray):
        width = ap.ndarray.from_numpy(width)
    if isinstance(ones_rbf, np.ndarray):
        ones_rbf = ap.ndarray.from_numpy(ones_rbf)
    if isinstance(mean_rows_in, np.ndarray):
        mean_rows_in = ap.ndarray.from_numpy(mean_rows_in)
    if isinstance(mean_rows_bd, np.ndarray):
        mean_rows_bd = ap.ndarray.from_numpy(mean_rows_bd)
    if isinstance(neg_half_in, np.ndarray):
        neg_half_in = ap.ndarray.from_numpy(neg_half_in)
    if isinstance(neg_half_bd, np.ndarray):
        neg_half_bd = ap.ndarray.from_numpy(neg_half_bd)
    if isinstance(two_in, np.ndarray):
        two_in = ap.ndarray.from_numpy(two_in)
    if isinstance(neg_four_in, np.ndarray):
        neg_four_in = ap.ndarray.from_numpy(neg_four_in)
    if isinstance(four_in, np.ndarray):
        four_in = ap.ndarray.from_numpy(four_in)
    if isinstance(neg_one_in, np.ndarray):
        neg_one_in = ap.ndarray.from_numpy(neg_one_in)
    if isinstance(one_in, np.ndarray):
        one_in = ap.ndarray.from_numpy(one_in)
    if isinstance(neg_one_bd, np.ndarray):
        neg_one_bd = ap.ndarray.from_numpy(neg_one_bd)
    if isinstance(one_bd, np.ndarray):
        one_bd = ap.ndarray.from_numpy(one_bd)
    if isinstance(boundary_loss_weight, np.ndarray):
        boundary_loss_weight = ap.ndarray.from_numpy(boundary_loss_weight)
    if isinstance(alpha_weight, np.ndarray):
        alpha_weight = ap.ndarray.from_numpy(alpha_weight)
    if isinstance(alpha_centre, np.ndarray):
        alpha_centre = ap.ndarray.from_numpy(alpha_centre)
    if isinstance(alpha_width, np.ndarray):
        alpha_width = ap.ndarray.from_numpy(alpha_width)
    if isinstance(width_min, np.ndarray):
        width_min = ap.ndarray.from_numpy(width_min)
    if isinstance(width_max, np.ndarray):
        width_max = ap.ndarray.from_numpy(width_max)
    width_squared = ap.multiply(width, width)
    width_root = ap.sqrt(width_squared)
    width_cubed = ap.multiply(width_squared, width_root)
    width_fourth = ap.multiply(width_squared, width_squared)
    width_fifth = ap.multiply(width_fourth, width_root)
    dx_in = ap.subtract(x_in, centre_x)
    dy_in = ap.subtract(y_in, centre_y)
    radius_squared_in = ap.add(ap.multiply(dx_in, dx_in), ap.multiply(dy_in, dy_in))
    phi_in = ap.exp(ap.multiply(ap.divide(radius_squared_in, width_squared), neg_half_in))
    laplace_factor_in = ap.subtract(ap.divide(radius_squared_in, width_fourth), ap.divide(two_in, width_squared))
    phi_laplace_in = ap.multiply(phi_in, laplace_factor_in)
    laplacian_in = ap.matmul(phi_laplace_in, weight)
    negative_laplacian = ap.negative(laplacian_in)
    dx_bd = ap.subtract(x_bd, centre_x)
    dy_bd = ap.subtract(y_bd, centre_y)
    radius_squared_bd = ap.add(ap.multiply(dx_bd, dx_bd), ap.multiply(dy_bd, dy_bd))
    phi_bd = ap.exp(ap.multiply(ap.divide(radius_squared_bd, width_squared), neg_half_bd))
    boundary_prediction = ap.matmul(phi_bd, weight)
    loss_pde = ap.smooth_l1_loss(negative_laplacian, source)
    loss_boundary = ap.smooth_l1_loss(boundary_prediction, boundary_target)
    weighted_boundary_loss = ap.multiply(boundary_loss_weight, loss_boundary)
    total_loss = ap.add(loss_pde, weighted_boundary_loss)
    residual_pde = ap.subtract(negative_laplacian, source)
    residual_bd = ap.subtract(boundary_prediction, boundary_target)
    residual_gradient_pde = ap.clip(residual_pde, neg_one_in, one_in)
    residual_gradient_bd = ap.clip(residual_bd, neg_one_bd, one_bd)
    gradient_matrix_pde = ap.outer(residual_gradient_pde, ones_rbf)
    gradient_matrix_bd = ap.outer(residual_gradient_bd, ones_rbf)
    pde_weight_integrand = ap.multiply(phi_laplace_in, gradient_matrix_pde)
    gradient_weight_pde = ap.negative(ap.matmul(mean_rows_in, pde_weight_integrand))
    dphi_centre_x_in = ap.multiply(ap.divide(dx_in, width_squared), phi_in)
    dphi_centre_y_in = ap.multiply(ap.divide(dy_in, width_squared), phi_in)
    dfactor_centre_x_in = ap.divide(ap.multiply(two_in, dx_in), width_fourth)
    dfactor_centre_y_in = ap.divide(ap.multiply(two_in, dy_in), width_fourth)
    dphi_width_in = ap.multiply(ap.divide(radius_squared_in, width_cubed), phi_in)
    dfactor_width_in = ap.add(ap.divide(ap.multiply(neg_four_in, radius_squared_in), width_fifth), ap.divide(four_in, width_cubed))
    gradient_centre_x_integrand_pde = ap.multiply(ap.multiply(weight, ap.add(ap.multiply(dfactor_centre_x_in, phi_in), ap.multiply(laplace_factor_in, dphi_centre_x_in))), gradient_matrix_pde)
    gradient_centre_y_integrand_pde = ap.multiply(ap.multiply(weight, ap.add(ap.multiply(dfactor_centre_y_in, phi_in), ap.multiply(laplace_factor_in, dphi_centre_y_in))), gradient_matrix_pde)
    gradient_width_integrand_pde = ap.multiply(ap.multiply(weight, ap.add(ap.multiply(dfactor_width_in, phi_in), ap.multiply(laplace_factor_in, dphi_width_in))), gradient_matrix_pde)
    gradient_centre_x_pde = ap.negative(ap.matmul(mean_rows_in, gradient_centre_x_integrand_pde))
    gradient_centre_y_pde = ap.negative(ap.matmul(mean_rows_in, gradient_centre_y_integrand_pde))
    gradient_width_pde = ap.negative(ap.matmul(mean_rows_in, gradient_width_integrand_pde))
    phi_gradient_bd = ap.multiply(phi_bd, gradient_matrix_bd)
    gradient_weight_bd = ap.multiply(boundary_loss_weight, ap.matmul(mean_rows_bd, phi_gradient_bd))
    common_bd = ap.multiply(weight, phi_gradient_bd)
    gradient_centre_x_bd = ap.multiply(boundary_loss_weight, ap.matmul(mean_rows_bd, ap.multiply(ap.divide(dx_bd, width_squared), common_bd)))
    gradient_centre_y_bd = ap.multiply(boundary_loss_weight, ap.matmul(mean_rows_bd, ap.multiply(ap.divide(dy_bd, width_squared), common_bd)))
    gradient_width_bd = ap.multiply(boundary_loss_weight, ap.matmul(mean_rows_bd, ap.multiply(ap.divide(radius_squared_bd, width_cubed), common_bd)))
    gradient_weight = ap.add(gradient_weight_pde, gradient_weight_bd)
    gradient_centre_x = ap.add(gradient_centre_x_pde, gradient_centre_x_bd)
    gradient_centre_y = ap.add(gradient_centre_y_pde, gradient_centre_y_bd)
    gradient_width = ap.add(gradient_width_pde, gradient_width_bd)
    next_weight = ap.apply_gradient_descent(weight, alpha_weight, gradient_weight)
    next_centre_x = ap.apply_gradient_descent(centre_x, alpha_centre, gradient_centre_x)
    next_centre_y = ap.apply_gradient_descent(centre_y, alpha_centre, gradient_centre_y)
    raw_width = ap.apply_gradient_descent(width, alpha_width, gradient_width)
    next_width = ap.clip(raw_width, width_min, width_max)
    return (total_loss, next_weight, next_centre_x, next_centre_y, next_width)
if __name__ == '__main__':
    result = forward(*build_inputs())
    print('loss:', float(np.asarray(result[0]).reshape(-1)[0]))
    print('weight norm:', float(np.linalg.norm(result[1])))