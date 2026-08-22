"""Poisson Ritz 的完整动态单轮计算契约。

``build_inputs`` 只负责在 CPU 构造确定性配点、初始参数和广播常量；RBF 前向、
Laplacian、鲁棒损失、解析梯度与四组参数更新全部位于唯一的 ``forward`` 内。
NumPilot 因而可以先扫描整轮所需接口，只对真正缺失的两个领域算子做定向生成并
将合格产物保存在本地。首轮生成不拉取 ``ops-nn``。
"""
import numpy as np
from numpilot import operator_spec
import asnumpy as ap

@operator_spec(name='smooth_l1_loss', version='2.mean.beta1', numpy_oracle='smooth_l1_loss_v2_mean_beta1', target_api='asnumpy.smooth_l1_loss', provenance='custom_generation_candidate', route='ascendc_registry_candidate', signature_variants=({'positional_args': 2, 'keyword_args': ()},), dtype_rules=('float32',), shape_rules=('rank>=1', 'contiguous'), source='numpilot://local-llm-generation/operator/smooth_l1_loss', license='CANN-Open-Software-License-Agreement-2.0', target_soc='ascend910b', cann_version='9.0.0')
def smooth_l1_loss_v2_mean_beta1(predict, label):
    """SmoothL1V2 的固定 mean/beta=1 NumPy oracle。"""
    error = np.subtract(predict, label)
    absolute = np.abs(error)
    bounded = np.minimum(absolute, np.float32(1.0))
    quadratic = np.divide(np.multiply(bounded, bounded), np.float32(2.0))
    linear = np.subtract(absolute, bounded)
    return np.mean(np.add(quadratic, linear))

@operator_spec(name='apply_gradient_descent', version='1', numpy_oracle='apply_gradient_descent_oracle', target_api='asnumpy.apply_gradient_descent', provenance='custom_generation_candidate', route='ascendc_registry_candidate', signature_variants=({'positional_args': 3, 'keyword_args': ()},), dtype_rules=('float32',), shape_rules=('rank>=1', 'contiguous'), source='numpilot://local-llm-generation/operator/apply_gradient_descent', license='CANN-Open-Software-License-Agreement-2.0', target_soc='ascend910b', cann_version='9.0.0')
def apply_gradient_descent_oracle(var, alpha, delta):
    """ApplyGradientDescent 的非原地 NumPy oracle。"""
    return np.subtract(var, np.multiply(alpha, delta))

def build_inputs():
    """构造 8×8 内部点、四边各 8 点和 3×3 RBF 的确定性状态。"""
    rng = np.random.default_rng(42)
    interior_axis = np.linspace(0.04, 0.96, 8, dtype=np.float32)
    (grid_x, grid_y) = np.meshgrid(interior_axis, interior_axis)
    x_in = np.ascontiguousarray(grid_x.ravel(), dtype=np.float32)
    y_in = np.ascontiguousarray(grid_y.ravel(), dtype=np.float32)
    boundary_axis = np.linspace(0.0, 1.0, 8, dtype=np.float32)
    x_bd = np.ascontiguousarray(np.concatenate((np.zeros_like(boundary_axis), np.ones_like(boundary_axis), boundary_axis, boundary_axis)), dtype=np.float32)
    y_bd = np.ascontiguousarray(np.concatenate((boundary_axis, boundary_axis, np.zeros_like(boundary_axis), np.ones_like(boundary_axis))), dtype=np.float32)
    centre_axis = np.linspace(0.15, 0.85, 3, dtype=np.float32)
    (centre_x, centre_y) = np.meshgrid(centre_axis, centre_axis)
    cx = np.ascontiguousarray(centre_x.ravel(), dtype=np.float32)
    cy = np.ascontiguousarray(centre_y.ravel(), dtype=np.float32)
    width = np.full(9, np.float32(0.18), dtype=np.float32)
    weight = rng.normal(0.0, 0.05, 9).astype(np.float32)
    source = np.multiply(np.float32(2.0 * np.pi * np.pi), np.multiply(np.sin(np.multiply(np.float32(np.pi), x_in)), np.sin(np.multiply(np.float32(np.pi), y_in)))).astype(np.float32)
    boundary_target = np.zeros_like(x_bd)
    source[[5, 37]] += np.array([10.0, -10.0], dtype=np.float32)
    boundary_target[[3, 22]] += np.array([-10.0, 10.0], dtype=np.float32)
    count_in = x_in.size
    count_bd = x_bd.size
    count_rbf = weight.size
    ones_rbf = np.ones(count_rbf, dtype=np.float32)
    ones_in = np.ones(count_in, dtype=np.float32)
    ones_bd = np.ones(count_bd, dtype=np.float32)
    return (x_in, y_in, np.ascontiguousarray(source, dtype=np.float32), x_bd, y_bd, np.ascontiguousarray(boundary_target, dtype=np.float32), weight, cx, cy, width, ones_rbf, ones_in, ones_bd, np.full(count_in, np.float32(1.0 / count_in), dtype=np.float32), np.full(count_bd, np.float32(1.0 / count_bd), dtype=np.float32), np.full((count_in, count_rbf), np.float32(-0.5), dtype=np.float32), np.full((count_bd, count_rbf), np.float32(-0.5), dtype=np.float32), np.full((count_in, count_rbf), np.float32(2.0), dtype=np.float32), np.full((count_in, count_rbf), np.float32(-4.0), dtype=np.float32), np.full((count_in, count_rbf), np.float32(4.0), dtype=np.float32), np.full(count_in, np.float32(-1.0), dtype=np.float32), np.full(count_in, np.float32(1.0), dtype=np.float32), np.full(count_bd, np.float32(-1.0), dtype=np.float32), np.full(count_bd, np.float32(1.0), dtype=np.float32), np.array([np.float32(5.0)], dtype=np.float32), np.array([np.float32(0.003)], dtype=np.float32), np.array([np.float32(0.0)], dtype=np.float32), np.array([np.float32(0.0)], dtype=np.float32), np.full(count_rbf, np.float32(0.07), dtype=np.float32), np.full(count_rbf, np.float32(0.3), dtype=np.float32))

def forward(x_in, y_in, source, x_bd, y_bd, boundary_target, weight, centre_x, centre_y, width, ones_rbf, ones_in, ones_bd, mean_rows_in, mean_rows_bd, neg_half_in, neg_half_bd, two_in, neg_four_in, four_in, neg_one_in, one_in, neg_one_bd, one_bd, boundary_loss_weight, alpha_weight, alpha_centre, alpha_width, width_min, width_max):
    """在一个可迁移的 straight-line NPU 岛内完成完整训练迭代。"""
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
    if isinstance(ones_in, np.ndarray):
        ones_in = ap.ndarray.from_numpy(ones_in)
    if isinstance(ones_bd, np.ndarray):
        ones_bd = ap.ndarray.from_numpy(ones_bd)
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
    width_squared_vector = ap.multiply(width, width)
    x_in_matrix = ap.outer(x_in, ones_rbf)
    cx_in_matrix = ap.outer(ones_in, centre_x)
    y_in_matrix = ap.outer(y_in, ones_rbf)
    cy_in_matrix = ap.outer(ones_in, centre_y)
    dx_in = ap.subtract(x_in_matrix, cx_in_matrix)
    dy_in = ap.subtract(y_in_matrix, cy_in_matrix)
    radius_squared_in = ap.add(ap.multiply(dx_in, dx_in), ap.multiply(dy_in, dy_in))
    width_squared_in = ap.outer(ones_in, width_squared_vector)
    phi_in = ap.exp(ap.multiply(ap.divide(radius_squared_in, width_squared_in), neg_half_in))
    width_fourth_in = ap.multiply(width_squared_in, width_squared_in)
    laplace_factor_in = ap.subtract(ap.divide(radius_squared_in, width_fourth_in), ap.divide(two_in, width_squared_in))
    laplacian_in = ap.matmul(ap.multiply(phi_in, laplace_factor_in), weight)
    negative_laplacian = ap.negative(laplacian_in)
    x_bd_matrix = ap.outer(x_bd, ones_rbf)
    cx_bd_matrix = ap.outer(ones_bd, centre_x)
    y_bd_matrix = ap.outer(y_bd, ones_rbf)
    cy_bd_matrix = ap.outer(ones_bd, centre_y)
    dx_bd = ap.subtract(x_bd_matrix, cx_bd_matrix)
    dy_bd = ap.subtract(y_bd_matrix, cy_bd_matrix)
    radius_squared_bd = ap.add(ap.multiply(dx_bd, dx_bd), ap.multiply(dy_bd, dy_bd))
    width_squared_bd = ap.outer(ones_bd, width_squared_vector)
    phi_bd = ap.exp(ap.multiply(ap.divide(radius_squared_bd, width_squared_bd), neg_half_bd))
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
    pde_weight_integrand = ap.multiply(ap.multiply(phi_in, laplace_factor_in), gradient_matrix_pde)
    gradient_weight_pde = ap.negative(ap.matmul(mean_rows_in, pde_weight_integrand))
    width_root_in = ap.sqrt(width_squared_in)
    width_cubed_in = ap.multiply(width_squared_in, width_root_in)
    width_fifth_in = ap.multiply(width_fourth_in, width_root_in)
    dphi_centre_x_in = ap.multiply(ap.divide(dx_in, width_squared_in), phi_in)
    dphi_centre_y_in = ap.multiply(ap.divide(dy_in, width_squared_in), phi_in)
    dfactor_centre_x_in = ap.divide(ap.multiply(two_in, dx_in), width_fourth_in)
    dfactor_centre_y_in = ap.divide(ap.multiply(two_in, dy_in), width_fourth_in)
    dphi_width_in = ap.multiply(ap.divide(radius_squared_in, width_cubed_in), phi_in)
    dfactor_width_in = ap.add(ap.divide(ap.multiply(neg_four_in, radius_squared_in), width_fifth_in), ap.divide(four_in, width_cubed_in))
    weight_matrix_in = ap.outer(ones_in, weight)
    gradient_centre_x_integrand_pde = ap.multiply(ap.multiply(weight_matrix_in, ap.add(ap.multiply(dfactor_centre_x_in, phi_in), ap.multiply(laplace_factor_in, dphi_centre_x_in))), gradient_matrix_pde)
    gradient_centre_y_integrand_pde = ap.multiply(ap.multiply(weight_matrix_in, ap.add(ap.multiply(dfactor_centre_y_in, phi_in), ap.multiply(laplace_factor_in, dphi_centre_y_in))), gradient_matrix_pde)
    gradient_width_integrand_pde = ap.multiply(ap.multiply(weight_matrix_in, ap.add(ap.multiply(dfactor_width_in, phi_in), ap.multiply(laplace_factor_in, dphi_width_in))), gradient_matrix_pde)
    gradient_centre_x_pde = ap.negative(ap.matmul(mean_rows_in, gradient_centre_x_integrand_pde))
    gradient_centre_y_pde = ap.negative(ap.matmul(mean_rows_in, gradient_centre_y_integrand_pde))
    gradient_width_pde = ap.negative(ap.matmul(mean_rows_in, gradient_width_integrand_pde))
    gradient_weight_bd = ap.multiply(boundary_loss_weight, ap.matmul(mean_rows_bd, ap.multiply(phi_bd, gradient_matrix_bd)))
    width_root_bd = ap.sqrt(width_squared_bd)
    width_cubed_bd = ap.multiply(width_squared_bd, width_root_bd)
    weight_matrix_bd = ap.outer(ones_bd, weight)
    common_bd = ap.multiply(ap.multiply(weight_matrix_bd, phi_bd), gradient_matrix_bd)
    gradient_centre_x_bd = ap.multiply(boundary_loss_weight, ap.matmul(mean_rows_bd, ap.multiply(ap.divide(dx_bd, width_squared_bd), common_bd)))
    gradient_centre_y_bd = ap.multiply(boundary_loss_weight, ap.matmul(mean_rows_bd, ap.multiply(ap.divide(dy_bd, width_squared_bd), common_bd)))
    gradient_width_bd = ap.multiply(boundary_loss_weight, ap.matmul(mean_rows_bd, ap.multiply(ap.divide(radius_squared_bd, width_cubed_bd), common_bd)))
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