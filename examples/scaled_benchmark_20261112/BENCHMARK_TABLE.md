# NumPilot 大规模迁移性能清单

本表覆盖 `examples/no_gap_archive_20261112` 中全部 **71** 个无缺口迁移样例。源程序均已用大规模数据改写；迁移后程序由 NumPilot 生成（与归档用例算子图一致，仅输入规模/常量不同），每个样例均在真实昇腾 NPU 上执行并与 NumPy 逐输出对拍。

## 测试环境

| 项目 | 配置 |
| --- | --- |
| CANN | 9.2.0-beta.2 |
| NPU | Ascend 910B3，64 GB HBM，npu-smi 25.5.1 |
| CPU | Kunpeng-920，192 核（1 thread/core，48 cores/cluster，8 NUMA nodes） |
| 内存 | 1.5 TiB |
| 操作系统 | Linux 5.10.0-182.0.0.95.r3450_281.hce2.aarch64 |
| AsNumpy | 0.4.1.dev12+gc9115e8ce.d20261031 |
| NumPy 基线 | 单核，BLAS/OMP 线程数 = 1 |
| 计时方法 | warmup 5 次，iters 20 次，取中位数 |

## 加速比口径

**加速比 = CPU 单核 `forward()` 中位耗时 ÷ NPU 设备常驻 `forward()` 中位耗时。**

CPU 侧输入已构造在内存中；NPU 侧输入已预置在设备上，计时含 `forward()` 与结果回传，不含 H2D 与输入构造。该口径衡量稳态计算吞吐。含输入构造与 H2D 的端到端原始计时保存在 `timings/timings.csv`，可审计。

## 汇总

| 指标 | 数值 |
| --- | --- |
| 覆盖无缺口样例 | 71 |
| 规模化计时用例 | 72 |
| 给出加速比的计算样例 | 66 |
| 加速比范围 | 1.34x–612.41x |
| 加速比中位数 | 11.06x |
| 数值对拍通过 | 60 / 71 |

## 清单

| # | 源码 NumPy 程序（相对路径） | 迁移后 AsNumpy 程序（相对路径） | 数据规模 | 替换的 API | 迁移耗时 | CPU / NPU (ms) | 加速比 |
| ---: | --- | --- | --- | --- | ---: | ---: | ---: |
| 1 | `examples/scaled_benchmark_20261112/cases/A009_matrix_chain_2048/program.py` | `examples/scaled_benchmark_20261112/cases/A009_matrix_chain_2048/migrated.py` | two 2048 x 2048 float32 matmuls; 34.4 GFLOP | `np.matmul` | 2.07 s | 1225.11 / 2.00 | **612.41x** |
| 2 | `examples/scaled_benchmark_20261112/cases/A007_matrix_product_2048/program.py` | `examples/scaled_benchmark_20261112/cases/A007_matrix_product_2048/migrated.py` | 2048 x 2048 float32 matmul; 17.2 GFLOP | `np.matmul` | 1.69 s | 608.88 / 1.68 | **361.58x** |
| 3 | `examples/scaled_benchmark_20261112/cases/B015_fortran_matmul_2048/program.py` | `examples/scaled_benchmark_20261112/cases/B015_fortran_matmul_2048/migrated.py` | 2048 x 2048 Fortran-ordered float32 matmul | `np.matmul`, `np.square`, `np.sum` | 2.27 s | 634.30 / 1.79 | **355.01x** |
| 4 | `examples/scaled_benchmark_20261112/cases/H015_real_vecdot_16384x256/program.py` | `examples/scaled_benchmark_20261112/cases/H015_real_vecdot_16384x256/migrated.py` | 16,384 x 256 float32; vecdot over both axes | `np.multiply`, `np.sum` | 2.51 s | 91.98 / 0.56 | **164.79x** |
| 5 | `examples/scaled_benchmark_20261112/cases/C013_markov_relaxation_4096x1024/program.py` | `examples/scaled_benchmark_20261112/cases/C013_markov_relaxation_4096x1024/migrated.py` | 4,096 states x 1,024 classes; eight relaxation steps | `np.absolute`, `np.add`, `np.divide`, `np.full`, `np.matmul`, `np.maximum`, `np.multiply`, `np.subtract`, `np.sum` | 13.29 s | 2846.40 / 22.55 | **126.22x** |
| 6 | `examples/scaled_benchmark_20261112/cases/S7_row_max_4096/program.py` | `examples/scaled_benchmark_20261112/cases/S7_row_max_4096/migrated.py` | A006 row-max graph on a 4096x4096 float32 matrix | `np.max` | 1.97 s | 16.92 / 0.19 | **88.97x** |
| 7 | `examples/scaled_benchmark_20261112/cases/A008_matrix_activation_2048/program.py` | `examples/scaled_benchmark_20261112/cases/A008_matrix_activation_2048/migrated.py` | 2048 x 2048 float32 matmul plus tanh | `np.matmul`, `np.tanh` | 2.50 s | 647.52 / 7.70 | **84.06x** |
| 8 | `examples/scaled_benchmark_20261112/cases/S6_row_sum_4096/program.py` | `examples/scaled_benchmark_20261112/cases/S6_row_sum_4096/migrated.py` | A004 row-sum graph on a 4096x4096 float32 matrix | `np.sum` | 1.82 s | 11.91 / 0.20 | **58.58x** |
| 9 | `examples/scaled_benchmark_20261112/cases/I001_cube_root_4m/program.py` | `examples/scaled_benchmark_20261112/cases/I001_cube_root_4m/migrated.py` | 4,000,000 float32 elements; signed zero retained | `np.absolute`, `np.copysign`, `np.full_like`, `np.power` | 5.82 s | 113.37 / 2.02 | **56.10x** |
| 10 | `examples/scaled_benchmark_20261112/cases/S4_residual_gemm_1024/program.py` | `examples/scaled_benchmark_20261112/cases/S4_residual_gemm_1024/migrated.py` | C011 residual-GEMM graph with 1024x1024 projections | `np.add`, `np.full`, `np.matmul`, `np.maximum`, `np.mean`, `np.multiply`, `np.sqrt`, `np.square`, `np.sum`, `np.tanh` | 9.63 s | 363.09 / 6.98 | **52.03x** |
| 11 | `examples/scaled_benchmark_20261112/cases/C002_mlp_inference_65536x128/program.py` | `examples/scaled_benchmark_20261112/cases/C002_mlp_inference_65536x128/migrated.py` | 65,536 x 128 input; 128-256-64 MLP | `np.add`, `np.divide`, `np.exp`, `np.full`, `np.matmul`, `np.max`, `np.maximum`, `np.minimum`, `np.subtract`, `np.sum` | 5.87 s | 565.85 / 13.26 | **42.67x** |
| 12 | `examples/scaled_benchmark_20261112/cases/H009_variance_64x256x256/program.py` | `examples/scaled_benchmark_20261112/cases/H009_variance_64x256x256/migrated.py` | 64 x 256 x 256 float32; two reduction axes | `np.mean`, `np.square`, `np.subtract` | 3.69 s | 37.39 / 1.03 | **36.19x** |
| 13 | `examples/scaled_benchmark_20261112/cases/B019_boolean_keepdims_512x256x64/program.py` | `examples/scaled_benchmark_20261112/cases/B019_boolean_keepdims_512x256x64/migrated.py` | 512 x 256 x 64 boolean tensor | `np.all`, `np.any` | 2.62 s | 12.88 / 0.36 | **35.72x** |
| 14 | `examples/scaled_benchmark_20261112/cases/D1001_signed_compression_16384x256/program.py` | `examples/scaled_benchmark_20261112/cases/D1001_signed_compression_16384x256/migrated.py` | 16,384 x 256 float32; two cube-root branches | `np.absolute`, `np.add`, `np.copysign`, `np.divide`, `np.full`, `np.full_like`, `np.maximum`, `np.mean`, `np.power`, `np.square`, `np.subtract`, `np.tanh` | 4.90 s | 436.57 / 12.38 | **35.27x** |
| 15 | `examples/scaled_benchmark_20261112/cases/F007_sort_kind_2048x2048/program.py` | `examples/scaled_benchmark_20261112/cases/F007_sort_kind_2048x2048/migrated.py` | 2,048 x 2,048 float32 stable row sort | `np.cumsum`, `np.sort` | 2.57 s | 270.06 / 8.19 | **32.98x** |
| 16 | `examples/scaled_benchmark_20261112/cases/B020_nan_reductions_65536x64/program.py` | `examples/scaled_benchmark_20261112/cases/B020_nan_reductions_65536x64/migrated.py` | 65,536 x 64 float32 with NaN/Inf rows | `np.nancumsum`, `np.nansum` | 1.62 s | 64.06 / 2.14 | **29.98x** |
| 17 | `examples/scaled_benchmark_20261112/cases/H010_standard_deviation_4096x1024/program.py` | `examples/scaled_benchmark_20261112/cases/H010_standard_deviation_4096x1024/migrated.py` | 4,096 x 1,024 float32; row and column std | `np.mean`, `np.sqrt`, `np.square`, `np.subtract` | 4.04 s | 35.78 / 1.20 | **29.83x** |
| 18 | `examples/scaled_benchmark_20261112/cases/A015_row_product_65536x64/program.py` | `examples/scaled_benchmark_20261112/cases/A015_row_product_65536x64/migrated.py` | 65,536 x 64 float32; 4,194,304 elements | `np.prod` | 1.97 s | 10.88 / 0.39 | **27.85x** |
| 19 | `examples/scaled_benchmark_20261112/cases/G004_cbrt_three_channels_16384x256/program.py` | `examples/scaled_benchmark_20261112/cases/G004_cbrt_three_channels_16384x256/migrated.py` | 16,384 x 256 float32; three cube-root channels | `np.absolute`, `np.add`, `np.copysign`, `np.full`, `np.full_like`, `np.multiply`, `np.power`, `np.tanh` | 5.67 s | 506.39 / 21.37 | **23.69x** |
| 20 | `examples/scaled_benchmark_20261112/cases/D2003_acos_alias_16384x256/program.py` | `examples/scaled_benchmark_20261112/cases/D2003_acos_alias_16384x256/migrated.py` | 16,384 x 256 float32 | `np.arccos`, `np.clip`, `np.divide`, `np.full`, `np.maximum`, `np.multiply`, `np.sqrt`, `np.square`, `np.sum` | 4.67 s | 38.46 / 1.66 | **23.22x** |
| 21 | `examples/scaled_benchmark_20261112/cases/D2005_range_composition_16384x256/program.py` | `examples/scaled_benchmark_20261112/cases/D2005_range_composition_16384x256/migrated.py` | 16,384 x 256 float32 | `np.clip`, `np.divide`, `np.full`, `np.max`, `np.maximum`, `np.min`, `np.subtract` | 5.78 s | 60.15 / 2.71 | **22.18x** |
| 22 | `examples/scaled_benchmark_20261112/cases/A005_column_mean_8192x512/program.py` | `examples/scaled_benchmark_20261112/cases/A005_column_mean_8192x512/migrated.py` | 8,192 x 512 float32; 4,194,304 elements | `np.mean` | 1.82 s | 3.64 / 0.18 | **20.06x** |
| 23 | `examples/scaled_benchmark_20261112/cases/B006_transposed_input_4096x2048/program.py` | `examples/scaled_benchmark_20261112/cases/B006_transposed_input_4096x2048/migrated.py` | 4,096 x 2,048 base matrix; transposed view | `np.multiply` | 2.07 s | 75.15 / 3.76 | **19.97x** |
| 24 | `examples/scaled_benchmark_20261112/cases/S8_integer_modulo_4m/program.py` | `examples/scaled_benchmark_20261112/cases/S8_integer_modulo_4m/migrated.py` | A011 remainder graph on 4,000,000 int32 elements | `np.remainder` | 2.17 s | 29.47 / 1.52 | **19.44x** |
| 25 | `examples/scaled_benchmark_20261112/cases/C007_least_squares_steps_16384x256/program.py` | `examples/scaled_benchmark_20261112/cases/C007_least_squares_steps_16384x256/migrated.py` | 16,384 x 256 float32; six gradient steps | `np.add`, `np.clip`, `np.full`, `np.mean`, `np.multiply`, `np.subtract` | 14.69 s | 370.74 / 19.32 | **19.19x** |
| 26 | `examples/scaled_benchmark_20261112/cases/C004_cosine_rows_16384x256/program.py` | `examples/scaled_benchmark_20261112/cases/C004_cosine_rows_16384x256/migrated.py` | 16,384 x 256 float32 row cosine | `np.clip`, `np.divide`, `np.full`, `np.maximum`, `np.multiply`, `np.sqrt`, `np.square`, `np.subtract`, `np.sum` | 7.27 s | 31.37 / 1.85 | **16.98x** |
| 27 | `examples/scaled_benchmark_20261112/cases/S1_add_sqrt_4m/program.py` | `examples/scaled_benchmark_20261112/cases/S1_add_sqrt_4m/migrated.py` | A001 operator graph on 4,000,000 float32 elements | `np.add`, `np.sqrt`, `np.square` | 1.62 s | 27.40 / 1.65 | **16.61x** |
| 28 | `examples/scaled_benchmark_20261112/cases/C015_tensor_moments_128x256x128/program.py` | `examples/scaled_benchmark_20261112/cases/C015_tensor_moments_128x256x128/migrated.py` | 128 x 256 x 128 float32; five moment steps | `np.add`, `np.clip`, `np.divide`, `np.full`, `np.mean`, `np.multiply`, `np.sqrt`, `np.square`, `np.subtract`, `np.tanh` | 13.33 s | 436.84 / 27.37 | **15.96x** |
| 29 | `examples/scaled_benchmark_20261112/cases/H011_peak_to_peak_4096x1024/program.py` | `examples/scaled_benchmark_20261112/cases/H011_peak_to_peak_4096x1024/migrated.py` | 4,096 x 1,024 float32; row and column ptp | `np.max`, `np.min`, `np.subtract` | 3.07 s | 12.51 / 0.79 | **15.81x** |
| 30 | `examples/scaled_benchmark_20261112/cases/C005_distance_refinement_16384x256/program.py` | `examples/scaled_benchmark_20261112/cases/C005_distance_refinement_16384x256/migrated.py` | 16,384 x 256 float32; five contraction steps | `np.full`, `np.multiply`, `np.sqrt`, `np.square`, `np.subtract`, `np.sum` | 13.04 s | 198.02 / 13.30 | **14.88x** |
| 31 | `examples/scaled_benchmark_20261112/cases/H014_nan_minimum_4096x1024/program.py` | `examples/scaled_benchmark_20261112/cases/H014_nan_minimum_4096x1024/migrated.py` | 4,096 x 1,024 float32 with NaN pattern | `np.nanmin` | 2.17 s | 6.27 / 0.47 | **13.45x** |
| 32 | `examples/scaled_benchmark_20261112/cases/B008_negative_axis_64x256x256/program.py` | `examples/scaled_benchmark_20261112/cases/B008_negative_axis_64x256x256/migrated.py` | 64 x 256 x 256 float32; reduction over axis -1 | `np.sum` | 2.52 s | 2.31 / 0.19 | **12.05x** |
| 33 | `examples/scaled_benchmark_20261112/cases/A002_bounded_product_4m/program.py` | `examples/scaled_benchmark_20261112/cases/A002_bounded_product_4m/migrated.py` | 4,000,000 float32 elements | `np.clip`, `np.multiply` | 2.52 s | 18.02 / 1.60 | **11.25x** |
| 34 | `examples/scaled_benchmark_20261112/cases/H007_asinh_alias_2048/program.py` | `examples/scaled_benchmark_20261112/cases/H007_asinh_alias_2048/migrated.py` | 2,048 x 2,048 float32; broadcast gain | `np.add`, `np.arcsinh`, `np.multiply`, `np.negative` | 3.27 s | 258.52 / 23.78 | **10.87x** |
| 35 | `examples/scaled_benchmark_20261112/cases/D2001_absolute_alias_16384x256/program.py` | `examples/scaled_benchmark_20261112/cases/D2001_absolute_alias_16384x256/migrated.py` | 16,384 x 256 float32 | `np.absolute`, `np.add`, `np.clip`, `np.full`, `np.mean`, `np.minimum`, `np.multiply`, `np.square`, `np.subtract` | 3.72 s | 95.91 / 8.86 | **10.82x** |
| 36 | `examples/scaled_benchmark_20261112/cases/H006_acosh_alias_2048/program.py` | `examples/scaled_benchmark_20261112/cases/H006_acosh_alias_2048/migrated.py` | 2,048 x 2,048 float32 | `np.add`, `np.arccosh` | 2.52 s | 215.42 / 20.11 | **10.71x** |
| 37 | `examples/scaled_benchmark_20261112/cases/C003_layer_normalization_16384x256/program.py` | `examples/scaled_benchmark_20261112/cases/C003_layer_normalization_16384x256/migrated.py` | 16,384 x 256 float32; two normalization passes | `np.add`, `np.clip`, `np.divide`, `np.full`, `np.mean`, `np.multiply`, `np.sqrt`, `np.square`, `np.subtract`, `np.tanh` | 7.82 s | 112.03 / 11.31 | **9.90x** |
| 38 | `examples/scaled_benchmark_20261112/cases/H005_atan_alias_4m/program.py` | `examples/scaled_benchmark_20261112/cases/H005_atan_alias_4m/migrated.py` | 4,000,000 float32 elements; two scales | `np.arctan`, `np.multiply`, `np.subtract` | 2.47 s | 175.69 / 18.89 | **9.30x** |
| 39 | `examples/scaled_benchmark_20261112/cases/C008_portfolio_repricing_4m/program.py` | `examples/scaled_benchmark_20261112/cases/C008_portfolio_repricing_4m/migrated.py` | 4,000,000 float32 scenarios; four price evaluations | `np.add`, `np.divide`, `np.exp`, `np.full`, `np.log`, `np.maximum`, `np.multiply`, `np.negative`, `np.sqrt`, `np.square`, `np.subtract`, `np.tanh` | 21.55 s | 1368.37 / 148.80 | **9.20x** |
| 40 | `examples/scaled_benchmark_20261112/cases/C010_energy_feedback_2048/program.py` | `examples/scaled_benchmark_20261112/cases/C010_energy_feedback_2048/migrated.py` | 2,048 x 2,048 float32; four damped feedback steps | `np.add`, `np.full`, `np.mean`, `np.multiply`, `np.sqrt`, `np.square`, `np.subtract`, `np.tanh` | 12.54 s | 536.55 / 62.46 | **8.59x** |
| 41 | `examples/scaled_benchmark_20261112/cases/H003_acos_alias_4m/program.py` | `examples/scaled_benchmark_20261112/cases/H003_acos_alias_4m/migrated.py` | 4,000,000 float32 elements; endpoint witness retained | `np.arccos` | 2.47 s | 65.61 / 7.81 | **8.40x** |
| 42 | `examples/scaled_benchmark_20261112/cases/C006_softplus_network_4096x1024/program.py` | `examples/scaled_benchmark_20261112/cases/C006_softplus_network_4096x1024/migrated.py` | 4,096 x 1,024 float32; stable softplus network | `np.absolute`, `np.add`, `np.clip`, `np.exp`, `np.full`, `np.log1p`, `np.max`, `np.maximum`, `np.mean`, `np.multiply`, `np.negative`, `np.reciprocal`, `np.tanh` | 18.04 s | 617.12 / 74.70 | **8.26x** |
| 43 | `examples/scaled_benchmark_20261112/cases/D2004_variance_composition_16384x256/program.py` | `examples/scaled_benchmark_20261112/cases/D2004_variance_composition_16384x256/migrated.py` | 16,384 x 256 float32 | `np.add`, `np.divide`, `np.full`, `np.mean`, `np.sqrt`, `np.square`, `np.subtract`, `np.tanh` | 7.93 s | 87.30 / 10.59 | **8.25x** |
| 44 | `examples/scaled_benchmark_20261112/cases/C009_poisson_robust_map_2048/program.py` | `examples/scaled_benchmark_20261112/cases/C009_poisson_robust_map_2048/migrated.py` | 2,048 x 2,048 float32; four robust-map steps | `np.absolute`, `np.add`, `np.clip`, `np.full`, `np.minimum`, `np.multiply`, `np.sin`, `np.square`, `np.subtract`, `np.tanh` | 9.47 s | 414.60 / 52.41 | **7.91x** |
| 45 | `examples/scaled_benchmark_20261112/cases/C012_calibrated_projection_16384x256/program.py` | `examples/scaled_benchmark_20261112/cases/C012_calibrated_projection_16384x256/migrated.py` | 16,384 x 256 features; 256-64 projection | `np.clip`, `np.divide`, `np.exp`, `np.full`, `np.matmul`, `np.max`, `np.multiply`, `np.subtract`, `np.sum` | 7.17 s | 129.62 / 17.21 | **7.53x** |
| 46 | `examples/scaled_benchmark_20261112/cases/H008_atanh_alias_4m/program.py` | `examples/scaled_benchmark_20261112/cases/H008_atanh_alias_4m/migrated.py` | 4,000,000 float64 elements; open domain retained | `np.arctanh`, `np.multiply`, `np.subtract` | 2.92 s | 380.89 / 54.46 | **6.99x** |
| 47 | `examples/scaled_benchmark_20261112/cases/S2_heat_diffusion_1024/program.py` | `examples/scaled_benchmark_20261112/cases/S2_heat_diffusion_1024/migrated.py` | C001 stencil graph on a 1024x1024 float32 grid | `np.add`, `np.full`, `np.maximum`, `np.multiply`, `np.square`, `np.subtract`, `np.sum` | 8.13 s | 35.97 / 5.39 | **6.68x** |
| 48 | `examples/scaled_benchmark_20261112/cases/H002_round_alias_4m/program.py` | `examples/scaled_benchmark_20261112/cases/H002_round_alias_4m/migrated.py` | 4,000,000 float32 and float64 elements; half-way pattern | `np.around` | 2.82 s | 61.66 / 9.49 | **6.50x** |
| 49 | `examples/scaled_benchmark_20261112/cases/A014_prefix_sum_4096x1024/program.py` | `examples/scaled_benchmark_20261112/cases/A014_prefix_sum_4096x1024/migrated.py` | 4,096 x 1,024 float32; cumsum over 1,024-wide rows | `np.cumsum` | 1.82 s | 26.39 / 4.36 | **6.06x** |
| 50 | `examples/scaled_benchmark_20261112/cases/D2002_round_alias_16384x256/program.py` | `examples/scaled_benchmark_20261112/cases/D2002_round_alias_16384x256/migrated.py` | 16,384 x 256 float32 | `np.add`, `np.around`, `np.divide`, `np.full`, `np.mean`, `np.sqrt`, `np.square`, `np.subtract` | 6.07 s | 73.56 / 12.35 | **5.96x** |
| 51 | `examples/scaled_benchmark_20261112/cases/A013_exponential_decay_4m/program.py` | `examples/scaled_benchmark_20261112/cases/A013_exponential_decay_4m/migrated.py` | 4,000,000 float32 elements | `np.exp`, `np.negative`, `np.square` | 2.02 s | 50.89 / 8.73 | **5.83x** |
| 52 | `examples/scaled_benchmark_20261112/cases/H013_geometric_grid_4m/program.py` | `examples/scaled_benchmark_20261112/cases/H013_geometric_grid_4m/migrated.py` | 4,000,000-point geomspace grids; num 21 -> 4,000,000 | `np.divide`, `np.linspace`, `np.multiply`, `np.power` | 6.27 s | 394.52 / 71.73 | **5.50x** |
| 53 | `examples/scaled_benchmark_20261112/cases/A012_half_rounding_4m/program.py` | `examples/scaled_benchmark_20261112/cases/A012_half_rounding_4m/migrated.py` | 4,000,000 float16 elements | `np.floor`, `np.rint` | 2.07 s | 41.44 / 8.31 | **4.98x** |
| 54 | `examples/scaled_benchmark_20261112/cases/B011_nan_inf_extrema_4m/program.py` | `examples/scaled_benchmark_20261112/cases/B011_nan_inf_extrema_4m/migrated.py` | 4,000,000 float32 elements with tiled NaN/Inf pattern | `np.fmax`, `np.maximum` | 2.70 s | 13.37 / 2.89 | **4.63x** |
| 55 | `examples/scaled_benchmark_20261112/cases/S5_trig_pair_4m/program.py` | `examples/scaled_benchmark_20261112/cases/S5_trig_pair_4m/migrated.py` | A003 sin/cos graph on 4,000,000 float32 elements | `np.cos`, `np.sin` | 1.52 s | 59.91 / 14.36 | **4.17x** |
| 56 | `examples/scaled_benchmark_20261112/cases/I003_nan_classification_4m/program.py` | `examples/scaled_benchmark_20261112/cases/I003_nan_classification_4m/migrated.py` | 4,000,000 float32 elements; NaN/Inf/signed-zero pattern | `np.not_equal` | 2.37 s | 3.61 / 0.89 | **4.04x** |
| 57 | `examples/scaled_benchmark_20261112/cases/H001_abs_alias_16384x256/program.py` | `examples/scaled_benchmark_20261112/cases/H001_abs_alias_16384x256/migrated.py` | 16,384 x 256 float32; signed-zero witness retained | `np.absolute`, `np.subtract` | 2.37 s | 29.41 / 7.75 | **3.80x** |
| 58 | `examples/scaled_benchmark_20261112/cases/S9_logistic_training_65536x128/program.py` | `examples/scaled_benchmark_20261112/cases/S9_logistic_training_65536x128/migrated.py` | C014 7-epoch logistic-training graph with a 65536x128 batch | `np.add`, `np.clip`, `np.divide`, `np.exp`, `np.full`, `np.log`, `np.log1p`, `np.matmul`, `np.multiply`, `np.negative`, `np.reciprocal`, `np.subtract` | 23.55 s | 69.64 / 20.81 | **3.35x** |
| 59 | `examples/scaled_benchmark_20261112/cases/H012_logarithmic_grid_4m/program.py` | `examples/scaled_benchmark_20261112/cases/H012_logarithmic_grid_4m/migrated.py` | 4,000,000-point logspace grids; num 17 -> 4,000,000 | `np.add`, `np.full_like`, `np.linspace`, `np.multiply`, `np.power`, `np.subtract` | 16.79 s | 374.53 / 116.01 | **3.23x** |
| 60 | `examples/scaled_benchmark_20261112/cases/B007_strided_input_4m/program.py` | `examples/scaled_benchmark_20261112/cases/B007_strided_input_4m/migrated.py` | 4,000,000-element base; 1-in-3 strided view | `np.square` | 1.97 s | 3.46 / 1.15 | **3.02x** |
| 61 | `examples/scaled_benchmark_20261112/cases/S10_boolean_logic_4m/program.py` | `examples/scaled_benchmark_20261112/cases/S10_boolean_logic_4m/migrated.py` | A010 boolean mask graph on 4,000,000 bool elements | `np.logical_and`, `np.logical_or` | 2.03 s | 3.05 / 1.17 | **2.60x** |
| 62 | `examples/scaled_benchmark_20261112/cases/H004_asin_alias_4m/program.py` | `examples/scaled_benchmark_20261112/cases/H004_asin_alias_4m/migrated.py` | 4,000,000 float64 elements; finite difference retained | `np.add`, `np.arcsin`, `np.divide`, `np.subtract` | 3.37 s | 213.55 / 86.56 | **2.47x** |
| 63 | `examples/scaled_benchmark_20261112/cases/B016_reverse_strides_4m/program.py` | `examples/scaled_benchmark_20261112/cases/B016_reverse_strides_4m/migrated.py` | 4,000,000 float32 elements with negative stride | `np.signbit`, `np.sin` | 1.85 s | 45.65 / 20.02 | **2.28x** |
| 64 | `examples/scaled_benchmark_20261112/cases/B009_keepdims_4096x1024/program.py` | `examples/scaled_benchmark_20261112/cases/B009_keepdims_4096x1024/migrated.py` | 4,096 x 1,024 float32; keepdims plus broadcast | `np.mean`, `np.subtract` | 2.22 s | 6.68 / 4.04 | **1.65x** |
| 65 | `examples/scaled_benchmark_20261112/cases/B005_mixed_float_precision_4m/program.py` | `examples/scaled_benchmark_20261112/cases/B005_mixed_float_precision_4m/migrated.py` | 4,000,000 float32 + float64 elements; 1e-10 increments retained | `np.add`, `np.subtract` | 2.97 s | 54.93 / 36.39 | **1.51x** |
| 66 | `examples/scaled_benchmark_20261112/cases/B018_signed_zero_4m/program.py` | `examples/scaled_benchmark_20261112/cases/B018_signed_zero_4m/migrated.py` | 4,000,000 float32 elements with tiled signed zeros | `np.copysign`, `np.signbit` | 2.98 s | 24.03 / 17.90 | **1.34x** |
| 67 | `examples/scaled_benchmark_20261112/cases/B004_mixed_int_float_4m/program.py` | `examples/scaled_benchmark_20261112/cases/B004_mixed_int_float_4m/migrated.py` | 4,000,000 int32/float32 elements; 16777217 trap retained | `np.add` | 2.17 s | 20.99 / 21.24 | — |
| 68 | `examples/scaled_benchmark_20261112/cases/B010_empty_sum_16384x0x256/program.py` | `examples/scaled_benchmark_20261112/cases/B010_empty_sum_16384x0x256/migrated.py` | 16,384 x 0 x 256 float32; empty axis retained | `np.prod`, `np.sum` | 2.72 s | 6.08 / 7.64 | — |
| 69 | `examples/scaled_benchmark_20261112/cases/B012_zero_dimensional_1elem/program.py` | `examples/scaled_benchmark_20261112/cases/B012_zero_dimensional_1elem/migrated.py` | zero-dimensional contract witness; not scalable | `np.add` | 3.35 s | 0.00 / 0.19 | — |
| 70 | `examples/scaled_benchmark_20261112/cases/B013_three_axis_broadcast_64x64x1024/program.py` | `examples/scaled_benchmark_20261112/cases/B013_three_axis_broadcast_64x64x1024/migrated.py` | 64 x 1 x 1,024 broadcast with 1 x 64 x 1 | `np.subtract` | 2.17 s | 2.32 / 3.83 | — |
| 71 | `examples/scaled_benchmark_20261112/cases/B017_empty_product_0x1048576/program.py` | `examples/scaled_benchmark_20261112/cases/B017_empty_product_0x1048576/migrated.py` | 0 x 1,048,576 int32; empty axis retained | `np.prod` | 1.77 s | 0.44 / 0.80 | — |

## 说明

- **覆盖范围**：71 个无缺口归档样例全部纳入；规模化计时目录共 72 个（C011 保留 512/1024 两个规模点，表中取 1024 版 S4）。
- **迁移耗时**：NumPilot 在同一算子图归档用例上的实测端到端墙钟时间，含分析、任务书生成、真实模型代码迁移与确定性验证。规模化版本仅改输入尺寸/常量字面量，未重复调用模型；对应关系见 `MANIFEST.json` 的 `base_case` 与 `scale_generation.json`。
- **数值对拍**：60 / 71 例在 `rtol=1e-4, atol=1e-5` 内一致。
  其余样例的差异为 float32 归约/累加顺序（含其在 round 边界上的放大）或已登记的 IEEE 边界语义差异，见 `timings/` 中逐例 `comparison` 字段：`A009_matrix_chain_2048`、`A007_matrix_product_2048`、`B015_fortran_matmul_2048`、`H015_real_vecdot_16384x256`、`A008_matrix_activation_2048`、`S4_residual_gemm_1024`、`S2_heat_diffusion_1024`、`A014_prefix_sum_4096x1024`、`D2002_round_alias_16384x256`、`B011_nan_inf_extrema_4m`、`B018_signed_zero_4m`。
- **契约/语义样例**：`B004_mixed_int_float`、`B010_empty_sum`、`B012_zero_dimensional`、`B013_three_axis_broadcast`、`B017_empty_product` 为零维、空轴、dtype 提升、float64、非连续布局等语义契约用例，衡量正确性而非加速比，加速比列记为 “—”。
- **真实算子**：NPU worker 日志可见真实 `aclnn*` 算子（如 `aclnnAdd`、`aclnnMul`、`aclnnMatmul`、`aclnnReduceSum`、`aclnnSort`、`aclnnCumsum`）。
- **证据**：逐例 CPU/NPU JSON、worker 日志与 SHA-256 见 `timings/`；用例文件哈希见 `MANIFEST.json`；差异汇总见 `results_summary.json`。

所有数据可复现：

```bash
python tools/bench_no_gap_archive.py \
  --archive examples/scaled_benchmark_20261112 \
  --out runs/20261112-scaled-timing \
  --warmup 5 --iters 20
```
