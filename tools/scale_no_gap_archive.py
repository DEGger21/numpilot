#!/usr/bin/env python3
"""Generate large-scale performance variants for the archived no-gap cases.

The operator graph (the whole module except build_inputs) is copied verbatim
from the archived case.  Only input sizes, and the few constants that encode
the original input size in the forward (H012/H013 num), are changed.

Usage:
  python tools/scale_no_gap_archive.py [--check]
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SOURCE_ROOT = REPO / "examples/no_gap_archive_20261112/cases"
SCALED_ROOT = REPO / "examples/scaled_benchmark_20261112/cases"

# base cases already represented by the S1-S10 scaled variants
REUSED = {
    "A001_add_sqrt": "S1_add_sqrt_4m",
    "A003_trig_pair": "S5_trig_pair_4m",
    "A004_row_sum": "S6_row_sum_4096",
    "A006_row_max": "S7_row_max_4096",
    "A010_boolean_logic": "S10_boolean_logic_4m",
    "A011_integer_modulo": "S8_integer_modulo_4m",
    "C001_heat_diffusion": "S2_heat_diffusion_1024",
    "C011_residual_gemm": "S4_residual_gemm_1024",
    "C014_logistic_training": "S9_logistic_training_65536x128",
}

# base case -> (dir suffix, one-line scale note, build_inputs body)
SPEC = {
    "A002_bounded_product": (
        "4m",
        "4,000,000 float32 elements",
        """\
    rng = np.random.default_rng(2)
    return {
        "x": rng.standard_normal((4_000_000,), dtype=np.float32),
        "y": rng.standard_normal((4_000_000,), dtype=np.float32),
    }
""",
    ),
    "A005_column_mean": (
        "8192x512",
        "8,192 x 512 float32; 4,194,304 elements",
        """\
    rng = np.random.default_rng(5)
    return {
        "x": rng.standard_normal((8192, 512), dtype=np.float32),
    }
""",
    ),
    "A007_matrix_product": (
        "2048",
        "2048 x 2048 float32 matmul; 17.2 GFLOP",
        """\
    rng = np.random.default_rng(7)
    return {
        "x": rng.standard_normal((2048, 2048), dtype=np.float32),
        "w": rng.standard_normal((2048, 2048), dtype=np.float32),
    }
""",
    ),
    "A008_matrix_activation": (
        "2048",
        "2048 x 2048 float32 matmul plus tanh",
        """\
    rng = np.random.default_rng(8)
    return {
        "x": rng.standard_normal((2048, 2048), dtype=np.float32),
        "w": rng.standard_normal((2048, 2048), dtype=np.float32),
    }
""",
    ),
    "A009_matrix_chain": (
        "2048",
        "two 2048 x 2048 float32 matmuls; 34.4 GFLOP",
        """\
    rng = np.random.default_rng(9)
    return {
        "x": rng.standard_normal((2048, 2048), dtype=np.float32),
        "w": rng.standard_normal((2048, 2048), dtype=np.float32),
        "v": rng.standard_normal((2048, 2048), dtype=np.float32),
    }
""",
    ),
    "A012_half_rounding": (
        "4m",
        "4,000,000 float16 elements",
        """\
    pattern = np.array([-2.5, -0.5, 0.5, 1.5, 2.5], dtype=np.float16)
    return {
        "x": np.tile(pattern, 800_000),
    }
""",
    ),
    "A013_exponential_decay": (
        "4m",
        "4,000,000 float32 elements",
        """\
    rng = np.random.default_rng(13)
    return {
        "x": rng.standard_normal((4_000_000,), dtype=np.float32),
    }
""",
    ),
    "A014_prefix_sum": (
        "4096x1024",
        "4,096 x 1,024 float32; cumsum over 1,024-wide rows",
        """\
    rng = np.random.default_rng(14)
    return {
        "x": rng.standard_normal((4096, 1024), dtype=np.float32),
    }
""",
    ),
    "A015_row_product": (
        "65536x64",
        "65,536 x 64 float32; 4,194,304 elements",
        """\
    return {
        "x": np.linspace(0.8, 1.2, 4_194_304, dtype=np.float32).reshape(65536, 64),
    }
""",
    ),
    "B004_mixed_int_float": (
        "4m",
        "4,000,000 int32/float32 elements; 16777217 trap retained",
        """\
    rng = np.random.default_rng(204)
    x = rng.integers(-1000, 1000, size=(4_000_000,), dtype=np.int32)
    x[0] = 16777217
    return {
        "x": x,
        "y": rng.standard_normal((4_000_000,), dtype=np.float32),
    }
""",
    ),
    "B005_mixed_float_precision": (
        "4m",
        "4,000,000 float32 + float64 elements; 1e-10 increments retained",
        """\
    rng = np.random.default_rng(205)
    return {
        "x": rng.standard_normal((4_000_000,), dtype=np.float32),
        "y": rng.standard_normal((4_000_000,), dtype=np.float64) * 1e-10,
    }
""",
    ),
    "B006_transposed_input": (
        "4096x2048",
        "4,096 x 2,048 base matrix; transposed view",
        """\
    base = np.arange(8_388_608, dtype=np.float32).reshape(4096, 2048)
    return {
        "x": base.T,
        "y": np.full((2048, 4096), 0.5, dtype=np.float32),
    }
""",
    ),
    "B007_strided_input": (
        "4m",
        "4,000,000-element base; 1-in-3 strided view",
        """\
    base = np.linspace(-3, 3, 4_000_000, dtype=np.float32)
    return {
        "x": base[1::3],
    }
""",
    ),
    "B008_negative_axis": (
        "64x256x256",
        "64 x 256 x 256 float32; reduction over axis -1",
        """\
    rng = np.random.default_rng(208)
    return {
        "x": rng.standard_normal((64, 256, 256), dtype=np.float32),
    }
""",
    ),
    "B009_keepdims": (
        "4096x1024",
        "4,096 x 1,024 float32; keepdims plus broadcast",
        """\
    rng = np.random.default_rng(209)
    return {
        "x": rng.standard_normal((4096, 1024), dtype=np.float32),
    }
""",
    ),
    "B010_empty_sum": (
        "16384x0x256",
        "16,384 x 0 x 256 float32; empty axis retained",
        """\
    return {
        "x": np.empty((16384, 0, 256), dtype=np.float32),
    }
""",
    ),
    "B011_nan_inf_extrema": (
        "4m",
        "4,000,000 float32 elements with tiled NaN/Inf pattern",
        """\
    return {
        "x": np.resize(np.array([np.nan, np.inf, -np.inf, 1], dtype=np.float32), (4_000_000,)),
        "y": np.resize(np.array([2, 3, -4, np.nan], dtype=np.float32), (4_000_000,)),
    }
""",
    ),
    "B012_zero_dimensional": (
        "1elem",
        "zero-dimensional contract witness; not scalable",
        """\
    return {
        "x": np.array(3.5, dtype=np.float32),
        "y": np.array(-1.25, dtype=np.float32),
    }
""",
    ),
    "B013_three_axis_broadcast": (
        "64x64x1024",
        "64 x 1 x 1,024 broadcast with 1 x 64 x 1",
        """\
    return {
        "x": np.arange(64 * 1024, dtype=np.float32).reshape(64, 1, 1024),
        "y": np.arange(64, dtype=np.float32).reshape(1, 64, 1),
    }
""",
    ),
    "B015_fortran_matmul": (
        "2048",
        "2048 x 2048 Fortran-ordered float32 matmul",
        """\
    rng = np.random.default_rng(215)
    x = np.asfortranarray(rng.standard_normal((2048, 2048)).astype(np.float32))
    y = np.asfortranarray(rng.standard_normal((2048, 2048)).astype(np.float32))
    return {
        "x": x,
        "y": y,
    }
""",
    ),
    "B016_reverse_strides": (
        "4m",
        "4,000,000 float32 elements with negative stride",
        """\
    base = np.linspace(-1, 1, 4_000_000, dtype=np.float32)
    return {
        "x": base[::-1],
    }
""",
    ),
    "B017_empty_product": (
        "0x1048576",
        "0 x 1,048,576 int32; empty axis retained",
        """\
    return {
        "x": np.empty((0, 1_048_576), dtype=np.int32),
    }
""",
    ),
    "B018_signed_zero": (
        "4m",
        "4,000,000 float32 elements with tiled signed zeros",
        """\
    return {
        "x": np.resize(np.array([0.0, -0.0, 1.0, -1.0], dtype=np.float32), (4_000_000,)),
        "y": np.ones(4_000_000, dtype=np.float32),
    }
""",
    ),
    "B019_boolean_keepdims": (
        "512x256x64",
        "512 x 256 x 64 boolean tensor",
        """\
    rng = np.random.default_rng(219)
    return {
        "x": rng.integers(0, 2, (512, 256, 64)).astype(bool),
    }
""",
    ),
    "B020_nan_reductions": (
        "65536x64",
        "65,536 x 64 float32 with NaN/Inf rows",
        """\
    rng = np.random.default_rng(220)
    x = rng.standard_normal((65536, 64), dtype=np.float32)
    x[:, 0] = np.nan
    x[::3, 1] = np.nan
    x[::5, 2] = np.inf
    x[::7, 3] = -np.inf
    x[::1000, :] = np.nan
    return {
        "x": x,
    }
""",
    ),
    "C002_mlp_inference": (
        "65536x128",
        "65,536 x 128 input; 128-256-64 MLP",
        """\
    rng = np.random.default_rng(302)
    return {
        "x": rng.standard_normal((65536, 128), dtype=np.float32),
        "w1": rng.standard_normal((128, 256), dtype=np.float32) * 0.1,
        "b1": np.linspace(-0.1, 0.1, 256, dtype=np.float32),
        "w2": rng.standard_normal((256, 64), dtype=np.float32) * 0.2,
        "b2": np.zeros(64, dtype=np.float32),
    }
""",
    ),
    "C003_layer_normalization": (
        "16384x256",
        "16,384 x 256 float32; two normalization passes",
        """\
    rng = np.random.default_rng(303)
    x = rng.standard_normal((16384, 256), dtype=np.float32)
    x[0] = 4.0
    return {
        "x": x,
        "gain": np.linspace(0.5, 1.5, 256, dtype=np.float32),
        "bias": np.linspace(-0.2, 0.2, 256, dtype=np.float32),
    }
""",
    ),
    "C004_cosine_rows": (
        "16384x256",
        "16,384 x 256 float32 row cosine",
        """\
    rng = np.random.default_rng(304)
    x = rng.standard_normal((16384, 256), dtype=np.float32)
    y = rng.standard_normal((16384, 256), dtype=np.float32)
    x[0] = 0
    y[1] = 0
    return {
        "x": x,
        "y": y,
    }
""",
    ),
    "C005_distance_refinement": (
        "16384x256",
        "16,384 x 256 float32; five contraction steps",
        """\
    rng = np.random.default_rng(305)
    return {
        "x": rng.standard_normal((16384, 256), dtype=np.float32),
        "target": rng.standard_normal((16384, 256), dtype=np.float32),
        "weight": np.linspace(0.5, 1.0, 256, dtype=np.float32),
    }
""",
    ),
    "C006_softplus_network": (
        "4096x1024",
        "4,096 x 1,024 float32; stable softplus network",
        """\
    x = np.linspace(-80, 80, 4_194_304, dtype=np.float32).reshape(4096, 1024)
    return {
        "x": x,
        "bias": np.linspace(-1, 1, 1024, dtype=np.float32),
        "residual": np.full((4096, 1024), 0.1, dtype=np.float32),
    }
""",
    ),
    "C007_least_squares_steps": (
        "16384x256",
        "16,384 x 256 float32; six gradient steps",
        """\
    rng = np.random.default_rng(307)
    return {
        "predict": rng.standard_normal((16384, 256), dtype=np.float32),
        "label": rng.standard_normal((16384, 256), dtype=np.float32),
        "penalty": np.full((16384, 256), 0.05, dtype=np.float32),
    }
""",
    ),
    "C008_portfolio_repricing": (
        "4m",
        "4,000,000 float32 scenarios; four price evaluations",
        """\
    return {
        "spot": np.linspace(80, 160, 4_000_000, dtype=np.float32),
        "strike": np.linspace(50, 68, 4_000_000, dtype=np.float32),
        "expiry": np.linspace(0.1, 2, 4_000_000, dtype=np.float32),
        "volatility": np.linspace(0.12, 0.45, 4_000_000, dtype=np.float32),
        "rate": np.full(4_000_000, 0.03, dtype=np.float32),
    }
""",
    ),
    "C009_poisson_robust_map": (
        "2048",
        "2,048 x 2,048 float32; four robust-map steps",
        """\
    return {
        "x": np.linspace(0.04, 0.96, 4_194_304, dtype=np.float32).reshape(2048, 2048),
        "y": np.linspace(0.96, 0.04, 4_194_304, dtype=np.float32).reshape(2048, 2048),
        "scale": np.full((2048, 2048), 0.92, dtype=np.float32),
    }
""",
    ),
    "C010_energy_feedback": (
        "2048",
        "2,048 x 2,048 float32; four damped feedback steps",
        """\
    return {
        "x": np.linspace(-2, 2, 4_194_304, dtype=np.float32).reshape(2048, 2048),
        "y": np.linspace(2, -2, 4_194_304, dtype=np.float32).reshape(2048, 2048),
        "drive": np.full((2048, 2048), 0.05, dtype=np.float32),
    }
""",
    ),
    "C012_calibrated_projection": (
        "16384x256",
        "16,384 x 256 features; 256-64 projection",
        """\
    rng = np.random.default_rng(312)
    return {
        "features": rng.standard_normal((16384, 256), dtype=np.float32) * 0.01,
        "weights": rng.standard_normal((256, 64), dtype=np.float32) * 0.02,
        "target": np.full((16384, 64), 0.25, dtype=np.float32),
    }
""",
    ),
    "C013_markov_relaxation": (
        "4096x1024",
        "4,096 states x 1,024 classes; eight relaxation steps",
        """\
    rng = np.random.default_rng(313)
    matrix = rng.uniform(0.1, 1, (1024, 1024)).astype(np.float32)
    matrix /= matrix.sum(axis=1, keepdims=True)
    state = np.zeros((4096, 1024), dtype=np.float32)
    state[:, 0] = 1.0
    return {
        "transition": matrix,
        "state": state,
        "prior": np.full((1, 1024), 1 / 1024, dtype=np.float32),
    }
""",
    ),
    "C015_tensor_moments": (
        "128x256x128",
        "128 x 256 x 128 float32; five moment steps",
        """\
    rng = np.random.default_rng(315)
    return {
        "x": rng.standard_normal((128, 256, 128), dtype=np.float32),
        "drive": rng.standard_normal((128, 1, 128), dtype=np.float32) * 0.1,
        "gain": np.linspace(0.8, 1.2, 128, dtype=np.float32),
    }
""",
    ),
    "D1001_signed_compression": (
        "16384x256",
        "16,384 x 256 float32; two cube-root branches",
        """\
    rng = np.random.default_rng(401)
    return {
        "x": rng.standard_normal((16384, 256), dtype=np.float32),
        "y": rng.standard_normal((16384, 256), dtype=np.float32),
    }
""",
    ),
    "D2001_absolute_alias": (
        "16384x256",
        "16,384 x 256 float32",
        """\
    rng = np.random.default_rng(451)
    return {
        "x": rng.standard_normal((16384, 256), dtype=np.float32),
        "y": rng.standard_normal((16384, 256), dtype=np.float32),
    }
""",
    ),
    "D2002_round_alias": (
        "16384x256",
        "16,384 x 256 float32",
        """\
    rng = np.random.default_rng(452)
    return {
        "x": rng.standard_normal((16384, 256), dtype=np.float32),
        "y": rng.standard_normal((16384, 256), dtype=np.float32),
    }
""",
    ),
    "D2003_acos_alias": (
        "16384x256",
        "16,384 x 256 float32",
        """\
    rng = np.random.default_rng(453)
    return {
        "x": rng.standard_normal((16384, 256), dtype=np.float32),
        "y": rng.standard_normal((16384, 256), dtype=np.float32),
    }
""",
    ),
    "D2004_variance_composition": (
        "16384x256",
        "16,384 x 256 float32",
        """\
    rng = np.random.default_rng(454)
    return {
        "x": rng.standard_normal((16384, 256), dtype=np.float32),
        "y": rng.standard_normal((16384, 256), dtype=np.float32),
    }
""",
    ),
    "D2005_range_composition": (
        "16384x256",
        "16,384 x 256 float32",
        """\
    rng = np.random.default_rng(455)
    return {
        "x": rng.standard_normal((16384, 256), dtype=np.float32),
        "y": rng.standard_normal((16384, 256), dtype=np.float32),
    }
""",
    ),
    "F007_sort_kind": (
        "2048x2048",
        "2,048 x 2,048 float32 stable row sort",
        """\
    rng = np.random.default_rng(707)
    return {
        "x": rng.integers(-100, 100, (2048, 2048)).astype(np.float32),
    }
""",
    ),
    "G004_cbrt_three_channels": (
        "16384x256",
        "16,384 x 256 float32; three cube-root channels",
        """\
    rng = np.random.default_rng(704)
    return {
        "a": rng.standard_normal((16384, 256), dtype=np.float32),
        "b": rng.standard_normal((16384, 256), dtype=np.float32),
        "c": rng.standard_normal((16384, 256), dtype=np.float32),
    }
""",
    ),
    "H001_abs_alias": (
        "16384x256",
        "16,384 x 256 float32; signed-zero witness retained",
        """\
    rng = np.random.default_rng(801)
    x = rng.standard_normal((16384, 256), dtype=np.float32)
    x[0, 0] = -0.0
    return {
        "x": x,
        "y": x[:, ::-1].copy(),
    }
""",
    ),
    "H002_round_alias": (
        "4m",
        "4,000,000 float32 and float64 elements; half-way pattern",
        """\
    return {
        "x": np.resize(np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], dtype=np.float32), (4_000_000,)),
        "y": np.resize(np.array([0.125, 0.375, 0.625, 0.875], dtype=np.float64), (4_000_000,)),
    }
""",
    ),
    "H003_acos_alias": (
        "4m",
        "4,000,000 float32 elements; endpoint witness retained",
        """\
    x = np.linspace(-1, 1, 4_000_000, dtype=np.float32).reshape(2000, 2000)
    return {
        "x": x,
        "y": np.array([-1.0, 0.0, 1.0], dtype=np.float32),
    }
""",
    ),
    "H004_asin_alias": (
        "4m",
        "4,000,000 float64 elements; finite difference retained",
        """\
    x = np.linspace(-0.9, 0.9, 4_000_000, dtype=np.float64)
    return {
        "x": x,
        "offset": np.full(4_000_000, 0.025, dtype=np.float64),
    }
""",
    ),
    "H005_atan_alias": (
        "4m",
        "4,000,000 float32 elements; two scales",
        """\
    x = np.linspace(-100, 100, 4_000_000, dtype=np.float32)
    return {
        "x": x,
        "scale": np.full(4_000_000, 0.25, dtype=np.float32),
    }
""",
    ),
    "H006_acosh_alias": (
        "2048",
        "2,048 x 2,048 float32",
        """\
    distance = np.linspace(0, 8, 4_194_304, dtype=np.float32).reshape(2048, 2048)
    return {
        "x": 1 + distance,
        "y": 1 + distance * distance,
    }
""",
    ),
    "H007_asinh_alias": (
        "2048",
        "2,048 x 2,048 float32; broadcast gain",
        """\
    x = np.linspace(-30, 30, 4_194_304, dtype=np.float32).reshape(2048, 2048)
    return {
        "x": x,
        "gain": np.linspace(0.1, 1.0, 2048, dtype=np.float32),
    }
""",
    ),
    "H008_atanh_alias": (
        "4m",
        "4,000,000 float64 elements; open domain retained",
        """\
    x = np.linspace(-0.95, 0.95, 4_000_000, dtype=np.float64)
    return {
        "x": x,
        "shrink": np.full(4_000_000, 0.5, dtype=np.float64),
    }
""",
    ),
    "H009_variance": (
        "64x256x256",
        "64 x 256 x 256 float32; two reduction axes",
        """\
    rng = np.random.default_rng(809)
    return {
        "x": rng.standard_normal((64, 256, 256), dtype=np.float32),
    }
""",
    ),
    "H010_standard_deviation": (
        "4096x1024",
        "4,096 x 1,024 float32; row and column std",
        """\
    rng = np.random.default_rng(810)
    return {
        "x": rng.standard_normal((4096, 1024), dtype=np.float32),
    }
""",
    ),
    "H011_peak_to_peak": (
        "4096x1024",
        "4,096 x 1,024 float32; row and column ptp",
        """\
    rng = np.random.default_rng(811)
    return {
        "x": rng.standard_normal((4096, 1024), dtype=np.float32),
    }
""",
    ),
    "H012_logarithmic_grid": (
        "4m",
        "4,000,000-point logspace grids; num 17 -> 4,000,000",
        """\
    return {
        "start": np.array(-2.0),
        "stop": np.array(2.0),
    }
""",
    ),
    "H013_geometric_grid": (
        "4m",
        "4,000,000-point geomspace grids; num 21 -> 4,000,000",
        """\
    return {
        "start": np.array(0.125),
        "stop": np.array(128.0),
    }
""",
    ),
    "H014_nan_minimum": (
        "4096x1024",
        "4,096 x 1,024 float32 with NaN pattern",
        """\
    rng = np.random.default_rng(814)
    x = rng.standard_normal((4096, 1024), dtype=np.float32)
    x[::3, ::7] = np.nan
    x[::5, 1] = np.nan
    return {
        "x": x,
    }
""",
    ),
    "H015_real_vecdot": (
        "16384x256",
        "16,384 x 256 float32; vecdot over both axes",
        """\
    rng = np.random.default_rng(815)
    return {
        "x": rng.standard_normal((16384, 256), dtype=np.float32),
        "y": rng.standard_normal((16384, 256), dtype=np.float32),
    }
""",
    ),
    "I001_cube_root": (
        "4m",
        "4,000,000 float32 elements; signed zero retained",
        """\
    x = np.linspace(-27, 27, 4_000_000, dtype=np.float32)
    x[2_000_000] = -0.0
    return {
        "x": x,
    }
""",
    ),
    "I003_nan_classification": (
        "4m",
        "4,000,000 float32 elements; NaN/Inf/signed-zero pattern",
        """\
    return {
        "x": np.resize(np.array([0, -0.0, np.nan, np.inf, -np.inf, -1, 1], dtype=np.float32), (4_000_000,)),
    }
""",
    ),
}

# (base case, old token, new token) applied to the whole program and migrated
# source; only the fixed grid size in the forward uses these literals.
FORWARD_CONSTANTS = {
    "H012_logarithmic_grid": ("17", "4_000_000"),
    "H013_geometric_grid": ("21", "4_000_000"),
}


def _replace_build_inputs(text: str, body: str) -> str:
    tree = ast.parse(text)
    fn = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "build_inputs":
            fn = node
            break
    if fn is None:
        raise SystemExit("build_inputs not found")
    lines = text.splitlines(keepends=True)
    start = fn.lineno - 1
    end = fn.end_lineno
    replacement = "def build_inputs():\n" + body.rstrip() + "\n"
    return "".join(lines[:start]) + replacement + "".join(lines[end:])


def _apply_constants(text: str, old: str, new: str) -> str:
    return re.sub(r"\b%s\b" % re.escape(old), new, text)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(check: bool) -> int:
    missing = sorted(set(p.name for p in SOURCE_ROOT.iterdir() if p.is_dir())
                     - set(SPEC) - set(REUSED))
    if missing:
        raise SystemExit("unmapped source cases: %s" % ", ".join(missing))
    summary = []
    for base in sorted(SPEC):
        tag, note, body = SPEC[base]
        source_dir = SOURCE_ROOT / base
        source_program = source_dir / "program.py"
        source_migrated = source_dir / "migrated.py"
        case_name = "%s_%s" % (base, tag)
        case_dir = SCALED_ROOT / case_name
        program_text = source_program.read_text(encoding="utf-8")
        migrated_text = source_migrated.read_text(encoding="utf-8")
        program_text = _replace_build_inputs(program_text, body)
        migrated_text = _replace_build_inputs(migrated_text, body)
        if base in FORWARD_CONSTANTS:
            old, new = FORWARD_CONSTANTS[base]
            program_text = _apply_constants(program_text, old, new)
            migrated_text = _apply_constants(migrated_text, old, new)
        entry = {
            "base_case": base,
            "case": case_name,
            "scale": note,
            "forward_constants_changed": base in FORWARD_CONSTANTS,
            "source_program_sha256": _sha(source_program),
            "scaled_program_sha256": hashlib.sha256(program_text.encode()).hexdigest(),
            "scaled_migrated_sha256": hashlib.sha256(migrated_text.encode()).hexdigest(),
        }
        if check:
            summary.append(entry)
            continue
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "program.py").write_text(program_text, encoding="utf-8")
        (case_dir / "migrated.py").write_text(migrated_text, encoding="utf-8")
        metadata = {
            "case": case_name,
            "base_case": base,
            "kind": "scaled_performance_variant",
            "note": "%s: %s." % (base, note),
            "provenance": (
                "Operator graph and migrated code derived from the archived "
                "no-gap case; only input sizes%s changed."
                % ("/constants" if base in FORWARD_CONSTANTS else "")
            ),
            "scalable": base != "B012_zero_dimensional",
        }
        (case_dir / "case.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        summary.append(entry)
    (SCALED_ROOT.parent / "scale_generation.json").write_text(
        json.dumps(
            {
                "schema": "numpilot.scaled-variant-generation",
                "schema_version": 1,
                "source_archive": "examples/no_gap_archive_20261112/cases",
                "method": (
                    "build_inputs replaced; forward copied verbatim except "
                    "H012/H013 fixed grid size"
                ),
                "reused": REUSED,
                "generated": summary,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    for item in summary:
        print("%-46s %s" % (item["case"], item["scale"]))
    print("generated %d cases" % len(summary))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    return generate(args.check)


if __name__ == "__main__":
    raise SystemExit(main())
