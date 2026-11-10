# Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
# Licensed under the Apache License, Version 2.0. See source/LICENSE.
# Original: gpu4pyscf/lib/cupy_helper.py:1224-1241, sandwich_dot.
# Upstream commit: 5fde981316127cd3506d9d3c7d5633647fcc4508.
# NumPilot demo adaptation: real float32, one 2-D operator, two explicit matmuls.
# This is a scientific-kernel specialization, not a full SCF/DFT calculation.
# The original supports complex/batched arrays and an allocation/out interface;
# those branches are outside this demonstration. Source provenance is archived.
"""Transform an AO-basis operator A into an orbital basis: C.T @ A @ C.

The deterministic synthetic inputs model a real symmetric operator and dense
orbital coefficients. They are not molecular integrals or a converged wavefunction.
Input preparation makes both orientations of C C-contiguous to expose the same
two rank-2 matmul calls on NumPy and AsNumpy. It is excluded from forward timing.
"""
import os
import numpy as np


SEED = 20261110


def build_inputs():
    n = int(os.environ.get("NUMPILOT_SANDWICH_N", "2048"))
    m = int(os.environ.get("NUMPILOT_SANDWICH_M", "1024"))
    if n < 1 or m < 1 or m > n:
        raise ValueError("require 1 <= m <= n; input sizes are never reduced automatically")
    rng = np.random.default_rng(SEED)
    raw = rng.standard_normal((n, n), dtype=np.float32)
    scale = np.sqrt(np.float32(n))
    A = np.ascontiguousarray((raw + raw.T) * (np.float32(0.5) / scale))
    C = np.ascontiguousarray(rng.standard_normal((n, m), dtype=np.float32) / scale)
    C_T = np.ascontiguousarray(C.T)
    return A, C_T, C


def forward(A, C_T, C):
    intermediate = np.matmul(C_T, A)
    return np.matmul(intermediate, C)
