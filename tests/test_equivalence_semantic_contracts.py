"""Regression tests for fail-closed NumPy equivalence lowering.

These cases exercise semantics that cannot be proven by ordinary numerical
allclose alone: signed zero ordering, broadcast shape, dtype preservation, and
exception behavior.
"""
import numpy as np

from numpilot.benchmark_lowering import lower_benchmark_program


def _lowered(source):
    return lower_benchmark_program(source)


def test_sort_stable_keeps_signed_zero_contract():
    source = """
import numpy as np

def build_inputs():
    return {'x': np.array([0.0, -0.0, 1.0, -1.0], dtype=np.float64)}

def forward(x):
    return {'out': np.sort(x, kind='stable')}
"""
    lowered = _lowered(source)

    # Removing kind can reorder +0/-0. A value-only proof is insufficient.
    assert "np.sort" in lowered.source
    assert "contract" not in [item.kind for item in lowered.rewrites]


def test_logspace_array_inputs_keep_axis_zero_shape_contract():
    source = """
import numpy as np

def build_inputs():
    return {
        'start': np.array([-2.0, -1.0], dtype=np.float64),
        'stop': np.array([2.0, 1.0], dtype=np.float64),
    }

def forward(start, stop):
    return {'out': np.logspace(start, stop, 4)}
"""
    lowered = _lowered(source)

    # The current composition broadcasts (2,) with (4,) and is invalid.
    assert "np.logspace" in lowered.source
    assert "composition" not in [item.kind for item in lowered.rewrites]


def test_logspace_float32_keeps_dtype_contract():
    source = """
import numpy as np

def build_inputs():
    return {
        'start': np.array(-2.0, dtype=np.float32),
        'stop': np.array(2.0, dtype=np.float32),
    }

def forward(start, stop):
    return {'out': np.logspace(start, stop, 17)}
"""
    lowered = _lowered(source)

    # The current composition creates a float64 linspace and changes output dtype.
    assert "np.logspace" in lowered.source
    assert "composition" not in [item.kind for item in lowered.rewrites]


def test_geomspace_zero_keeps_exception_contract():
    source = """
import numpy as np

def build_inputs():
    return {
        'start': np.array(0.0, dtype=np.float64),
        'stop': np.array(1.0, dtype=np.float64),
    }

def forward(start, stop):
    return {'out': np.geomspace(start, stop, 5)}
"""
    lowered = _lowered(source)

    # NumPy raises ValueError for zero endpoints; divide/power would emit NaN/Inf.
    assert "np.geomspace" in lowered.source
    assert "composition" not in [item.kind for item in lowered.rewrites]

