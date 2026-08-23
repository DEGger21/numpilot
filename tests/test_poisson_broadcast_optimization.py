from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np

from numpilot.agents.analysis import AnalysisAgent
from numpilot.agents.decision import DecisionAgent
from numpilot.agents.migration import MigrationAgent
from tools.poisson_broadcast_performance_evidence import _build_scaled_inputs


ROOT = Path(__file__).parents[1]
BASELINE = ROOT / "examples" / "poisson_ritz_iteration.py"
OPTIMIZED = ROOT / "examples" / "poisson_ritz_iteration_broadcast.py"


def _load(name, path):
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_broadcast_contract_uses_column_coordinates_and_contiguous_arrays():
    inputs = _build_scaled_inputs(8, 8, 3)

    assert len(inputs) == 28
    assert inputs[0].shape == (64, 1)
    assert inputs[3].shape == (32, 1)
    assert inputs[6].shape == (9,)
    assert all(value.dtype == np.float32 for value in inputs)
    assert all(value.flags.c_contiguous for value in inputs)


def test_broadcast_iteration_is_bitwise_equal_to_baseline_for_25_rounds():
    baseline = _load("poisson_baseline", BASELINE)
    optimized = _load("poisson_broadcast", OPTIMIZED)
    baseline_inputs = list(baseline.build_inputs())
    optimized_inputs = list(optimized.build_inputs())

    for _ in range(25):
        baseline_output = baseline.forward(*baseline_inputs)
        optimized_output = optimized.forward(*optimized_inputs)
        for expected, actual in zip(baseline_output, optimized_output):
            np.testing.assert_array_equal(actual, expected)
        baseline_inputs[6:10] = baseline_output[1:5]
        optimized_inputs[6:10] = optimized_output[1:5]


def test_broadcast_iteration_reduces_static_calls_from_117_to_101():
    source = OPTIMIZED.read_text(encoding="utf-8")
    analysis = AnalysisAgent().analyze(source, str(OPTIMIZED))

    assert analysis.migratable
    migration = MigrationAgent().migrate(source, DecisionAgent().plan(analysis))
    assert sum(migration.rewrites.values()) == 101
    assert migration.rewrites["outer"] == 2
    assert migration.rewrites["multiply"] == 40
    assert migration.rewrites["sqrt"] == 1
