import numpy as np

from numpilot.comparison import compare_outputs


def test_verifier_rejects_zero_reference_mismatch():
    result = compare_outputs(
        np.zeros((3,), dtype=np.float32),
        np.ones((3,), dtype=np.float32),
    )
    assert not result.consistent
    assert result.max_abs_diff == 1.0
    assert result.max_rel_diff > 1.0
    assert result.failure_index == "(0,)"


def test_verifier_checks_shape_and_dtype():
    shape = compare_outputs(
        np.zeros((2, 1), dtype=np.float32),
        np.zeros((2,), dtype=np.float32),
    )
    dtype = compare_outputs(
        np.zeros((2,), dtype=np.float32),
        np.zeros((2,), dtype=np.float64),
    )
    assert not shape.consistent and "shape" in shape.reason
    assert not dtype.consistent and "dtype" in dtype.reason


def test_verifier_handles_nan_inf_and_nested_outputs():
    reference = {
        "values": [
            np.array([np.nan, np.inf, -np.inf, 0.0], dtype=np.float32),
            np.array([1, 2, 3], dtype=np.int32),
        ],
        "flag": True,
    }
    same = {
        "values": [
            np.array([np.nan, np.inf, -np.inf, 0.0], dtype=np.float32),
            np.array([1, 2, 3], dtype=np.int32),
        ],
        "flag": True,
    }
    wrong_inf = {
        "values": [
            np.array([np.nan, -np.inf, -np.inf, 0.0], dtype=np.float32),
            np.array([1, 2, 3], dtype=np.int32),
        ],
        "flag": True,
    }
    assert compare_outputs(reference, same).consistent
    result = compare_outputs(reference, wrong_inf)
    assert not result.consistent
    assert result.failure_path == "$['values'][0]"


def test_integer_and_boolean_outputs_are_exact():
    result = compare_outputs(
        (np.array([1, 2], dtype=np.int64), np.array([True, False])),
        (np.array([1, 3], dtype=np.int64), np.array([True, False])),
    )
    assert not result.consistent
    assert result.failure_path == "$[0]"


def test_empty_arrays_compare_without_reduction_error():
    result = compare_outputs(
        np.empty((0, 2), dtype=np.float32),
        np.empty((0, 2), dtype=np.float32),
    )
    assert result.consistent
    assert result.n_elements == 0


def test_nested_comparison_aggregates_all_leaves():
    result = compare_outputs(
        [np.array([0.0], dtype=np.float32), np.array([0.0], dtype=np.float32)],
        [np.array([1e-3], dtype=np.float32), np.array([1e6], dtype=np.float32)],
    )
    assert not result.consistent
    assert result.max_abs_diff == 1e6
    assert result.n_elements == 2


def test_dict_order_is_not_semantic_and_scalar_index_is_reported():
    ordered_differently = compare_outputs(
        {"a": np.array([1]), "b": np.array([2])},
        {"b": np.array([2]), "a": np.array([1])},
    )
    scalar_mismatch = compare_outputs(np.int64(1), np.int64(2))
    assert ordered_differently.consistent
    assert not scalar_mismatch.consistent
    assert scalar_mismatch.failure_index == "()"
    assert scalar_mismatch.max_abs_diff == 1.0


def test_zero_reference_within_atol_has_finite_metric():
    result = compare_outputs(
        np.array([0.0], dtype=np.float32),
        np.array([5e-6], dtype=np.float32),
    )
    assert result.consistent
    assert np.isfinite(result.max_rel_diff)
