import numpy as np
import pytest

from tools.poisson_iteration_evidence import _advance, _normalise_inputs


def test_advance_keeps_constants_and_carries_four_parameter_tensors():
    inputs = [np.array([np.float32(index)]) for index in range(30)]
    constants = tuple(inputs[:6] + inputs[10:])
    calls = []

    def forward(*args):
        calls.append(args)
        next_parameters = tuple(
            np.add(args[index], np.array([np.float32(1.0)]))
            for index in range(6, 10)
        )
        return (np.array([np.float32(0.0)]), *next_parameters)

    output = _advance(forward, inputs, 3)

    assert len(calls) == 3
    for original, retained in zip(constants, inputs[:6] + inputs[10:]):
        assert retained is original
    for index, parameter in enumerate(output[1:], start=6):
        np.testing.assert_array_equal(
            parameter,
            np.array([np.float32(index + 3)]),
        )
        assert inputs[index] is parameter


def test_normalise_inputs_rejects_non_array_or_wrong_contract_size():
    with pytest.raises(ValueError, match="30 inputs"):
        _normalise_inputs(tuple(np.ones(1, dtype=np.float32) for _ in range(29)))
    with pytest.raises(TypeError, match="NumPy ndarray"):
        _normalise_inputs(tuple(np.ones(1, dtype=np.float32) for _ in range(29)) + (1,))
