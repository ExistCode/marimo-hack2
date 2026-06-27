import numpy as np
import pytest

from attention_sink_lab.simulation import SimulationConfig, causal_attention, simulate


def test_causal_attention_is_normalized_and_causal() -> None:
    attention = causal_attention(tokens=9, sink_strength=0.7)

    np.testing.assert_allclose(attention.sum(axis=1), np.ones(9))
    assert np.allclose(np.triu(attention, k=1), 0.0)


def test_sink_limits_perturbation_spread() -> None:
    config = SimulationConfig(depth=20, sink_strength=0.72, sink_value_scale=0.03)

    baseline = simulate(config, with_sink=False)
    sink = simulate(config, with_sink=True)

    assert sink.final_spill < baseline.final_spill
    assert sink.diversity[-1] > baseline.diversity[-1]


def test_simulation_is_deterministic() -> None:
    config = SimulationConfig(seed=4)

    first = simulate(config, with_sink=True)
    second = simulate(config, with_sink=True)

    np.testing.assert_allclose(first.perturbation, second.perturbation)
    np.testing.assert_allclose(first.diversity, second.diversity)


def test_zero_sink_matches_baseline() -> None:
    config = SimulationConfig(sink_strength=0.0, sink_value_scale=0.0)

    baseline = simulate(config, with_sink=False)
    sink = simulate(config, with_sink=True)

    np.testing.assert_allclose(sink.perturbation, baseline.perturbation)
    np.testing.assert_allclose(sink.diversity, baseline.diversity)


def test_invalid_config_is_rejected() -> None:
    with pytest.raises(ValueError, match="tokens"):
        SimulationConfig(tokens=2)
