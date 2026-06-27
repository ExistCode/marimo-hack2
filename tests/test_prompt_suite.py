from types import SimpleNamespace

import numpy as np
import pytest
import torch

from attention_sink_lab import prompt_suite
from attention_sink_lab.prompt_suite import (
    aggregate_sink_profiles,
    sink_profiles_from_batched_attentions,
)


def test_aggregate_sink_profiles_calculates_prompt_metrics_and_mean() -> None:
    first = np.array([[0.1, 0.3], [0.4, 0.2]])
    second = np.array([[0.5, 0.1], [0.8, 0.6]])

    result = aggregate_sink_profiles([first, second], [5, 8], threshold=0.3)

    assert result.sink_fractions == (0.5, 0.75)
    assert result.token_lengths == (5, 8)
    np.testing.assert_allclose(result.mean_sink_profile, [[0.3, 0.2], [0.6, 0.4]])


def test_aggregate_sink_profiles_rejects_incompatible_shapes() -> None:
    with pytest.raises(ValueError, match="same layer-by-head shape"):
        aggregate_sink_profiles([np.zeros((2, 3)), np.zeros((3, 2))], [4, 4])


def test_aggregate_sink_profiles_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one sink profile"):
        aggregate_sink_profiles([], [])


@pytest.mark.parametrize("threshold", [float("nan"), float("inf")])
def test_aggregate_sink_profiles_rejects_non_finite_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="threshold must be finite"):
        aggregate_sink_profiles([np.zeros((2, 3))], [4], threshold=threshold)


def test_sink_profiles_ignore_position_zero_and_padded_queries() -> None:
    attentions = np.zeros((2, 2, 1, 4, 4), dtype=np.float64)
    attentions[:, 0, 0, :, 0] = [[99.0, 0.2, 0.4, 0.6], [99.0, 0.1, 0.3, 0.5]]
    attentions[:, 1, 0, :, 0] = [[99.0, 0.8, 77.0, 88.0], [99.0, 0.6, 77.0, 88.0]]
    attention_mask = np.array([[1, 1, 1, 1], [1, 1, 0, 0]])

    profiles = sink_profiles_from_batched_attentions(attentions, attention_mask)

    np.testing.assert_allclose(profiles[:, :, 0], [[0.4, 0.3], [0.8, 0.6]])


def test_sink_profiles_use_zero_when_bos_is_the_only_valid_query() -> None:
    attentions = np.ones((1, 1, 2, 1, 1), dtype=np.float64)

    profiles = sink_profiles_from_batched_attentions(attentions, [[1]])

    np.testing.assert_array_equal(profiles, np.zeros((1, 1, 2)))


class FakeTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = None

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, object]]] = []

    def __call__(self, prompts: list[str], **options):
        self.calls.append((prompts, options))
        return {"input_ids": [[10, 11, 12], [20]]}


class FakeModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, **options):
        self.calls.append(options)
        first_layer = torch.zeros((2, 2, 4, 4), dtype=torch.float32)
        second_layer = torch.zeros((2, 2, 4, 4), dtype=torch.float32)
        first_layer[0, :, 1:, 0] = torch.tensor([[0.2, 0.4, 0.6], [0.1, 0.3, 0.5]])
        first_layer[1, :, 1:, 0] = torch.tensor([[0.8, 9.0, 9.0], [0.6, 9.0, 9.0]])
        second_layer[0, :, 1:, 0] = torch.tensor([[0.3, 0.5, 0.7], [0.2, 0.4, 0.6]])
        second_layer[1, :, 1:, 0] = torch.tensor([[0.4, 9.0, 9.0], [0.2, 9.0, 9.0]])
        return SimpleNamespace(attentions=(first_layer, second_layer))


def test_run_prompt_suite_uses_one_batch_and_one_forward(monkeypatch) -> None:
    tokenizer = FakeTokenizer()
    model = FakeModel()
    synchronization_calls: list[str] = []
    aggregate_calls: list[dict[str, object]] = []
    original_aggregate_sink_profiles = prompt_suite.aggregate_sink_profiles

    monkeypatch.setattr(
        prompt_suite,
        "_load_model",
        lambda model_id: (tokenizer, model, "cpu"),
    )
    monkeypatch.setattr(
        prompt_suite,
        "_synchronize_device",
        lambda _torch, device: synchronization_calls.append(device),
    )

    def aggregate_spy(sink_profiles, token_lengths, *, threshold=0.3):
        aggregate_calls.append(
            {
                "sink_profiles_shape": np.asarray(sink_profiles).shape,
                "token_lengths": tuple(token_lengths),
                "threshold": threshold,
            }
        )
        return original_aggregate_sink_profiles(
            sink_profiles,
            token_lengths,
            threshold=threshold,
        )

    monkeypatch.setattr(prompt_suite, "aggregate_sink_profiles", aggregate_spy)

    result = prompt_suite.run_prompt_suite(
        "example/model",
        (prompt for prompt in ["short", "longer"]),
        max_tokens=32,
        threshold=0.4,
    )

    assert tokenizer.calls == [
        (
            ["short", "longer"],
            {"add_special_tokens": False, "truncation": True, "max_length": 32},
        )
    ]
    assert len(model.calls) == 1
    np.testing.assert_array_equal(
        model.calls[0]["input_ids"],
        [[1, 10, 11, 12], [1, 20, 2, 2]],
    )
    np.testing.assert_array_equal(
        model.calls[0]["attention_mask"],
        [[1, 1, 1, 1], [1, 1, 0, 0]],
    )
    assert model.calls[0]["output_attentions"] is True
    assert model.calls[0]["use_cache"] is False
    assert synchronization_calls == ["cpu", "cpu"]
    assert result.model_id == "example/model"
    assert result.prompts == ("short", "longer")
    assert result.sink_fractions == (0.75, 0.75)
    assert result.token_lengths == (4, 2)
    assert result.prompt_count == 2
    assert result.batch_size == 2
    assert result.token_count == 6
    assert result.max_tokens == 32
    assert result.threshold == 0.4
    assert result.device == "cpu"
    assert result.elapsed_seconds >= 0
    assert (result.layers, result.heads) == (2, 2)
    np.testing.assert_allclose(result.mean_sink_profile, [[0.6, 0.45], [0.45, 0.3]])
    assert aggregate_calls == [
        {
            "sink_profiles_shape": (2, 2, 2),
            "token_lengths": (4, 2),
            "threshold": 0.4,
        }
    ]


def test_run_prompt_suite_rejects_empty_input_before_probing(monkeypatch) -> None:
    def unexpected_load(*args, **kwargs):
        pytest.fail("the model should not be loaded")

    monkeypatch.setattr(prompt_suite, "_load_model", unexpected_load)

    with pytest.raises(ValueError, match="at least one prompt"):
        prompt_suite.run_prompt_suite("example/model", [])


@pytest.mark.parametrize("threshold", [float("nan"), float("inf")])
def test_run_prompt_suite_rejects_non_finite_threshold_before_probing(
    monkeypatch,
    threshold: float,
) -> None:
    def unexpected_load(*args, **kwargs):
        pytest.fail("the model should not be loaded")

    monkeypatch.setattr(prompt_suite, "_load_model", unexpected_load)

    with pytest.raises(ValueError, match="threshold must be finite"):
        prompt_suite.run_prompt_suite("example/model", ["prompt"], threshold=threshold)


@pytest.mark.parametrize("device, backend_name", [("cuda:0", "cuda"), ("mps", "mps")])
def test_synchronize_device_uses_supported_accelerator_backend(device, backend_name) -> None:
    calls: list[str] = []
    backend = SimpleNamespace(synchronize=lambda: calls.append(backend_name))
    torch_module = SimpleNamespace(cuda=None, mps=None)
    setattr(torch_module, backend_name, backend)

    prompt_suite._synchronize_device(torch_module, device)

    assert calls == [backend_name]
