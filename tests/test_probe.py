from types import SimpleNamespace

import numpy as np
import pytest
import torch

import attention_sink_lab.probe as probe_module
from attention_sink_lab.probe import ProbeResult, _repeat_kv_norms, probe_perturbation


def test_repeat_kv_norms_maps_grouped_heads() -> None:
    norms = np.asarray([[1.0, 2.0], [3.0, 4.0]])

    repeated = _repeat_kv_norms(norms, attention_heads=4)

    np.testing.assert_allclose(
        repeated,
        np.asarray([[1.0, 2.0], [1.0, 2.0], [3.0, 4.0], [3.0, 4.0]]),
    )


def test_repeat_kv_norms_rejects_invalid_grouping() -> None:
    with pytest.raises(ValueError, match="divisible"):
        _repeat_kv_norms(np.ones((2, 3)), attention_heads=3)


def test_bos_value_ratio_uses_subsequent_token_median() -> None:
    result = ProbeResult(
        model_id="test",
        tokens=("<bos>", "a", "b"),
        attentions=np.zeros((1, 1, 3, 3)),
        sink_profile=np.zeros((1, 1)),
        hidden_diversity=np.zeros(2),
        value_norms=np.asarray([[[1.0, 4.0, 6.0]]]),
        device="cpu",
        elapsed_seconds=0.0,
    )

    assert result.bos_value_ratio(0, 0) == pytest.approx(0.2)


class PerturbationTokenizer:
    bos_token_id = 1

    sequences = {
        "red cat sat": [10, 20, 30],
        "red dog sat": [10, 21, 30],
    }

    def __call__(self, text: str, **options):
        return {"input_ids": torch.tensor([self.sequences[text]], dtype=torch.long)}

    def convert_ids_to_tokens(self, token_ids: list[int]) -> list[str]:
        return [f"tok_{token_id}" for token_id in token_ids]


class BatchedPerturbationModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, **options):
        input_ids = options["input_ids"].detach().cpu()
        attention_mask = options["attention_mask"].detach().cpu()
        self.calls.append(
            {
                "input_ids": input_ids.clone(),
                "attention_mask": attention_mask.clone(),
                "output_attentions": options["output_attentions"],
                "output_hidden_states": options["output_hidden_states"],
                "use_cache": options["use_cache"],
            }
        )

        batch_size, token_count = input_ids.shape
        first_hidden = input_ids.float().unsqueeze(-1)
        second_hidden = first_hidden * 2.0
        attentions = []
        for original_attention in (0.2, 0.4):
            layer = torch.zeros((batch_size, 1, token_count, token_count), dtype=torch.float32)
            if token_count > 1:
                layer[0, 0, 1:, 0] = original_attention
                layer[1, 0, 1:, 0] = 0.95
            attentions.append(layer)

        return SimpleNamespace(
            attentions=tuple(attentions),
            hidden_states=(first_hidden, second_hidden),
        )


def test_probe_perturbation_batches_original_and_changed_per_bos_condition(
    monkeypatch,
) -> None:
    tokenizer = PerturbationTokenizer()
    model = BatchedPerturbationModel()
    monkeypatch.setattr(
        probe_module,
        "_load_model",
        lambda model_id: (tokenizer, model, "cpu"),
    )

    result = probe_perturbation(
        "example/model",
        "red cat sat",
        "red dog sat",
        max_tokens=8,
    )

    assert len(model.calls) == 2
    np.testing.assert_array_equal(
        model.calls[0]["input_ids"],
        [[1, 10, 20, 30], [1, 10, 21, 30]],
    )
    np.testing.assert_array_equal(model.calls[0]["attention_mask"], np.ones((2, 4)))
    np.testing.assert_array_equal(
        model.calls[1]["input_ids"],
        [[10, 20, 30], [10, 21, 30]],
    )
    np.testing.assert_array_equal(model.calls[1]["attention_mask"], np.ones((2, 3)))

    for call in model.calls:
        assert call["output_attentions"] is True
        assert call["output_hidden_states"] is True
        assert call["use_cache"] is False

    assert result.tokens_with_bos == ("tok_1", "tok_10", "tok_20", "tok_30")
    assert result.tokens_without_bos == ("tok_10", "tok_20", "tok_30")
    assert result.first_changed_with_bos == 2
    assert result.first_changed_without_bos == 1
    np.testing.assert_allclose(
        result.difference_with_bos,
        [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 2.0, 0.0]],
    )
    np.testing.assert_allclose(
        result.difference_without_bos,
        [[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]],
    )
    assert result.sink_score_with_bos == pytest.approx(0.3)
    assert result.sink_score_without_bos == pytest.approx(0.3)
