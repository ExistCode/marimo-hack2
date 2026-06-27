import numpy as np
import pytest
import torch

from attention_sink_lab.head_gate import (
    _gate_o_proj_input,
    _projected_head_contribution_l2,
    summarize_head_gate,
)


def test_gate_o_proj_input_changes_only_requested_batch_heads() -> None:
    values = torch.ones((3, 2, 12))

    gated = _gate_o_proj_input(
        values,
        attention_heads=3,
        sink_head=1,
        comparison_head=2,
    ).reshape(3, 2, 3, 4)

    assert torch.all(gated[0] == 1)
    assert torch.all(gated[1, :, 1] == 0)
    assert torch.all(gated[1, :, (0, 2)] == 1)
    assert torch.all(gated[2, :, 2] == 0)
    assert torch.all(gated[2, :, (0, 1)] == 1)


def test_gate_o_proj_input_rejects_bad_head_index() -> None:
    with pytest.raises(ValueError, match="sink_head"):
        _gate_o_proj_input(
            torch.ones((3, 1, 8)),
            attention_heads=2,
            sink_head=2,
            comparison_head=0,
        )


def test_projected_head_contribution_uses_final_token_and_head_weight_slice() -> None:
    projection = torch.nn.Linear(4, 3, bias=False)
    with torch.no_grad():
        projection.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 10.0, 0.0],
                    [0.0, 2.0, 0.0, 20.0],
                    [1.0, 1.0, 1.0, 1.0],
                ]
            )
        )
    values = torch.zeros((3, 2, 4))
    values[0, 0] = torch.tensor([99.0, 99.0, 99.0, 99.0])
    values[0, 1] = torch.tensor([3.0, 4.0, 5.0, 6.0])

    first_head = _projected_head_contribution_l2(
        projection,
        values,
        attention_heads=2,
        head=0,
    )
    second_head = _projected_head_contribution_l2(
        projection,
        values,
        attention_heads=2,
        head=1,
    )

    np.testing.assert_allclose(first_head, np.linalg.norm([3.0, 8.0, 7.0]))
    np.testing.assert_allclose(second_head, np.linalg.norm([50.0, 120.0, 11.0]))


def test_summarize_head_gate_measures_behavioral_drift() -> None:
    logits = np.array([[5.0, 1.0, 0.0], [5.0, 1.0, 0.0], [0.0, 1.0, 5.0]])
    hidden = np.array([[1.0, 2.0], [1.0, 2.0], [4.0, 6.0]])

    summary = summarize_head_gate(logits, hidden)

    np.testing.assert_allclose(summary.js_divergences[:2], [0.0, 0.0], atol=1e-12)
    assert summary.js_divergences[2] > 0.5
    np.testing.assert_allclose(summary.final_hidden_l2, [0.0, 0.0, 5.0])
    np.testing.assert_array_equal(summary.top_token_ids, [0, 0, 2])
