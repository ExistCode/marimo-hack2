import numpy as np
import pytest

from attention_sink_lab.sink_swap import SinkSwapResult, summarize_sink_swaps


def test_summarize_sink_swaps_compares_every_variant_to_first() -> None:
    profiles = np.array(
        [
            [[0.4, 0.2], [0.8, 0.1]],
            [[0.5, 0.7], [0.1, 0.2]],
            [[0.05, 0.1], [0.2, 0.1]],
        ]
    )
    logits = np.array([[4.0, 1.0, 0.0], [4.0, 1.0, 0.0], [0.0, 1.0, 4.0]])

    summary = summarize_sink_swaps(profiles, logits, threshold=0.3)

    np.testing.assert_allclose(summary.sink_fractions, [0.5, 0.5, 0.0])
    np.testing.assert_allclose(summary.mean_sink_scores, [0.375, 0.375, 0.1125])
    np.testing.assert_allclose(summary.js_divergences[:2], [0.0, 0.0], atol=1e-12)
    assert summary.js_divergences[2] > 0.4
    np.testing.assert_array_equal(summary.top_token_ids, [0, 0, 2])


def test_summarize_sink_swaps_rejects_misaligned_logits() -> None:
    with pytest.raises(ValueError, match="one row per sink variant"):
        summarize_sink_swaps(np.zeros((3, 2, 2)), np.zeros((2, 10)))


def test_sink_swap_result_recomputes_threshold_reactively() -> None:
    result = SinkSwapResult(
        model_id="example/model",
        candidate_names=("Native BOS", "Period"),
        candidate_tokens=("<s>", "."),
        sink_profiles=np.array([[[0.8, 0.4]], [[0.2, 0.4]]]),
        mean_sink_scores=np.array([0.6, 0.3]),
        js_divergences=np.array([0.0, 0.1]),
        top_predictions=("A", "B"),
        device="cuda",
        accelerator_name="Test GPU",
        elapsed_seconds=0.02,
        token_count=40,
    )

    np.testing.assert_allclose(result.sink_fractions(0.3), [1.0, 0.5])
    np.testing.assert_allclose(result.sink_fractions(0.7), [0.5, 0.0])
    assert (result.batch_size, result.layers, result.heads) == (2, 1, 2)
