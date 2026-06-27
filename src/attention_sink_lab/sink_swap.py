"""Batched first-token replacement experiment for attention sinks."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.typing import ArrayLike, NDArray

from attention_sink_lab.probe import _load_model


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class SinkSwapSummary:
    """Pure summary of batched attention profiles and final-token logits."""

    sink_fractions: FloatArray
    mean_sink_scores: FloatArray
    js_divergences: FloatArray
    top_token_ids: IntArray


@dataclass(frozen=True)
class SinkSwapResult:
    """Measurements from replacing the first token across one GPU batch."""

    model_id: str
    candidate_names: tuple[str, ...]
    candidate_tokens: tuple[str, ...]
    sink_profiles: FloatArray
    mean_sink_scores: FloatArray
    js_divergences: FloatArray
    top_predictions: tuple[str, ...]
    device: str
    accelerator_name: str
    elapsed_seconds: float
    token_count: int

    @property
    def batch_size(self) -> int:
        return len(self.candidate_names)

    @property
    def layers(self) -> int:
        return int(self.sink_profiles.shape[1])

    @property
    def heads(self) -> int:
        return int(self.sink_profiles.shape[2])

    def sink_fractions(self, threshold: float = 0.3) -> FloatArray:
        return np.mean(self.sink_profiles >= threshold, axis=(1, 2))


def _softmax(logits: FloatArray) -> FloatArray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / np.sum(exponentiated, axis=-1, keepdims=True)


def summarize_sink_swaps(
    sink_profiles: ArrayLike,
    final_logits: ArrayLike,
    *,
    threshold: float = 0.3,
) -> SinkSwapSummary:
    """Summarize swap outputs without requiring PyTorch or a model download."""

    profiles = np.asarray(sink_profiles, dtype=np.float64)
    logits = np.asarray(final_logits, dtype=np.float64)
    if profiles.ndim != 3:
        raise ValueError("sink_profiles must have shape variants × layers × heads")
    if logits.ndim != 2 or logits.shape[0] != profiles.shape[0]:
        raise ValueError("final_logits must have one row per sink variant")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")

    probabilities = _softmax(logits)
    baseline = probabilities[0]
    midpoint = 0.5 * (probabilities + baseline[None, :])
    eps = np.finfo(np.float64).tiny
    candidate_kl = np.sum(
        probabilities * (np.log(np.maximum(probabilities, eps)) - np.log(midpoint)), axis=1
    )
    baseline_kl = np.sum(
        baseline[None, :] * (np.log(np.maximum(baseline[None, :], eps)) - np.log(midpoint)),
        axis=1,
    )
    js_divergences = 0.5 * (candidate_kl + baseline_kl)
    js_divergences[0] = 0.0

    return SinkSwapSummary(
        sink_fractions=np.mean(profiles >= threshold, axis=(1, 2)),
        mean_sink_scores=np.mean(profiles, axis=(1, 2)),
        js_divergences=js_divergences,
        top_token_ids=np.argmax(logits, axis=1).astype(np.int64),
    )


def _synchronize(torch_module: object, device: str) -> None:
    if device == "cuda":
        torch_module.cuda.synchronize()
    elif device == "mps" and hasattr(torch_module, "mps"):
        torch_module.mps.synchronize()


def _accelerator_name(torch_module: object, device: str) -> str:
    if device == "cuda":
        return str(torch_module.cuda.get_device_name(0))
    if device == "mps":
        return "Apple Metal Performance Shaders"
    return "CPU fallback"


def _candidate_id(tokenizer, name: str, token_text: str) -> int:
    if name == "Native BOS":
        if tokenizer.bos_token_id is None:
            raise ValueError("the selected tokenizer does not define a BOS token")
        return int(tokenizer.bos_token_id)
    ids = tokenizer(token_text, add_special_tokens=False)["input_ids"]
    if len(ids) != 1:
        raise ValueError(f"candidate {name!r} must tokenize to exactly one token")
    return int(ids[0])


def run_sink_swap(
    model_id: str,
    prompt: str,
    candidates: Iterable[tuple[str, str]],
    *,
    max_tokens: int = 96,
    threshold: float = 0.3,
) -> SinkSwapResult:
    """Replace token zero across a single accelerator batch and compare behavior."""

    import torch

    candidate_values = tuple(candidates)
    if len(candidate_values) < 2:
        raise ValueError("at least two sink candidates are required")
    if candidate_values[0][0] != "Native BOS":
        raise ValueError("the first candidate must be named 'Native BOS'")

    tokenizer, model, device = _load_model(model_id)
    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=False,
        truncation=True,
        max_length=max_tokens - 1,
    )["input_ids"]
    if not prompt_ids:
        raise ValueError("prompt must produce at least one token")

    candidate_ids = [
        _candidate_id(tokenizer, name, token_text) for name, token_text in candidate_values
    ]
    rows = [[candidate_id, *prompt_ids] for candidate_id in candidate_ids]
    input_ids = torch.tensor(rows, dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)

    _synchronize(torch, device)
    started = perf_counter()
    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            use_cache=False,
        )
    _synchronize(torch, device)
    elapsed = perf_counter() - started
    if not output.attentions:
        raise RuntimeError("the selected model did not return attention tensors")

    profiles_tensor = torch.stack(
        [layer[:, :, 1:, 0].mean(dim=2) for layer in output.attentions], dim=1
    )
    sink_profiles = profiles_tensor.detach().float().cpu().numpy().astype(np.float64)
    final_logits = output.logits[:, -1].detach().float().cpu().numpy().astype(np.float64)
    summary = summarize_sink_swaps(sink_profiles, final_logits, threshold=threshold)

    return SinkSwapResult(
        model_id=model_id,
        candidate_names=tuple(name for name, _ in candidate_values),
        candidate_tokens=tuple(tokenizer.convert_ids_to_tokens(candidate_ids)),
        sink_profiles=sink_profiles,
        mean_sink_scores=summary.mean_sink_scores,
        js_divergences=summary.js_divergences,
        top_predictions=tuple(tokenizer.convert_ids_to_tokens(summary.top_token_ids.tolist())),
        device=device,
        accelerator_name=_accelerator_name(torch, device),
        elapsed_seconds=elapsed,
        token_count=input_ids.numel(),
    )


__all__ = ["SinkSwapResult", "run_sink_swap", "summarize_sink_swaps"]
