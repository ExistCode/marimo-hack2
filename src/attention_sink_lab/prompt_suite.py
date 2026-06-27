"""Run and aggregate attention probes across a suite of prompts."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.typing import ArrayLike, NDArray

from attention_sink_lab.probe import _load_model


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SinkProfileAggregation:
    """Prompt-level measurements and their mean layer-by-head sink profile."""

    sink_fractions: tuple[float, ...]
    token_lengths: tuple[int, ...]
    mean_sink_profile: FloatArray


@dataclass(frozen=True)
class PromptSuiteResult:
    """Aggregated result from probing an ordered suite of prompts."""

    model_id: str
    prompts: tuple[str, ...]
    sink_fractions: tuple[float, ...]
    token_lengths: tuple[int, ...]
    mean_sink_profile: FloatArray
    device: str
    elapsed_seconds: float
    batch_size: int
    token_count: int
    max_tokens: int = 96
    threshold: float = 0.3

    @property
    def prompt_count(self) -> int:
        return len(self.prompts)

    @property
    def layers(self) -> int:
        return int(self.mean_sink_profile.shape[0])

    @property
    def heads(self) -> int:
        return int(self.mean_sink_profile.shape[1])


def aggregate_sink_profiles(
    sink_profiles: Iterable[ArrayLike],
    token_lengths: Iterable[int],
    *,
    threshold: float = 0.3,
) -> SinkProfileAggregation:
    """Aggregate compatible sink profiles without loading or calling a model."""

    _validate_threshold(threshold)

    profiles = tuple(np.asarray(profile, dtype=np.float64) for profile in sink_profiles)
    lengths = tuple(int(length) for length in token_lengths)

    if not profiles:
        raise ValueError("at least one sink profile is required")
    if len(profiles) != len(lengths):
        raise ValueError("sink_profiles and token_lengths must have the same length")
    if any(length < 0 for length in lengths):
        raise ValueError("token lengths must be non-negative")

    expected_shape = profiles[0].shape
    if len(expected_shape) != 2:
        raise ValueError("sink profiles must be two-dimensional layer-by-head arrays")
    if any(profile.ndim != 2 or profile.shape != expected_shape for profile in profiles[1:]):
        raise ValueError("all sink profiles must have the same layer-by-head shape")

    fractions = tuple(float(np.mean(profile >= threshold)) for profile in profiles)
    mean_profile = np.mean(np.stack(profiles), axis=0)

    return SinkProfileAggregation(
        sink_fractions=fractions,
        token_lengths=lengths,
        mean_sink_profile=mean_profile,
    )


def _validate_threshold(threshold: float) -> None:
    """Validate a sink threshold without changing its numeric semantics."""

    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")


def sink_profiles_from_batched_attentions(
    attentions: ArrayLike,
    attention_mask: ArrayLike,
) -> FloatArray:
    """Compute per-prompt sink profiles from padded causal attention maps.

    The returned array is batch-by-layer-by-head. Padding queries and query
    position zero are excluded from each mean; position zero remains the key
    whose received attention is measured.
    """

    values = np.asarray(attentions, dtype=np.float64)
    mask = np.asarray(attention_mask)

    if values.ndim != 5:
        raise ValueError(
            "attentions must be a layer-by-batch-by-head-by-query-by-key array"
        )
    if mask.ndim != 2:
        raise ValueError("attention_mask must be a batch-by-token array")

    _, batch_size, _, query_count, key_count = values.shape
    if mask.shape != (batch_size, query_count):
        raise ValueError("attention_mask must match the attention batch and query dimensions")
    if key_count < 1:
        raise ValueError("attentions must include at least one key position")

    valid_queries = mask.astype(bool, copy=True)
    if query_count and not np.all(valid_queries[:, 0]):
        raise ValueError("position zero must be valid for every prompt")
    if query_count:
        valid_queries[:, 0] = False

    attention_to_first = values[..., 0]
    query_weights = valid_queries[None, :, None, :]
    attention_sum = np.where(query_weights, attention_to_first, 0.0).sum(axis=-1)
    query_totals = valid_queries.sum(axis=-1)[None, :, None]
    layer_batch_head = np.divide(
        attention_sum,
        query_totals,
        out=np.zeros_like(attention_sum),
        where=query_totals != 0,
    )
    return np.transpose(layer_batch_head, (1, 0, 2))


def _padding_token_id(tokenizer) -> int:
    for name in ("pad_token_id", "eos_token_id", "bos_token_id"):
        token_id = getattr(tokenizer, name, None)
        if token_id is not None:
            return int(token_id)
    return 0


def _tokenize_prompt_batch(tokenizer, prompts: tuple[str, ...], *, max_tokens: int):
    """Tokenize once, add BOS uniformly, and right-pad for causal attention."""

    import torch

    encoded = tokenizer(
        list(prompts),
        add_special_tokens=False,
        truncation=True,
        max_length=max_tokens,
    )["input_ids"]
    bos_id = getattr(tokenizer, "bos_token_id", None)
    sequences: list[list[int]] = []
    for token_ids in encoded:
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        sequence = [int(token_id) for token_id in token_ids]
        if bos_id is not None:
            sequence.insert(0, int(bos_id))
        sequence = sequence[:max_tokens]
        if not sequence:
            raise ValueError("each prompt must produce at least one token")
        sequences.append(sequence)

    token_lengths = tuple(len(sequence) for sequence in sequences)
    padded_length = max(token_lengths)
    input_ids = torch.full(
        (len(sequences), padded_length),
        _padding_token_id(tokenizer),
        dtype=torch.long,
    )
    attention_mask = torch.zeros_like(input_ids)
    for index, sequence in enumerate(sequences):
        length = len(sequence)
        input_ids[index, :length] = torch.tensor(sequence, dtype=torch.long)
        attention_mask[index, :length] = 1

    return input_ids, attention_mask, token_lengths


def _synchronize_device(torch_module, device: str) -> None:
    """Wait for asynchronous accelerator work when the backend supports it."""

    device_name = str(device)
    backend = None
    if device_name.startswith("cuda"):
        backend = getattr(torch_module, "cuda", None)
    elif device_name.startswith("mps"):
        backend = getattr(torch_module, "mps", None)
    synchronize = getattr(backend, "synchronize", None)
    if callable(synchronize):
        synchronize()


def run_prompt_suite(
    model_id: str,
    prompts: Iterable[str],
    *,
    max_tokens: int = 96,
    threshold: float = 0.3,
) -> PromptSuiteResult:
    """Probe a prompt suite in one padded tokenizer batch and model forward."""

    prompt_values = tuple(prompts)
    if not prompt_values:
        raise ValueError("at least one prompt is required")
    if max_tokens < 1:
        raise ValueError("max_tokens must be at least one")
    _validate_threshold(threshold)

    import torch

    tokenizer, model, device = _load_model(model_id)
    device_name = str(device)
    input_ids, attention_mask, token_lengths = _tokenize_prompt_batch(
        tokenizer,
        prompt_values,
        max_tokens=max_tokens,
    )
    device_input_ids = input_ids.to(device)
    device_attention_mask = attention_mask.to(device)
    _synchronize_device(torch, device_name)
    started = perf_counter()
    with torch.inference_mode():
        output = model(
            input_ids=device_input_ids,
            attention_mask=device_attention_mask,
            output_attentions=True,
            use_cache=False,
        )
    if not output.attentions:
        raise RuntimeError("The selected model did not return attentions.")

    _synchronize_device(torch, device_name)
    elapsed_seconds = perf_counter() - started
    attention_values = np.stack(
        [layer.detach().float().cpu().numpy() for layer in output.attentions]
    )
    sink_profiles = sink_profiles_from_batched_attentions(
        attention_values,
        attention_mask.numpy(),
    )

    aggregation = aggregate_sink_profiles(
        sink_profiles,
        token_lengths,
        threshold=threshold,
    )

    return PromptSuiteResult(
        model_id=model_id,
        prompts=prompt_values,
        sink_fractions=aggregation.sink_fractions,
        token_lengths=aggregation.token_lengths,
        mean_sink_profile=aggregation.mean_sink_profile,
        device=device_name,
        elapsed_seconds=elapsed_seconds,
        batch_size=len(prompt_values),
        token_count=sum(token_lengths),
        max_tokens=max_tokens,
        threshold=threshold,
    )


probe_prompt_suite = run_prompt_suite


__all__ = [
    "PromptSuiteResult",
    "SinkProfileAggregation",
    "aggregate_sink_profiles",
    "probe_prompt_suite",
    "run_prompt_suite",
    "sink_profiles_from_batched_attentions",
]
