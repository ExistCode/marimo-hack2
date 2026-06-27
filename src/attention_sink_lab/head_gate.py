"""Causal head-gating experiment for testing whether sink heads are quiet."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.typing import ArrayLike, NDArray

from attention_sink_lab.probe import _encode, _load_model


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class HeadGateSummary:
    """Behavioral changes relative to an untouched baseline batch row."""

    js_divergences: FloatArray
    final_hidden_l2: FloatArray
    top_token_ids: IntArray


@dataclass(frozen=True)
class HeadGateResult:
    """Baseline, sink-head gate, and comparison-head gate from one forward."""

    model_id: str
    layer: int
    sink_head: int
    comparison_head: int
    sink_score: float
    comparison_score: float
    js_divergences: FloatArray
    final_hidden_l2: FloatArray
    projected_contribution_l2: FloatArray
    top_predictions: tuple[str, ...]
    device: str
    accelerator_name: str
    elapsed_seconds: float
    token_count: int

    @property
    def batch_size(self) -> int:
        return 3


def _softmax(logits: FloatArray) -> FloatArray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / np.sum(exponentiated, axis=-1, keepdims=True)


def summarize_head_gate(final_logits: ArrayLike, final_hidden: ArrayLike) -> HeadGateSummary:
    """Summarize a three-row baseline/sink/comparison intervention batch."""

    logits = np.asarray(final_logits, dtype=np.float64)
    hidden = np.asarray(final_hidden, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[0] != 3:
        raise ValueError("final_logits must have three rows")
    if hidden.ndim != 2 or hidden.shape[0] != 3:
        raise ValueError("final_hidden must have three rows")

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
    hidden_l2 = np.linalg.norm(hidden - hidden[0], axis=1)
    hidden_l2[0] = 0.0
    return HeadGateSummary(
        js_divergences=js_divergences,
        final_hidden_l2=hidden_l2,
        top_token_ids=np.argmax(logits, axis=1).astype(np.int64),
    )


def _gate_o_proj_input(values, *, attention_heads: int, sink_head: int, comparison_head: int):
    """Gate different concatenated head slices in batch rows one and two."""

    if values.ndim != 3 or values.shape[0] != 3:
        raise ValueError("head gating expects a three-row batch")
    hidden_size = values.shape[-1]
    if hidden_size % attention_heads:
        raise ValueError("hidden size must be divisible by the attention head count")
    if not 0 <= sink_head < attention_heads:
        raise ValueError("sink_head is out of range")
    if not 0 <= comparison_head < attention_heads:
        raise ValueError("comparison_head is out of range")

    head_dim = hidden_size // attention_heads
    gated = values.clone().reshape(values.shape[0], values.shape[1], attention_heads, head_dim)
    gated[1, :, sink_head, :] = 0
    gated[2, :, comparison_head, :] = 0
    return gated.reshape_as(values)


def _projected_head_contribution_l2(
    projection,
    values,
    *,
    attention_heads: int,
    head: int,
) -> float:
    """Measure one baseline head's final-token contribution after output projection."""

    if values.ndim != 3 or values.shape[0] < 1:
        raise ValueError("projected contribution expects a batch-by-token-by-hidden tensor")
    hidden_size = values.shape[-1]
    if hidden_size % attention_heads:
        raise ValueError("hidden size must be divisible by the attention head count")
    if not 0 <= head < attention_heads:
        raise ValueError("head is out of range")

    head_dim = hidden_size // attention_heads
    start = head * head_dim
    end = start + head_dim
    final_head_values = values[0, -1, start:end].float()
    head_weight = projection.weight[:, start:end].float()
    contribution = final_head_values @ head_weight.T
    return float(contribution.norm().detach().cpu())


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


def run_head_gate(
    model_id: str,
    prompt: str,
    *,
    layer: int,
    sink_head: int,
    comparison_head: int,
    sink_score: float,
    comparison_score: float,
    max_tokens: int = 96,
) -> HeadGateResult:
    """Gate a sink head and comparison head in separate rows of one batch."""

    import torch

    tokenizer, model, device = _load_model(model_id)
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise ValueError("head gating currently supports Llama-style causal language models")
    if not 0 <= layer < len(layers):
        raise ValueError("layer is out of range")
    attention_heads = int(model.config.num_attention_heads)
    projection = layers[layer].self_attn.o_proj

    encoded = _encode(tokenizer, prompt, include_bos=True, max_tokens=max_tokens)
    input_ids = encoded.repeat(3, 1).to(device)
    attention_mask = torch.ones_like(input_ids)
    projected_contributions: dict[str, float] = {}

    def gate_hook(_module, args):
        projected_contributions["sink"] = _projected_head_contribution_l2(
            projection,
            args[0],
            attention_heads=attention_heads,
            head=sink_head,
        )
        projected_contributions["comparison"] = _projected_head_contribution_l2(
            projection,
            args[0],
            attention_heads=attention_heads,
            head=comparison_head,
        )
        gated_values = _gate_o_proj_input(
            args[0],
            attention_heads=attention_heads,
            sink_head=sink_head,
            comparison_head=comparison_head,
        )
        return (gated_values, *args[1:])

    hook = projection.register_forward_pre_hook(gate_hook)
    try:
        _synchronize(torch, device)
        started = perf_counter()
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
        _synchronize(torch, device)
        elapsed = perf_counter() - started
    finally:
        hook.remove()

    final_logits = output.logits[:, -1].detach().float().cpu().numpy().astype(np.float64)
    final_hidden = (
        output.hidden_states[-1][:, -1].detach().float().cpu().numpy().astype(np.float64)
    )
    summary = summarize_head_gate(final_logits, final_hidden)
    return HeadGateResult(
        model_id=model_id,
        layer=layer,
        sink_head=sink_head,
        comparison_head=comparison_head,
        sink_score=float(sink_score),
        comparison_score=float(comparison_score),
        js_divergences=summary.js_divergences,
        final_hidden_l2=summary.final_hidden_l2,
        projected_contribution_l2=np.asarray(
            [0.0, projected_contributions["sink"], projected_contributions["comparison"]],
            dtype=np.float64,
        ),
        top_predictions=tuple(tokenizer.convert_ids_to_tokens(summary.top_token_ids.tolist())),
        device=device,
        accelerator_name=_accelerator_name(torch, device),
        elapsed_seconds=elapsed,
        token_count=input_ids.numel(),
    )


__all__ = ["HeadGateResult", "run_head_gate", "summarize_head_gate"]
