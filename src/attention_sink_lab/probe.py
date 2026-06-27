"""Optional real-model attention probe used by the marimo notebook."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from time import perf_counter

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ProbeResult:
    """Attention measurements extracted from a causal language model."""

    model_id: str
    tokens: tuple[str, ...]
    attentions: FloatArray
    sink_profile: FloatArray
    hidden_diversity: FloatArray
    value_norms: FloatArray | None
    device: str
    elapsed_seconds: float

    @property
    def layers(self) -> int:
        return int(self.attentions.shape[0])

    @property
    def heads(self) -> int:
        return int(self.attentions.shape[1])

    def sink_fraction(self, threshold: float = 0.3) -> float:
        return float(np.mean(self.sink_profile >= threshold))

    def bos_value_ratio(self, layer: int, head: int) -> float | None:
        """BOS value norm divided by the median norm of subsequent tokens."""

        if self.value_norms is None or self.value_norms.shape[-1] < 2:
            return None
        values = self.value_norms[layer, head]
        return float(values[0] / max(float(np.median(values[1:])), 1e-9))


@dataclass(frozen=True)
class PerturbationProbeResult:
    """Hidden-state differences for a lexical perturbation with/without BOS."""

    model_id: str
    tokens_with_bos: tuple[str, ...]
    tokens_without_bos: tuple[str, ...]
    difference_with_bos: FloatArray
    difference_without_bos: FloatArray
    first_changed_with_bos: int
    first_changed_without_bos: int
    sink_score_with_bos: float
    sink_score_without_bos: float
    device: str
    elapsed_seconds: float

    @property
    def downstream_with_bos(self) -> float:
        values = self.difference_with_bos[-1, self.first_changed_with_bos + 1 :]
        return float(values.mean()) if len(values) else 0.0

    @property
    def downstream_without_bos(self) -> float:
        values = self.difference_without_bos[-1, self.first_changed_without_bos + 1 :]
        return float(values.mean()) if len(values) else 0.0


def _device_name(torch_module: object) -> str:
    if torch_module.cuda.is_available():
        return "cuda"
    if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


@lru_cache(maxsize=2)
def _load_model(model_id: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = _device_name(torch)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model_options: dict[str, object] = {"attn_implementation": "eager"}
    if device == "cuda":
        model_options["dtype"] = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_options)
    model.eval()
    model.to(device)
    return tokenizer, model, device


def _pairwise_diversity(hidden: FloatArray) -> float:
    norms = np.linalg.norm(hidden, axis=1, keepdims=True)
    normalized = hidden / np.maximum(norms, 1e-9)
    similarity = normalized @ normalized.T
    values = similarity[np.triu_indices(len(hidden), k=1)]
    return float(np.mean(1.0 - values)) if len(values) else 0.0


def _encode(tokenizer, text: str, *, include_bos: bool, max_tokens: int):
    import torch

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_tokens,
        add_special_tokens=False,
    )["input_ids"]
    bos_id = tokenizer.bos_token_id
    if include_bos and bos_id is not None:
        bos = torch.tensor([[bos_id]], dtype=encoded.dtype)
        encoded = torch.cat([bos, encoded], dim=1)[:, :max_tokens]
    return encoded


def _run_model(model, input_ids, device: str):
    import torch

    input_ids = input_ids.to(device)
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=True,
            use_cache=False,
        )
    if not output.attentions or not output.hidden_states:
        raise RuntimeError("The selected model did not return attentions and hidden states.")
    return input_ids, output


def _repeat_kv_norms(norms: FloatArray, attention_heads: int) -> FloatArray:
    """Map grouped-query KV-head norms onto their corresponding query heads."""

    kv_heads = norms.shape[0]
    if attention_heads % kv_heads:
        raise ValueError("attention head count must be divisible by KV head count")
    return np.repeat(norms, attention_heads // kv_heads, axis=0)


def _extract_value_norms(model, hidden_states) -> FloatArray | None:
    """Extract per-token V-vector norms for Llama- and GPT-2-style models."""

    import torch

    collected: list[FloatArray] = []
    with torch.inference_mode():
        llama_layers = getattr(getattr(model, "model", None), "layers", None)
        if llama_layers is not None:
            attention_heads = int(model.config.num_attention_heads)
            kv_heads = int(model.config.num_key_value_heads)
            for index, layer in enumerate(llama_layers):
                normalized = layer.input_layernorm(hidden_states[index][0])
                values = layer.self_attn.v_proj(normalized)
                head_dim = values.shape[-1] // kv_heads
                norms = (
                    values.reshape(values.shape[0], kv_heads, head_dim)
                    .norm(dim=-1)
                    .transpose(0, 1)
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                )
                collected.append(_repeat_kv_norms(norms, attention_heads))
            return np.asarray(collected, dtype=np.float64)

        gpt_layers = getattr(getattr(model, "transformer", None), "h", None)
        if gpt_layers is not None:
            attention_heads = int(model.config.num_attention_heads)
            hidden_size = int(model.config.hidden_size)
            head_dim = hidden_size // attention_heads
            for index, layer in enumerate(gpt_layers):
                normalized = layer.ln_1(hidden_states[index][0])
                qkv = layer.attn.c_attn(normalized)
                values = qkv[..., 2 * hidden_size :]
                norms = (
                    values.reshape(values.shape[0], attention_heads, head_dim)
                    .norm(dim=-1)
                    .transpose(0, 1)
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                )
                collected.append(norms)
            return np.asarray(collected, dtype=np.float64)

    return None


def probe_attention(model_id: str, prompt: str, *, max_tokens: int = 96) -> ProbeResult:
    """Extract causal attention maps and first-token sink scores from a model."""

    started = perf_counter()
    tokenizer, model, device = _load_model(model_id)
    input_ids = _encode(tokenizer, prompt, include_bos=True, max_tokens=max_tokens)
    input_ids, output = _run_model(model, input_ids, device)

    attention = np.stack(
        [layer[0].detach().float().cpu().numpy() for layer in output.attentions]
    ).astype(np.float64)
    if attention.shape[-1] > 1:
        sink_profile = attention[:, :, 1:, 0].mean(axis=2)
    else:
        sink_profile = attention[:, :, :, 0].mean(axis=2)

    hidden_diversity = np.asarray(
        [
            _pairwise_diversity(layer[0].detach().float().cpu().numpy())
            for layer in output.hidden_states
        ],
        dtype=np.float64,
    )
    value_norms = _extract_value_norms(model, output.hidden_states)
    token_ids = input_ids[0].detach().cpu().tolist()
    token_labels = tuple(tokenizer.convert_ids_to_tokens(token_ids))

    return ProbeResult(
        model_id=model_id,
        tokens=token_labels,
        attentions=attention,
        sink_profile=sink_profile,
        hidden_diversity=hidden_diversity,
        value_norms=value_norms,
        device=device,
        elapsed_seconds=perf_counter() - started,
    )


def probe_perturbation(
    model_id: str,
    original: str,
    changed: str,
    *,
    max_tokens: int = 96,
) -> PerturbationProbeResult:
    """Compare a lexical perturbation with and without the BOS sink candidate."""

    started = perf_counter()
    tokenizer, model, device = _load_model(model_id)
    import torch

    condition_results: dict[bool, tuple[FloatArray, float, tuple[str, ...], int]] = {}

    for include_bos in (True, False):
        original_ids = _encode(
            tokenizer, original, include_bos=include_bos, max_tokens=max_tokens
        )
        changed_ids = _encode(
            tokenizer, changed, include_bos=include_bos, max_tokens=max_tokens
        )
        if original_ids.shape != changed_ids.shape:
            raise ValueError(
                "Original and changed prompts must tokenize to the same number of tokens."
            )

        differing = np.flatnonzero(original_ids[0].numpy() != changed_ids[0].numpy())
        if not len(differing):
            raise ValueError("The two prompts must differ by at least one token.")

        batched_ids, output = _run_model(
            model,
            torch.cat([original_ids, changed_ids], dim=0),
            device,
        )
        hidden_difference = np.stack(
            [
                np.linalg.norm(
                    layer[0].detach().float().cpu().numpy()
                    - layer[1].detach().float().cpu().numpy(),
                    axis=-1,
                )
                for layer in output.hidden_states
            ]
        ).astype(np.float64)
        attention = np.stack(
            [layer[0].detach().float().cpu().numpy() for layer in output.attentions]
        )
        sink_score = (
            float(attention[:, :, 1:, 0].mean()) if attention.shape[-1] > 1 else 0.0
        )
        labels = tuple(
            tokenizer.convert_ids_to_tokens(batched_ids[0].detach().cpu().tolist())
        )
        condition_results[include_bos] = (
            hidden_difference,
            sink_score,
            labels,
            int(differing[0]),
        )

    with_bos, without_bos = condition_results[True], condition_results[False]
    return PerturbationProbeResult(
        model_id=model_id,
        tokens_with_bos=with_bos[2],
        tokens_without_bos=without_bos[2],
        difference_with_bos=with_bos[0],
        difference_without_bos=without_bos[0],
        first_changed_with_bos=with_bos[3],
        first_changed_without_bos=without_bos[3],
        sink_score_with_bos=with_bos[1],
        sink_score_without_bos=without_bos[1],
        device=device,
        elapsed_seconds=perf_counter() - started,
    )
