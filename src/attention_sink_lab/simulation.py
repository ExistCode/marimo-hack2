"""Small, deterministic experiments for visualizing attention-driven over-mixing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SimulationConfig:
    """Parameters for a causal attention mixing simulation."""

    tokens: int = 14
    depth: int = 18
    dimensions: int = 24
    sink_strength: float = 0.68
    sink_value_scale: float = 0.06
    mix_rate: float = 0.34
    perturb_token: int = 1
    seed: int = 17

    def __post_init__(self) -> None:
        if self.tokens < 3:
            raise ValueError("tokens must be at least 3")
        if self.depth < 1:
            raise ValueError("depth must be positive")
        if not 0 <= self.sink_strength < 1:
            raise ValueError("sink_strength must be in [0, 1)")
        if not 0 <= self.sink_value_scale <= 1:
            raise ValueError("sink_value_scale must be in [0, 1]")
        if not 0 < self.mix_rate <= 1:
            raise ValueError("mix_rate must be in (0, 1]")
        if not 0 <= self.perturb_token < self.tokens:
            raise ValueError("perturb_token must index an existing token")


@dataclass(frozen=True)
class SimulationResult:
    """Layer-wise measurements produced by :func:`simulate`."""

    attention: FloatArray
    perturbation: FloatArray
    diversity: FloatArray
    effective_rank: FloatArray
    activity: FloatArray
    representations: FloatArray
    perturb_token: int

    @property
    def final_spill(self) -> float:
        """Perturbation magnitude outside the perturbed source token."""

        mask = np.ones(self.perturbation.shape[1], dtype=bool)
        mask[self.perturb_token] = False
        return float(self.perturbation[-1, mask].sum())


def causal_attention(tokens: int, sink_strength: float) -> FloatArray:
    """Construct a causal attention matrix with optional mass on token zero.

    The sink is implemented by moving ``sink_strength`` attention mass to the
    first token while preserving a normalized causal row.
    """

    attention = np.zeros((tokens, tokens), dtype=np.float64)
    for query in range(tokens):
        positions = np.arange(query + 1)
        recency = np.exp(-0.12 * (query - positions))
        row = recency / recency.sum()
        if query > 0 and sink_strength:
            row *= 1.0 - sink_strength
            row[0] += sink_strength
        attention[query, : query + 1] = row
    return attention


def _normalize_rows(values: FloatArray) -> FloatArray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-9)


def _mean_cosine_distance(values: FloatArray) -> float:
    normalized = _normalize_rows(values)
    similarity = normalized @ normalized.T
    upper = similarity[np.triu_indices(len(values), k=1)]
    return float(np.mean(1.0 - upper))


def _effective_rank(values: FloatArray) -> float:
    singular_values = np.linalg.svd(values, compute_uv=False)
    probabilities = singular_values / np.maximum(singular_values.sum(), 1e-12)
    entropy = -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12)))
    return float(np.exp(entropy))


def simulate(config: SimulationConfig, *, with_sink: bool) -> SimulationResult:
    """Run representations and perturbation sensitivity through causal mixing.

    Representation dynamics use residual attention updates. Perturbation spread is
    tracked with the corresponding linear sensitivity recurrence. A first-token
    sink only acts as an approximate no-op when its value vector is small. This
    mirrors the paper's mechanism: high attention on a low-norm value reduces how
    strongly a head updates the residual stream.
    """

    rng = np.random.default_rng(config.seed)
    initial = _normalize_rows(rng.normal(size=(config.tokens, config.dimensions)))
    clean = initial.copy()

    strength = config.sink_strength if with_sink else 0.0
    attention = causal_attention(config.tokens, strength)
    value_scale = np.ones(config.tokens, dtype=np.float64)
    if with_sink and strength > 0:
        value_scale[0] = config.sink_value_scale

    effective_attention = attention * value_scale[None, :]
    token_sensitivity = np.zeros(config.tokens, dtype=np.float64)
    token_sensitivity[config.perturb_token] = 1.0

    layer_perturbation = [token_sensitivity.copy()]
    diversity = [_mean_cosine_distance(clean)]
    effective_rank = [_effective_rank(clean)]
    activity: list[float] = [0.0]
    representations = [clean.copy()]

    for _ in range(config.depth):
        clean_values = clean * value_scale[:, None]
        clean_update = attention @ clean_values

        activity.append(float(np.mean(np.linalg.norm(clean_update, axis=1))))
        clean = _normalize_rows(clean + config.mix_rate * clean_update)
        token_sensitivity = token_sensitivity + config.mix_rate * (
            effective_attention @ token_sensitivity
        )

        layer_perturbation.append(token_sensitivity.copy())
        diversity.append(_mean_cosine_distance(clean))
        effective_rank.append(_effective_rank(clean))
        representations.append(clean.copy())

    return SimulationResult(
        attention=attention,
        perturbation=np.asarray(layer_perturbation),
        diversity=np.asarray(diversity),
        effective_rank=np.asarray(effective_rank),
        activity=np.asarray(activity),
        representations=np.asarray(representations),
        perturb_token=config.perturb_token,
    )
