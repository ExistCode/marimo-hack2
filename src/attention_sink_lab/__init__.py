"""Core experiments for Attention Sink Lab."""

from attention_sink_lab.probe import (
    PerturbationProbeResult,
    ProbeResult,
    probe_attention,
    probe_perturbation,
)
from attention_sink_lab.head_gate import HeadGateResult, run_head_gate
from attention_sink_lab.prompt_suite import PromptSuiteResult, probe_prompt_suite
from attention_sink_lab.simulation import SimulationConfig, SimulationResult, simulate
from attention_sink_lab.sink_swap import SinkSwapResult, run_sink_swap

__all__ = [
    "ProbeResult",
    "PerturbationProbeResult",
    "PromptSuiteResult",
    "HeadGateResult",
    "SinkSwapResult",
    "SimulationConfig",
    "SimulationResult",
    "probe_attention",
    "run_head_gate",
    "probe_prompt_suite",
    "probe_perturbation",
    "run_sink_swap",
    "simulate",
]
