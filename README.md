# Attention Sink Lab

An interactive marimo notebook based on
[Why do LLMs attend to the first token?](https://www.alphaxiv.org/abs/2504.02732).

The notebook contains:

- a deterministic mixing simulation comparing ordinary causal attention with a
  low-value first-token sink;
- reactive controls for context length, depth, sink strength, value norm, and
  mixing rate;
- perturbation-spread, representation-diversity, and head-activity measurements;
- an optional real-model probe for visualizing first-token attention by layer and
  head;
- value-vector norm analysis for testing the paper's approximate no-op account;
- a configurable 5–40 prompt, five-genre suite executed in one padded accelerator
  batch, with bootstrap confidence intervals, synchronized timing, threshold
  sensitivity, and a paper-style `ε=0.80`, 64-token setting;
- a BOS-present/BOS-removed lexical perturbation intervention over real hidden
  states;
- a custom sink-swap extension that replaces BOS with punctuation and lexical
  tokens in one GPU batch, measuring sink preservation against prediction drift;
- a causal head-gating ablation that compares an untouched baseline, a selected
  sink head, and a contrast head in one accelerator forward;
- the paper's LLaMA scale evidence and paper-reported BOS/data-packing ablations, with
  interactive comparisons and explicit metric caveats.

## Run locally

```bash
uv sync
uv run marimo run --no-sandbox notebook.py
```

For notebook editing:

```bash
uv run marimo edit notebook.py
```

The real-model probe downloads the selected Hugging Face checkpoint on first use.
CUDA is preferred, Apple MPS is supported locally, and CPU remains a functional
fallback. The 1.7B model option is intended for the molab GPU runtime; the 135M
model keeps local iteration fast. The synthetic experiment runs immediately and
does not require a model download.

If a 1.7B run exhausts GPU memory, reduce prompts per genre, lower the token
window, shorten the prompt, or switch to the 135M preview model. The notebook
keeps the high-value sections batched, so memory use scales with both batch size
and token window when `output_attentions=True`.

## Validate

```bash
uv run pytest
uv run ruff check .
uv run marimo check notebook.py
```
