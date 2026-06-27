# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "marimo>=0.22.4,<0.23",
#   "numpy>=2.0",
#   "plotly>=6.1",
#   "torch>=2.5",
#   "transformers>=5.0",
# ]
# ///

import marimo

__generated_with = "0.22.5"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from attention_sink_lab.probe import probe_attention, probe_perturbation
    from attention_sink_lab.prompt_suite import probe_prompt_suite
    from attention_sink_lab.simulation import SimulationConfig, simulate
    from attention_sink_lab.sink_swap import run_sink_swap
    from attention_sink_lab.head_gate import run_head_gate

    return (
        SimulationConfig,
        go,
        make_subplots,
        mo,
        np,
        probe_attention,
        probe_prompt_suite,
        probe_perturbation,
        run_head_gate,
        run_sink_swap,
        simulate,
    )


@app.cell
def _(mo):
    mo.Html(
        """
        <style>
        :root {
          --paper: #f6f0e3;
          --paper-deep: #eadfc9;
          --ink: #24343d;
          --ink-soft: #59686d;
          --coral: #ed6a5a;
          --mustard: #efbd4d;
          --teal: #3d8b7d;
          --blue: #527aa3;
        }
        body { background: var(--paper); color: var(--ink); }
        .marimo { font-family: "Avenir Next", "Trebuchet MS", sans-serif; }
        main { max-width: 1260px; margin: 0 auto; }
        h1, h2, h3 { font-family: Georgia, "Times New Roman", serif; color: var(--ink); }
        .lab-hero {
          position: relative;
          overflow: hidden;
          padding: clamp(2.8rem, 7vw, 6.6rem) clamp(1.4rem, 6vw, 5.2rem) 3.2rem;
          border-bottom: 1px solid color-mix(in srgb, var(--ink) 24%, transparent);
        }
        .lab-hero::after {
          content: "01";
          position: absolute;
          right: 3vw;
          top: -2.5rem;
          font: 700 clamp(8rem, 22vw, 19rem)/1 Georgia, serif;
          color: color-mix(in srgb, var(--coral) 11%, transparent);
          pointer-events: none;
        }
        .eyebrow {
          letter-spacing: .16em;
          text-transform: uppercase;
          font-size: .76rem;
          font-weight: 700;
          color: var(--teal);
        }
        .lab-hero h1 {
          max-width: 820px;
          margin: .7rem 0 1rem;
          font-size: clamp(3rem, 8vw, 7.3rem);
          line-height: .9;
          letter-spacing: -.055em;
        }
        .lab-hero p { max-width: 680px; font-size: clamp(1rem, 2vw, 1.25rem); line-height: 1.65; }
        .paper-link { color: var(--ink); text-decoration-color: var(--coral); text-underline-offset: .22em; }
        .experiment-route {
          display: flex;
          flex-wrap: wrap;
          gap: .55rem 1.15rem;
          padding: 1rem clamp(1.4rem, 6vw, 5.2rem);
          border-bottom: 1px solid color-mix(in srgb, var(--ink) 18%, transparent);
          color: var(--ink-soft);
          font-size: .82rem;
          letter-spacing: .04em;
        }
        .experiment-route span { white-space: nowrap; }
        .experiment-route b { color: var(--coral); font-family: Georgia, serif; margin-right: .28rem; }
        .reference-result {
          background: var(--ink);
          color: var(--paper);
          padding: clamp(1.2rem, 3vw, 2rem);
          border-radius: .3rem 1.2rem 1.2rem 1.2rem;
        }
        .reference-result .reference-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
          gap: 1rem;
          margin-top: 1rem;
        }
        .reference-result .reference-value {
          color: #ffd36c;
          font: 700 clamp(1.8rem, 4vw, 3rem)/1 Georgia, serif;
        }
        .reference-result small { color: #d6dfdf; line-height: 1.45; }
        .section-rule {
          display: grid;
          grid-template-columns: 3rem minmax(0, 1fr);
          gap: 1rem;
          align-items: start;
          margin: 4.5rem 0 1.6rem;
        }
        .section-rule .number { color: var(--coral); font: 700 1rem/1 Georgia, serif; padding-top: .48rem; }
        .section-rule h2 { margin: 0; font-size: clamp(2rem, 4vw, 3.8rem); letter-spacing: -.035em; }
        .citation-note {
          max-width: 860px;
          margin: .72rem 0 0;
          color: var(--ink-soft);
          font-size: .94rem;
          line-height: 1.55;
        }
        .citation-note a {
          color: var(--ink);
          text-decoration-color: var(--mustard);
          text-underline-offset: .2em;
        }
        .math-spine {
          background: #fbf7ee;
          border: 1px solid #d8cbb5;
          border-left: .45rem solid var(--mustard);
          padding: clamp(1rem, 2.5vw, 1.45rem);
          border-radius: .35rem 1rem 1rem .35rem;
          margin: 0 0 1.4rem;
        }
        .math-spine p { line-height: 1.55; }
        .math-spine h3 { margin-top: 0; }
        .math-card-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
          gap: .85rem;
          margin-top: 1rem;
        }
        .math-card {
          background: color-mix(in srgb, var(--paper-deep) 48%, #fbf7ee);
          border: 1px solid #d8cbb5;
          border-radius: .35rem .9rem .9rem .9rem;
          padding: .95rem;
        }
        .math-card .plain {
          color: var(--ink-soft);
          font-size: .92rem;
          line-height: 1.5;
        }
        .control-sheet {
          background: color-mix(in srgb, var(--paper-deep) 68%, transparent);
          border: 1px solid color-mix(in srgb, var(--ink) 18%, transparent);
          padding: clamp(1rem, 3vw, 2rem);
          border-radius: 1.2rem 1.2rem .3rem 1.2rem;
        }
        .margin-note {
          font-family: Georgia, "Times New Roman", serif;
          font-style: italic;
          color: var(--ink-soft);
          transform: rotate(-1deg);
        }
        .metric-strip { border-top: 1px solid #baa98d; border-bottom: 1px solid #baa98d; padding: 1rem 0; }
        .claim {
          padding: clamp(1.5rem, 4vw, 3rem);
          background: var(--ink);
          color: var(--paper);
          border-radius: .3rem 1.4rem 1.4rem 1.4rem;
        }
        .claim strong { color: #ffd36c; }
        .judge-path, .paper-map {
          margin: 1.35rem clamp(1.4rem, 6vw, 5.2rem) 0;
          padding: clamp(1.1rem, 3vw, 1.8rem);
          border: 1px solid color-mix(in srgb, var(--ink) 20%, transparent);
          background: color-mix(in srgb, var(--paper-deep) 72%, transparent);
          border-radius: 1.2rem 1.2rem .35rem 1.2rem;
        }
        .judge-grid, .paper-map-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
          gap: .8rem;
          margin-top: .9rem;
        }
        .judge-grid div, .paper-map-grid div {
          border-top: 1px solid color-mix(in srgb, var(--ink) 18%, transparent);
          padding-top: .7rem;
        }
        .judge-grid b, .paper-map-grid b { color: var(--coral); font-family: Georgia, serif; }
        .evidence-table {
          width: 100%;
          border-collapse: collapse;
          background: #fbf7ee;
          border: 1px solid #d8cbb5;
        }
        .evidence-table th, .evidence-table td {
          text-align: left;
          vertical-align: top;
          padding: .78rem .9rem;
          border-bottom: 1px solid #d8cbb5;
          line-height: 1.45;
        }
        .evidence-table th {
          font-size: .75rem;
          letter-spacing: .12em;
          text-transform: uppercase;
          color: var(--teal);
        }
        .plot-frame { background: #fbf7ee; border: 1px solid #d8cbb5; padding: .4rem; border-radius: .55rem; }
        @media (max-width: 720px) {
          .lab-hero::after { opacity: .55; }
          .section-rule { grid-template-columns: 2rem minmax(0, 1fr); }
        }
        @media (prefers-reduced-motion: reduce) { * { scroll-behavior: auto !important; } }
        </style>
        """
    )
    return


@app.cell
def _(mo):
    mo.Html(
        """
        <header class="lab-hero">
          <div class="eyebrow">An interactive research notebook · batched GPU experiment</div>
          <h1>Attention<br/>Sink Lab</h1>
          <p>
            Why would a language model spend attention on a nearly meaningless first token?
            This lab illustrates the paper's proposed answer: a sink can act like a pressure valve,
            slowing the spread of information before deep attention layers mix everything together.
          </p>
          <p><a class="paper-link" href="https://www.alphaxiv.org/abs/2504.02732" target="_blank">
            Read “Why do LLMs attend to the first token?” ↗</a></p>
        </header>
        """
    )
    return


@app.cell
def _(mo):
    mo.Html(
        """
        <div class="experiment-route" aria-label="Five-minute experiment path">
          <span><b>01</b>build intuition</span>
          <span><b>02</b>inspect a model</span>
          <span><b>03</b>batch contexts</span>
          <span><b>04</b>perturb BOS</span>
          <span><b>05</b>invent a substitute</span>
          <span><b>06</b>test causality</span>
          <span><b>07</b>compare scale</span>
          <span><b>08</b>inspect packing</span>
        </div>
        """
    )
    return


@app.cell
def _(mo):
    mo.Html(
        """
        <section class="judge-path" aria-label="Two-minute judge path">
          <div class="eyebrow">Two-minute judge path</div>
          <div style="font:1.35rem/1.35 Georgia,serif; margin-top:.35rem">
            If you only run three cells, run these.
          </div>
          <div class="judge-grid">
            <div><b>02 · Probe attention</b><br/>Shows whether the selected model forms prompt-local
            first-token sinks and whether token zero has a small value vector.</div>
            <div><b>03 · Batch contexts</b><br/>Runs 5–40 prompts as one padded accelerator batch,
            so the single-prompt result has a small robustness check.</div>
            <div><b>05/06 · Extension</b><br/>Sink swap tests first-token substitutes; head gate
            asks whether a selected sink head has lower functional impact than a contrast head.</div>
          </div>
        </section>
        """
    )
    return


@app.cell
def _(mo):
    mo.Html(
        """
        <div class="section-rule"><span class="number">01</span><div>
          <div class="eyebrow">Build the intuition</div>
          <h2>Give attention somewhere harmless to go.</h2>
          <p class="citation-note">
            Citation note: <a href="https://arxiv.org/abs/2504.02732" target="_blank">Barbero et al. (2025)</a>
            frame first-token sinks as a way to reduce over-mixing in deep causal Transformers.
            This section is a toy mechanism sketch, not a trained-model reproduction.
          </p>
        </div></div>
        """
    )
    return


@app.cell
def _(mo):
    import textwrap

    _math_decoder = r"""
    ### Math decoder: the paper's equations in plain English

    The paper's math is mostly asking one question: **are token representations getting mixed
    into the same soup, and can the first token act as a safe place to dump attention?**

    #### 1 · Attention is a weighted average

    $$
    z_i^{(\ell,h)} = \sum_{j \le i} \alpha_{ij}^{(\ell,h)} v_j^{(\ell,h)}
    $$

    Token $i$ updates itself by mixing value vectors $v_j$. The attention weights
    $\alpha_{ij}$ decide how much each earlier token contributes. If every token keeps
    mixing with every other token, their representations can blur together.

    #### 2 · A quiet sink is a low-impact destination

    $$
    \alpha_{i1}^{(\ell,h)} \text{ is large, but }
    \left\lVert v_1^{(\ell,h)} \right\rVert \text{ is small.}
    $$

    This is the "pressure valve" intuition. A head can spend attention on token zero
    without adding much content to the residual stream. In the notebook, this is checked
    by the **BOS value norm vs median** chart and the **head-gate projected contribution**
    measurement.

    #### 3 · Sink rate counts how many heads do this

    $$
    \text{sink rate} =
    \frac{1}{LH}\sum_{\ell,h}
    \mathbf{1}\left(
    \frac{1}{T}\sum_{i=1}^{T}\alpha_{i1}^{(\ell,h)} > \epsilon
    \right)
    $$

    Read it as: "What percent of all heads give token zero more than $\epsilon$ average
    attention?" The notebook's $\epsilon$ slider compares the loose demo threshold $0.30$
    with a stricter paper-style threshold such as $0.80$.

    #### 4 · Rank collapse means all tokens become the same soup

    $$
    \left\lVert
    V^{(L)} - \frac{1}{n}\mathbf{1}\mathbf{1}^{\top}V^{(L)}
    \right\rVert_F < \Delta
    $$

    $V^{(L)}$ is the matrix of all token representations. The middle term means
    "replace every token by the average token." If the distance is tiny, the sequence
    has lost token-specific structure. In the toy plot, this appears as falling
    effective rank.

    #### 5 · Representational collapse is the local version

    $$
    \left\lVert v_n^{(L)} - v_{n-1}^{(L)} \right\rVert_2 < \Delta
    $$

    Instead of checking the whole sequence, this asks whether two neighboring repeated-token
    representations have become almost identical. Barbero et al. show rank collapse implies
    this local collapse, so avoiding the "same soup" problem matters.
    """
    mo.md(textwrap.dedent(_math_decoder))
    return


@app.cell
def _(mo):
    token_count = mo.ui.slider(6, 26, value=14, step=1, show_value=True, label="Context tokens")
    model_depth = mo.ui.slider(4, 40, value=18, step=1, show_value=True, label="Transformer depth")
    sink_strength = mo.ui.slider(
        0.0, 0.92, value=0.68, step=0.02, show_value=True, label="Attention sent to token 0"
    )
    sink_value = mo.ui.slider(
        0.0, 1.0, value=0.06, step=0.02, show_value=True, label="Sink value-vector norm"
    )
    mixing_rate = mo.ui.slider(
        0.08, 0.75, value=0.34, step=0.01, show_value=True, label="Mixing per layer"
    )
    control_panel = mo.vstack(
        [
            mo.md("### Tune the pressure valve\n*A toy path-influence model—not a trained Transformer.*"),
            mo.hstack([token_count, model_depth], widths="equal", gap=1.5),
            sink_strength,
            sink_value,
            mixing_rate,
            mo.md(
                '<p class="margin-note">Perturbation starts at token 1—the second token, '
                "matching the paper's motivating experiment.</p>"
            ),
        ],
        gap=1.25,
    ).style(
        {
            "background": "#eadfc9aa",
            "border": "1px solid #24343d2e",
            "padding": "clamp(1rem, 3vw, 2rem)",
            "border-radius": "1.2rem 1.2rem .3rem 1.2rem",
        }
    )
    control_panel
    return mixing_rate, model_depth, sink_strength, sink_value, token_count


@app.cell
def _(SimulationConfig, mixing_rate, model_depth, sink_strength, sink_value, token_count):
    experiment_config = SimulationConfig(
        tokens=int(token_count.value),
        depth=int(model_depth.value),
        sink_strength=float(sink_strength.value),
        sink_value_scale=float(sink_value.value),
        mix_rate=float(mixing_rate.value),
    )
    return (experiment_config,)


@app.cell
def _(experiment_config, simulate):
    no_sink_result = simulate(experiment_config, with_sink=False)
    sink_result = simulate(experiment_config, with_sink=True)
    return no_sink_result, sink_result


@app.cell
def _(mo, no_sink_result, sink_result):
    spill_reduction = 100 * (
        1 - sink_result.final_spill / max(no_sink_result.final_spill, 1e-9)
    )
    diversity_retained = 100 * (
        sink_result.diversity[-1] / max(no_sink_result.diversity[-1], 1e-9) - 1
    )
    activity_reduction = 100 * (
        1 - sink_result.activity[-1] / max(no_sink_result.activity[-1], 1e-9)
    )
    metric_panel = mo.hstack(
        [
            mo.stat(f"{spill_reduction:.0f}%", "less perturbation spill", bordered=False),
            mo.stat(f"{diversity_retained:+.0f}%", "more final token diversity", bordered=False),
            mo.stat(f"{activity_reduction:.0f}%", "quieter attention update", bordered=False),
        ],
        widths="equal",
        gap=1,
    ).style(
        {
            "border-top": "1px solid #baa98d",
            "border-bottom": "1px solid #baa98d",
            "padding": "1rem 0",
        }
    )
    metric_panel
    return


@app.cell
def _(go, make_subplots, no_sink_result, np, sink_result):
    perturbation_figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("No sink · information spreads", "Sink · spread is contained"),
        horizontal_spacing=0.08,
    )
    heat_colors = [[0, "#f9f4e9"], [0.35, "#efbd4d"], [1, "#ed6a5a"]]
    perturbation_figure.add_trace(
        go.Heatmap(z=no_sink_result.perturbation.T, colorscale=heat_colors, showscale=False),
        row=1,
        col=1,
    )
    perturbation_figure.add_trace(
        go.Heatmap(
            z=sink_result.perturbation.T,
            colorscale=heat_colors,
            colorbar={"title": "change", "thickness": 12},
        ),
        row=1,
        col=2,
    )
    perturbation_figure.update_xaxes(title_text="layer")
    perturbation_figure.update_yaxes(title_text="token", autorange="reversed", row=1, col=1)
    perturbation_figure.update_yaxes(autorange="reversed", row=1, col=2)
    perturbation_figure.update_layout(
        height=430,
        margin={"l": 48, "r": 36, "t": 72, "b": 48},
        paper_bgcolor="#fbf7ee",
        plot_bgcolor="#fbf7ee",
        font={"family": "Avenir Next, sans-serif", "color": "#24343d"},
    )

    layers = np.arange(len(no_sink_result.diversity))
    collapse_figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Representation diversity", "Effective rank"),
        horizontal_spacing=0.12,
    )
    for result, label, color, dash in [
        (no_sink_result, "No sink", "#ed6a5a", "dash"),
        (sink_result, "With sink", "#3d8b7d", "solid"),
    ]:
        collapse_figure.add_trace(
            go.Scatter(x=layers, y=result.diversity, name=label, line={"color": color, "dash": dash, "width": 3}),
            row=1,
            col=1,
        )
        collapse_figure.add_trace(
            go.Scatter(x=layers, y=result.effective_rank, name=label, showlegend=False, line={"color": color, "dash": dash, "width": 3}),
            row=1,
            col=2,
        )
    collapse_figure.update_xaxes(title_text="layer")
    collapse_figure.update_layout(
        height=360,
        margin={"l": 48, "r": 28, "t": 70, "b": 48},
        paper_bgcolor="#fbf7ee",
        plot_bgcolor="#fbf7ee",
        legend={"orientation": "h", "y": 1.16, "x": 0},
        font={"family": "Avenir Next, sans-serif", "color": "#24343d"},
    )
    None
    return collapse_figure, perturbation_figure


@app.cell
def _(collapse_figure, mo, perturbation_figure):
    mo.vstack(
        [
            mo.Html('<div class="plot-frame">'),
            mo.ui.plotly(perturbation_figure),
            mo.Html("</div>"),
            mo.md(
                "The red sensitivity signal starts at token 1. This deliberately simplified "
                "recurrence shows how attention-weighted paths can carry influence to neighboring "
                "tokens. It isolates the paper's proposed mechanism; it does **not** measure a "
                "trained Transformer's hidden-state Jacobian."
            ),
            mo.Html('<div class="plot-frame">'),
            mo.ui.plotly(collapse_figure),
            mo.Html("</div>"),
        ],
        gap=1.1,
    )
    return


@app.cell
def _(mo):
    mo.Html(
        """
        <div class="claim">
          <span class="eyebrow" style="color:#86c8ba">The paper's central claim</span>
          <p style="font: clamp(1.45rem,3vw,2.6rem)/1.25 Georgia,serif; margin:.8rem 0 0">
            Attention sinks are not merely wasted attention. They may let selected heads become
            <strong>quiet by default</strong>, controlling over-mixing as context and depth grow.
          </p>
        </div>
        """
    )
    return


@app.cell
def _(mo):
    mo.Html(
        """
        <section class="paper-map" aria-label="Paper to notebook map">
          <div class="eyebrow">Paper → notebook map</div>
          <div style="font:1.25rem/1.4 Georgia,serif; margin-top:.35rem">
            The notebook separates paper reproduction from new, prompt-local extensions.
          </div>
          <div class="paper-map-grid">
            <div><b>Mechanism</b><br/>Toy recurrence makes the “attention pressure valve” intuition visible,
            but does not claim to reproduce a trained Transformer's Jacobian.</div>
            <div><b>Measured sink</b><br/>Live probes use a local ε=0.30 threshold for interactivity.
            The paper's Table 1 uses a stricter ε=0.80 over 170 prompts and first 64 tokens.</div>
            <div><b>Paper evidence</b><br/>Sections 07–08 show the paper's scale and BOS/data-packing
            results directly, with the stricter metric called out.</div>
            <div><b>Our extension</b><br/>Sink swap and head gate are inference-time tests on the chosen
            off-the-shelf model; they are suggestive controls, not pre-training replications.</div>
          </div>
        </section>
        """
    )
    return


@app.cell
def _(mo):
    mo.Html(
        """
        <section class="paper-map" aria-label="Attention sink terminology lineage">
          <div class="eyebrow">Where the name comes from</div>
          <div style="font:1.25rem/1.4 Georgia,serif; margin-top:.35rem">
            <a class="paper-link" href="https://arxiv.org/abs/2309.17453" target="_blank">
            Xiao et al. (2024)</a> first used the term <em>attention sink</em> for tokens
            that carry little semantic content but attract a large share of an attention head's
            probability mass.
          </div>
          <p style="margin:.75rem 0 0; color:#59686d; line-height:1.55">
            Their StreamingLLM work keeps those initial-token KV states around so sliding-window
            attention stays stable. Barbero et al. starts from that broader phenomenon and asks the
            mechanistic question this notebook explores: why does the first token become such a
            useful sink in the first place?
          </p>
        </section>
        """
    )
    return


@app.cell
def _(mo):
    import torch

    if torch.cuda.is_available():
        accelerator_label = f"CUDA · {torch.cuda.get_device_name(0)}"
        accelerator_message = (
            "GPU connected. Real-model experiments use inference mode, accelerator-resident "
            "tensors, synchronized timing, and batched forwards."
        )
        accelerator_kind = "success"
    elif torch.backends.mps.is_available():
        accelerator_label = "MPS · Apple GPU"
        accelerator_message = (
            "Apple GPU connected locally. The same code selects CUDA automatically on molab. "
            "Real-model experiments keep tensors on the accelerator and batch related prompts."
        )
        accelerator_kind = "success"
    else:
        accelerator_label = "CPU fallback"
        accelerator_message = (
            "No accelerator is visible in this runtime. The notebook still works, but the molab "
            "submission should run with a CUDA GPU to exercise the batched experiments."
        )
        accelerator_kind = "warn"

    mo.callout(
        f"**Accelerator: {accelerator_label}.** {accelerator_message}",
        kind=accelerator_kind,
    )
    return


@app.cell
def _(mo):
    mo.Html(
        """
        <div class="section-rule"><span class="number">02</span><div>
          <div class="eyebrow">Measure a real model</div>
          <h2>Find the heads staring at token zero.</h2>
          <p class="citation-note">
            Citation note: <a href="https://arxiv.org/abs/2410.10781" target="_blank">Gu et al. (2025)</a>
            popularize a sink-rate style metric for heads that attend strongly to the first token;
            <a href="https://arxiv.org/abs/2504.02732" target="_blank">Barbero et al. (2025)</a>
            ask what those heads are doing mechanically.
          </p>
        </div></div>
        """
    )
    return


@app.cell
def _(mo):
    probe_model = mo.ui.dropdown(
        {
            "SmolLM2 1.7B · competition GPU": "HuggingFaceTB/SmolLM2-1.7B",
            "SmolLM2 135M · local preview": "HuggingFaceTB/SmolLM2-135M",
        },
        value=(
            "SmolLM2 1.7B · competition GPU"
            if __import__("torch").cuda.is_available()
            else "SmolLM2 135M · local preview"
        ),
        label="Model",
        full_width=True,
    )
    probe_prompt = mo.ui.text_area(
        value=(
            "The harbor was quiet at dawn. A red bicycle leaned against the bakery wall, "
            "and someone had left a handwritten map beneath its basket."
        ),
        label="Prompt",
        rows=4,
        full_width=True,
    )
    probe_threshold = mo.ui.slider(
        0.1,
        0.8,
        value=0.3,
        step=0.05,
        show_value=True,
        label="Strong-sink threshold ε",
    )
    run_probe = mo.ui.run_button(label="Probe attention", kind="success", full_width=True)
    probe_panel = mo.vstack(
        [
            mo.callout(
                "CUDA defaults to the 1.7B competition model; local preview uses 135M. The first "
                "live run downloads the selected checkpoint, and nothing runs until you click. "
                "Use ε=0.30 for exploration; use ε=0.80 to mirror the paper's strict Table 1 threshold.",
                kind="info",
            ),
            probe_model,
            probe_prompt,
            probe_threshold,
            run_probe,
        ],
        gap=1,
    )
    probe_panel
    return probe_model, probe_prompt, probe_threshold, run_probe


@app.cell
def _(mo):
    mo.Html(
        """
          <div class="reference-result">
          <div class="eyebrow" style="color:#86c8ba">Immediate reference · reproducible below</div>
          <div style="font:1.35rem/1.35 Georgia,serif; margin-top:.45rem">
            SmolLM2 135M on the default prompt shows a pattern consistent with the paper's
            proposed no-op mechanism.
          </div>
          <div class="reference-grid">
            <div><div class="reference-value">86%</div><small>heads above this notebook's local 0.30 sink threshold</small></div>
            <div><div class="reference-value">97%</div><small>attention sent to token zero by layer 23 · head 7</small></div>
            <div><div class="reference-value">0.03×</div><small>BOS value norm relative to later tokens</small></div>
          </div>
          <small style="display:block; margin-top:1rem">Cached reference from this notebook's default prompt: HuggingFaceTB/SmolLM2-135M, max_tokens=96, ε=0.30, local MPS preview. Click “Probe attention” to replace it with a live measurement.</small>
        </div>
        """
    )
    return


@app.cell
def _(mo, probe_attention, probe_model, probe_prompt, run_probe):
    mo.stop(
        not run_probe.value,
        mo.md(
            '<p class="margin-note">Run the probe to replace the toy matrices with attention '
            "measured from a causal language model.</p>"
        ),
    )
    try:
        measured_probe = probe_attention(probe_model.value, probe_prompt.value)
    except Exception as probe_error:
        probe_error_text = str(probe_error)
        probe_oom_hint = (
            " If this is a GPU memory issue, switch to the 135M preview model for local debugging, "
            "then rerun the 1.7B model on molab GPU."
            if "memory" in probe_error_text.lower() or "oom" in probe_error_text.lower()
            else ""
        )
        mo.stop(True, mo.callout(f"Model probe failed: {probe_error}{probe_oom_hint}", kind="danger"))
    return (measured_probe,)


@app.cell
def _(measured_probe, mo, np):
    default_layer = min(measured_probe.layers // 2, measured_probe.layers - 1)
    default_head = 0
    if measured_probe.value_norms is not None and len(measured_probe.tokens) > 1:
        value_ratios = measured_probe.value_norms[:, :, 0] / np.maximum(
            np.median(measured_probe.value_norms[:, :, 1:], axis=2), 1e-9
        )
        quiet_sink_score = measured_probe.sink_profile / np.maximum(value_ratios, 0.02)
        default_layer, default_head = np.unravel_index(
            int(np.argmax(quiet_sink_score)), quiet_sink_score.shape
        )
    selected_layer = mo.ui.slider(
        0,
        measured_probe.layers - 1,
        value=int(default_layer),
        step=1,
        show_value=True,
        label="Layer",
    )
    selected_head = mo.ui.slider(
        0,
        measured_probe.heads - 1,
        value=int(default_head),
        step=1,
        show_value=True,
        label="Head",
    )
    return selected_head, selected_layer


@app.cell
def _(go, make_subplots, measured_probe, np, probe_threshold, selected_head, selected_layer):
    probe_head_tick_step = max(1, measured_probe.heads // 12)
    probe_layer_tick_step = max(1, measured_probe.layers // 12)
    probe_head_ticks = list(range(0, measured_probe.heads, probe_head_tick_step))
    probe_layer_ticks = list(range(0, measured_probe.layers, probe_layer_tick_step))
    probe_profile_figure = go.Figure(
        data=go.Heatmap(
            z=measured_probe.sink_profile,
            colorscale=[[0, "#f9f4e9"], [0.45, "#efbd4d"], [1, "#ed6a5a"]],
            zmin=0,
            zmax=max(float(probe_threshold.value), float(np.max(measured_probe.sink_profile))),
            colorbar={"title": "attention → token 0", "thickness": 12},
        )
    )
    probe_profile_figure.update_xaxes(
        tickmode="array",
        tickvals=probe_head_ticks,
        ticktext=[str(head) for head in probe_head_ticks],
        range=[-0.5, measured_probe.heads - 0.5],
    )
    probe_profile_figure.update_yaxes(
        tickmode="array",
        tickvals=probe_layer_ticks,
        ticktext=[str(layer) for layer in probe_layer_ticks],
        range=[measured_probe.layers - 0.5, -0.5],
    )
    probe_profile_figure.update_layout(
        title=f"First-token attention across the model · ε={float(probe_threshold.value):.2f}",
        xaxis_title="head",
        yaxis_title="layer",
        height=390,
        margin={"l": 48, "r": 32, "t": 64, "b": 48},
        paper_bgcolor="#fbf7ee",
        plot_bgcolor="#fbf7ee",
        font={"family": "Avenir Next, sans-serif", "color": "#24343d"},
    )

    token_labels = [token.replace("Ġ", "▁").replace("▁▁", "▁") for token in measured_probe.tokens]
    head_matrix = measured_probe.attentions[selected_layer.value, selected_head.value]
    probe_head_figure = go.Figure(
        data=go.Heatmap(
            z=head_matrix,
            x=token_labels,
            y=token_labels,
            colorscale=[[0, "#f9f4e9"], [0.4, "#527aa3"], [1, "#24343d"]],
            colorbar={"title": "attention", "thickness": 12},
        )
    )
    probe_head_figure.update_layout(
        title=f"Layer {selected_layer.value} · head {selected_head.value}",
        xaxis_title="key token",
        yaxis_title="query token",
        yaxis={"autorange": "reversed"},
        height=520,
        margin={"l": 70, "r": 32, "t": 64, "b": 90},
        paper_bgcolor="#fbf7ee",
        plot_bgcolor="#fbf7ee",
        font={"family": "Avenir Next, sans-serif", "color": "#24343d"},
    )

    probe_quiet_figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Mean attention received", "Value-vector norm"),
        horizontal_spacing=0.12,
    )
    mean_attention_received = head_matrix[1:].mean(axis=0)
    value_norms_for_head = (
        measured_probe.value_norms[selected_layer.value, selected_head.value]
        if measured_probe.value_norms is not None
        else np.zeros(len(token_labels))
    )
    token_colors = ["#ed6a5a", *(["#3d8b7d"] * (len(token_labels) - 1))]
    probe_quiet_figure.add_trace(
        go.Bar(
            x=token_labels,
            y=mean_attention_received,
            marker_color=token_colors,
            name="attention",
        ),
        row=1,
        col=1,
    )
    probe_quiet_figure.add_trace(
        go.Bar(
            x=token_labels,
            y=value_norms_for_head,
            marker_color=token_colors,
            name="value norm",
        ),
        row=1,
        col=2,
    )
    probe_quiet_figure.update_xaxes(tickangle=-55)
    probe_quiet_figure.update_layout(
        title="Does the sink implement an approximate no-op?",
        height=410,
        showlegend=False,
        margin={"l": 48, "r": 28, "t": 72, "b": 110},
        paper_bgcolor="#fbf7ee",
        plot_bgcolor="#fbf7ee",
        font={"family": "Avenir Next, sans-serif", "color": "#24343d"},
    )
    None
    return probe_head_figure, probe_profile_figure, probe_quiet_figure


@app.cell
def _(
    measured_probe,
    mo,
    probe_threshold,
    probe_head_figure,
    probe_profile_figure,
    probe_quiet_figure,
    selected_head,
    selected_layer,
):
    probe_threshold_value = float(probe_threshold.value)
    probe_sink_fraction = measured_probe.sink_fraction(probe_threshold_value)
    if probe_sink_fraction <= 0.01:
        probe_visuals = [
            mo.callout(
                f"This model produced no heads above ε={probe_threshold_value:.2f} for the current "
                "prompt. That is a valid negative result, so the notebook hides the visually "
                "empty heatmaps instead of presenting them as evidence.",
                kind="warn",
            )
        ]
    else:
        probe_visuals = [
            mo.ui.plotly(probe_profile_figure),
            mo.hstack([selected_layer, selected_head], widths="equal", gap=1.5),
            mo.ui.plotly(probe_head_figure),
            mo.hstack(
                [
                    mo.stat(
                        f"{measured_probe.sink_profile[selected_layer.value, selected_head.value]:.0%}",
                        "attention sent to token 0",
                    ),
                    mo.stat(
                        (
                            f"{measured_probe.bos_value_ratio(selected_layer.value, selected_head.value):.2f}×"
                            if measured_probe.bos_value_ratio(
                                selected_layer.value, selected_head.value
                            )
                            is not None
                            else "n/a"
                        ),
                        "BOS value norm vs median",
                    ),
                ],
                widths="equal",
                gap=1,
            ),
            mo.ui.plotly(probe_quiet_figure),
            mo.md(
                "The coral bar is token zero. A head is consistent with the paper's approximate "
                "no-op interpretation when it sends substantial attention there **and** token zero's "
                "value norm is small. The default selector searches for that combination; the head "
                "gate below adds a functional-impact check."
            ),
        ]
    mo.vstack(
        [
            mo.hstack(
                [
                    mo.stat(
                        f"{probe_sink_fraction:.0%}",
                        f"prompt-local heads above ε={probe_threshold_value:.2f}",
                    ),
                    mo.stat(str(len(measured_probe.tokens)), "tokens measured"),
                    mo.stat(measured_probe.device.upper(), "compute device", caption=f"{measured_probe.elapsed_seconds:.1f}s"),
                ],
                widths="equal",
                gap=1,
            ),
            *probe_visuals,
        ],
        gap=1.25,
    )
    return


@app.cell
def _(mo):
    mo.Html(
        """
        <div class="section-rule"><span class="number">03</span><div>
          <div class="eyebrow">Measure across contexts</div>
          <h2>One prompt can lie.</h2>
          <p class="citation-note">
            Citation note: the paper reports sink statistics over many prompts/windows rather than
            one hand-picked input. This notebook keeps that spirit with a small five-genre batch and
            explicitly exposes the demo threshold versus the stricter paper-style threshold.
          </p>
        </div></div>
        """
    )
    return


@app.cell
def _(mo):
    suite_names = ("Narrative", "Code", "Dialogue", "Science", "Instructions")
    suite_prompt_bank = {
        "Narrative": (
            "At sunrise the harbor master found a red bicycle beside the bakery, with a folded "
            "map tucked beneath its basket and a note addressed to nobody in particular.",
            "Rain tapped the observatory windows while Lena compared two star charts and noticed "
            "a mark that had not appeared in the previous night's survey.",
            "The last tram stopped between stations, its lights still glowing as every passenger "
            "heard the same distant bell from the tunnel ahead.",
            "A gardener opened the greenhouse before dawn and discovered a trail of blue petals "
            "leading from the locked door to an empty clay pot.",
            "On the island's quietest morning, the radio operator received a weather report dated "
            "three years in the future and signed with her own initials.",
            "The museum guard found one portrait turned toward the wall, though the security film "
            "showed nobody entering the gallery overnight.",
            "A child left a paper boat in the dry fountain, and by evening its handwritten route "
            "matched the streets flooded by an unexpected storm.",
            "When the mountain library reopened, a warm cup of tea waited beside a book that had "
            "been missing from the catalogue for decades.",
        ),
        "Code": (
            "Write a Python function that groups timestamped events by user, sorts each group "
            "chronologically, and returns sessions containing more than three events.",
            "Implement a stable merge of two sorted iterators without materializing either input, "
            "and include type hints plus tests for duplicate values.",
            "Create a function that validates a directed acyclic graph and returns a topological "
            "ordering or a readable cycle path when validation fails.",
            "Write an asynchronous rate limiter that allows five requests per second per client "
            "and remains correct when tasks are cancelled.",
            "Implement a streaming parser for newline-delimited JSON that reports malformed line "
            "numbers while yielding valid records without loading the file into memory.",
            "Create a NumPy function that computes pairwise cosine distance in chunks and never "
            "allocates the full square similarity matrix.",
            "Write a cache decorator with a time-to-live, bounded capacity, and deterministic "
            "least-recently-used eviction under concurrent access.",
            "Implement a compact diff for two nested dictionaries that distinguishes additions, "
            "deletions, and changed scalar values using tuple paths.",
        ),
        "Dialogue": (
            "Mara asked whether the lighthouse was still operating. Ilyas looked toward the fog "
            "and explained why the old rotating lens only runs during winter storms.",
            "Are you certain the samples stayed frozen, the technician asked. The courier checked "
            "the temperature log twice before answering.",
            "I thought the launch was tomorrow, Nia said. Omar pointed to the revised schedule and "
            "asked who had approved the overnight change.",
            "The map shows a bridge here, said Jun. There was one, replied the guide, before the "
            "river changed direction after the earthquake.",
            "Why did the alarm stop, the doctor asked. Because the backup sensor took over, the "
            "engineer said, but its readings are drifting.",
            "You translated the final line differently, Ada observed. The archivist nodded and "
            "placed two older editions beside the manuscript.",
            "Could the signal be reflected, Malik asked. Only if the source is below the horizon, "
            "the astronomer replied while rotating the antenna plot.",
            "We have enough fuel to return, said the pilot. The navigator quietly highlighted a "
            "new storm cell forming across their planned route.",
        ),
        "Science": (
            "Ocean circulation transports heat, carbon, and nutrients across enormous distances. "
            "Small changes in salinity alter water density and reshape circulation patterns.",
            "CRISPR systems evolved as microbial immune defenses, using stored genetic fragments "
            "to recognize and cut matching sequences during later infections.",
            "Superconductors carry current without electrical resistance below a critical "
            "temperature, but magnetic fields and material defects constrain practical devices.",
            "Pollinating insects integrate color, scent, and spatial memory when choosing flowers, "
            "creating feedback between plant traits and local foraging behavior.",
            "Gravitational lensing bends light around massive objects and lets astronomers infer "
            "matter distributions that may emit little or no detectable radiation.",
            "Sleep consolidates some memories by replaying neural activity patterns, while other "
            "representations weaken as synaptic connections are selectively adjusted.",
            "Volcanic aerosols can cool the climate by reflecting sunlight, although their short "
            "atmospheric lifetime differs from the persistence of greenhouse gases.",
            "Protein folding balances local chemical interactions with long-range constraints, "
            "producing stable structures from a vast landscape of possible conformations.",
        ),
        "Instructions": (
            "Before calibrating the sensor, disconnect the battery, record the ambient "
            "temperature, clean the optical window, and wait thirty seconds before restoring power.",
            "To archive the project, verify the checksum, remove generated caches, create a signed "
            "tag, and copy the release bundle to both storage locations.",
            "Rinse the filter with distilled water, seat the gasket evenly, tighten opposite bolts "
            "in sequence, and inspect the housing for leaks under low pressure.",
            "Export the timeline as XML, preserve source frame rates, relink missing audio, and "
            "watch the rendered file completely before delivering it.",
            "Place the seedlings twelve centimeters apart, water from below, label each variety, "
            "and rotate the tray every morning to keep growth even.",
            "Enable airplane mode, restart the receiver, open the diagnostics screen, and record "
            "the firmware version before pairing another device.",
            "Warm the pan over medium heat, toast the spices until fragrant, add the onions, and "
            "lower the temperature before the garlic begins to brown.",
            "Lock the stage, remove the protective cap, apply one drop of immersion oil, and focus "
            "with the fine adjustment only at high magnification.",
        ),
    }
    suite_samples = mo.ui.slider(
        1,
        8,
        value=4,
        step=1,
        show_value=True,
        label="Prompts per genre",
    )
    suite_threshold = mo.ui.slider(
        0.1,
        0.8,
        value=0.3,
        step=0.05,
        show_value=True,
        label="Strong-sink threshold ε",
    )
    suite_max_tokens = mo.ui.slider(
        32,
        128,
        value=64,
        step=16,
        show_value=True,
        label="Token window",
    )
    run_suite = mo.ui.run_button(
        label="Batch the prompt suite on the accelerator", kind="neutral", full_width=True
    )
    suite_panel = mo.vstack(
        [
            mo.callout(
                "The paper aggregates many fixed-length windows. Choose 1–8 prompts per genre, "
                "then measure all five genres in one padded accelerator batch. The closest paper-style "
                "setting here is ε=0.80 with a 64-token window; the lighter demo setting is ε=0.30.",
                kind="info",
            ),
            suite_samples,
            mo.vstack([suite_threshold, suite_max_tokens], gap=0.75),
            run_suite,
        ],
        gap=1,
    )
    suite_panel
    return run_suite, suite_max_tokens, suite_names, suite_prompt_bank, suite_samples, suite_threshold


@app.cell
def _(mo):
    mo.Html(
        """
        <div class="reference-result">
          <div class="eyebrow" style="color:#86c8ba">Static checkpoint · before you run it</div>
          <div style="font:1.25rem/1.4 Georgia,serif; margin-top:.45rem">
            A useful prompt-suite result is not “all red.” It is a robustness check: do sink-heavy
            layers survive when the text changes from story → code → dialogue → science → instructions?
          </div>
          <div class="reference-grid">
            <div><div class="reference-value">5–40</div><small>prompts measured in one padded batch</small></div>
            <div><div class="reference-value">0.30 / 0.80</div><small>toggle between local exploratory and paper-style strict thresholds</small></div>
            <div><div class="reference-value">CI</div><small>bootstrap intervals show genre-level variation, not population certainty</small></div>
          </div>
        </div>
        """
    )
    return


@app.cell
def _(suite_names, suite_prompt_bank, suite_samples):
    suite_prompts = tuple(
        prompt
        for genre in suite_names
        for prompt in suite_prompt_bank[genre][: suite_samples.value]
    )
    suite_genres = tuple(
        genre for genre in suite_names for _ in range(suite_samples.value)
    )
    return suite_genres, suite_prompts


@app.cell
def _(
    mo,
    probe_model,
    probe_prompt_suite,
    run_suite,
    suite_max_tokens,
    suite_prompts,
    suite_threshold,
):
    mo.stop(
        not run_suite.value,
        mo.md(
            '<p class="margin-note">Run the suite to see whether the single-prompt result '
            "survives a change of genre.</p>"
        ),
    )
    try:
        prompt_suite_result = probe_prompt_suite(
            probe_model.value,
            suite_prompts,
            max_tokens=int(suite_max_tokens.value),
            threshold=float(suite_threshold.value),
        )
    except Exception as suite_error:
        suite_error_text = str(suite_error)
        oom_hint = (
            " If this is a GPU memory issue, lower prompts per genre, reduce the token window, "
            "or switch to the 135M preview model. The 1.7B model with output attentions can be memory-heavy."
            if "memory" in suite_error_text.lower() or "oom" in suite_error_text.lower()
            else ""
        )
        mo.stop(True, mo.callout(f"Prompt suite failed: {suite_error}{oom_hint}", kind="danger"))
    return (prompt_suite_result,)


@app.cell
def _(go, make_subplots, np, prompt_suite_result, suite_genres, suite_names, suite_threshold):
    suite_profile = prompt_suite_result.mean_sink_profile
    _suite_threshold_value = float(suite_threshold.value)
    suite_color_max = max(_suite_threshold_value, float(np.max(suite_profile)))
    suite_head_ticks = list(range(prompt_suite_result.heads))
    suite_layer_ticks = list(range(prompt_suite_result.layers))
    suite_genre_values = [
        np.asarray(
            [
                value
                for value, prompt_genre in zip(
                    prompt_suite_result.sink_fractions, suite_genres, strict=True
                )
                if prompt_genre == genre
            ]
        )
        for genre in suite_names
    ]
    suite_genre_means = [float(values.mean()) for values in suite_genre_values]
    suite_rng = np.random.default_rng(7)
    suite_error_minus = []
    suite_error_plus = []
    for genre_values in suite_genre_values:
        if len(genre_values) == 1:
            lower = upper = float(genre_values[0])
        else:
            bootstrap_means = suite_rng.choice(
                genre_values,
                size=(1000, len(genre_values)),
                replace=True,
            ).mean(axis=1)
            lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
        mean_value = float(genre_values.mean())
        suite_error_minus.append(mean_value - lower)
        suite_error_plus.append(upper - mean_value)
    suite_figure = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.38, 0.62],
        subplot_titles=(f"Heads above ε={_suite_threshold_value:.2f}", "Mean attention → token 0"),
        vertical_spacing=0.14,
    )
    suite_figure.add_trace(
        go.Bar(
            x=suite_genre_means,
            y=list(suite_names),
            orientation="h",
            marker_color=["#ed6a5a", "#efbd4d", "#3d8b7d", "#527aa3", "#9a6b8f"],
            text=[f"{value:.0%}" for value in suite_genre_means],
            textposition="outside",
            cliponaxis=False,
            error_x={
                "type": "data",
                "array": suite_error_plus,
                "arrayminus": suite_error_minus,
                "color": "#24343d",
                "thickness": 1.2,
            },
        ),
        row=1,
        col=1,
    )
    suite_figure.add_trace(
        go.Heatmap(
            z=suite_profile,
            colorscale=[[0, "#f9f4e9"], [0.45, "#efbd4d"], [1, "#ed6a5a"]],
            zmin=0,
            zmax=suite_color_max,
            colorbar={
                "title": {"text": "mean attention<br>to token 0", "side": "top"},
                "thickness": 10,
                "len": 0.46,
                "y": 0.24,
                "yanchor": "middle",
            },
        ),
        row=2,
        col=1,
    )
    suite_figure.update_xaxes(
        title_text="fraction of heads", range=[0, 1.12], tickformat=".0%", row=1, col=1
    )
    suite_figure.update_yaxes(autorange="reversed", row=1, col=1)
    suite_figure.update_xaxes(
        title_text="head",
        tickmode="array",
        tickvals=suite_head_ticks,
        ticktext=[str(head) for head in suite_head_ticks],
        row=2,
        col=1,
    )
    suite_figure.update_yaxes(
        title_text="layer",
        tickmode="array",
        tickvals=suite_layer_ticks,
        ticktext=[str(layer) for layer in suite_layer_ticks],
        autorange="reversed",
        row=2,
        col=1,
    )
    suite_figure.update_layout(
        height=690,
        showlegend=False,
        margin={"l": 72, "r": 48, "t": 64, "b": 52},
        paper_bgcolor="#fbf7ee",
        plot_bgcolor="#fbf7ee",
        font={"family": "Avenir Next, sans-serif", "color": "#24343d"},
    )
    suite_figure.update_annotations(font={"size": 16})
    None
    return suite_figure, suite_genre_means


@app.cell
def _(mo, np, prompt_suite_result, suite_figure, suite_genre_means, suite_threshold):
    _suite_threshold_value = float(suite_threshold.value)
    suite_mean = float(np.mean(prompt_suite_result.sink_fractions))
    suite_range = float(
        max(suite_genre_means) - min(suite_genre_means)
    )
    if suite_mean == 0:
        suite_interpretation = mo.callout(
            f"No head crossed ε={_suite_threshold_value:.2f} in this prompt suite. This is a valid "
            "negative result; the heatmap still shows lower-strength attention rather than "
            "pretending the experiment failed.",
            kind="warn",
        )
    else:
        suite_interpretation = mo.md(
            "A stable red band across genres is stronger evidence than a single dramatic "
            "prompt. This remains a lightweight demonstration—not the paper's 170-prompt, "
            "fixed-window evaluation."
        )
    mo.vstack(
        [
            mo.hstack(
                [
                    mo.stat(f"{suite_mean:.0%}", "mean prompt-local sink fraction"),
                    mo.stat(f"{suite_range:.0%}", "variation across genres"),
                    mo.stat(str(prompt_suite_result.prompt_count), "prompts aggregated"),
                ],
                widths="equal",
                gap=1,
            ),
            mo.ui.plotly(suite_figure),
            mo.callout(
                f"One {prompt_suite_result.device.upper()} forward processed "
                f"{prompt_suite_result.batch_size} prompts and "
                f"{prompt_suite_result.token_count} valid tokens in "
                f"{prompt_suite_result.elapsed_seconds * 1000:.0f} ms. Padding queries are "
                f"masked out of every sink score. Threshold ε={_suite_threshold_value:.2f}.",
                kind="success" if prompt_suite_result.device != "cpu" else "warn",
            ),
            suite_interpretation,
        ],
        gap=1.1,
    )
    return


@app.cell
def _(go, np, prompt_suite_result):
    threshold_grid = np.linspace(0.1, 0.8, 15)
    threshold_sensitivity = [
        float(np.mean(prompt_suite_result.mean_sink_profile >= threshold))
        for threshold in threshold_grid
    ]
    sensitivity_figure = go.Figure(
        data=go.Scatter(
            x=threshold_grid,
            y=threshold_sensitivity,
            mode="lines+markers",
            line={"color": "#24343d", "width": 3},
            marker={"color": "#ed6a5a", "size": 8},
            hovertemplate="ε=%{x:.2f}<br>mean-profile heads %{y:.0%}<extra></extra>",
        )
    )
    sensitivity_figure.add_vline(
        x=0.3,
        line_dash="dash",
        line_color="#3d8b7d",
        annotation_text="demo ε=.30",
        annotation_position="top left",
    )
    sensitivity_figure.add_vline(
        x=0.8,
        line_dash="dash",
        line_color="#ed6a5a",
        annotation_text="paper ε=.80",
        annotation_position="top right",
    )
    sensitivity_figure.update_xaxes(title_text="sink threshold ε", range=[0.08, 0.82])
    sensitivity_figure.update_yaxes(title_text="fraction of heads", tickformat=".0%", range=[0, 1])
    sensitivity_figure.update_layout(
        title="Threshold sensitivity on the mean prompt-suite profile",
        height=350,
        margin={"l": 62, "r": 36, "t": 66, "b": 54},
        paper_bgcolor="#fbf7ee",
        plot_bgcolor="#fbf7ee",
        font={"family": "Avenir Next, sans-serif", "color": "#24343d"},
    )
    None
    return sensitivity_figure, threshold_grid, threshold_sensitivity


@app.cell
def _(mo, sensitivity_figure, threshold_sensitivity):
    demo_fraction = float(threshold_sensitivity[4])
    paper_fraction = float(threshold_sensitivity[-1])
    mo.vstack(
        [
            mo.ui.plotly(sensitivity_figure),
            mo.callout(
                f"On the mean profile, {demo_fraction:.0%} of heads clear ε=0.30, while "
                f"{paper_fraction:.0%} clear ε=0.80. This makes the threshold choice visible "
                "instead of burying it in a caveat.",
                kind="neutral",
            ),
        ],
        gap=1,
    )
    return


@app.cell
def _(mo):
    mo.Html(
        """
        <div class="section-rule"><span class="number">04</span><div>
          <div class="eyebrow">Reproduce the intervention</div>
          <h2>Change one word. Watch the disturbance travel.</h2>
          <p class="citation-note">
            Citation note: <a href="https://arxiv.org/abs/2504.02732" target="_blank">Barbero et al. (2025)</a>
            use perturbation-style evidence to connect first-token sinks with controlled information
            mixing. This notebook's BOS-present/BOS-removed probe is intentionally smaller and notes
            the positional-shift caveat.
          </p>
        </div></div>
        """
    )
    return


@app.cell
def _(mo):
    perturb_original = mo.ui.text_area(
        value=(
            "The greatest scientist in the world walked into the quiet harbor at dawn, carrying "
            "a red notebook filled with careful observations about the tides and weather."
        ),
        label="Original prompt",
        rows=3,
        full_width=True,
    )
    perturb_changed = mo.ui.text_area(
        value=(
            "The best scientist in the world walked into the quiet harbor at dawn, carrying a "
            "red notebook filled with careful observations about the tides and weather."
        ),
        label="One-token perturbation",
        rows=3,
        full_width=True,
    )
    run_intervention = mo.ui.run_button(
        label="Run BOS intervention", kind="warn", full_width=True
    )
    intervention_panel = mo.vstack(
        [
            mo.callout(
                "Keep both prompts the same token length. The experiment compares hidden-state "
                "differences with the BOS token present versus removed.",
                kind="neutral",
            ),
            mo.hstack([perturb_original, perturb_changed], widths="equal", gap=1.25),
            run_intervention,
        ],
        gap=1,
    )
    intervention_panel
    return perturb_changed, perturb_original, run_intervention


@app.cell
def _(mo):
    mo.Html(
        """
        <div class="reference-result">
          <div class="eyebrow" style="color:#86c8ba">Control note · BOS removal is imperfect</div>
          <div style="font:1.25rem/1.4 Georgia,serif; margin-top:.45rem">
            Removing BOS also shifts positions, so this cell is presented as a lightweight
            perturbation probe. A stricter follow-up would preserve sequence length by replacing
            BOS with a neutral token or intervening on the embedding directly.
          </div>
          <div class="reference-grid">
            <div><div class="reference-value">2×2</div><small>original/changed prompt with BOS present and absent</small></div>
            <div><div class="reference-value">L2</div><small>hidden-state difference tracks how a one-word change spreads</small></div>
            <div><div class="reference-value">caveat</div><small>positive containment supports the story; negative results are informative</small></div>
          </div>
        </div>
        """
    )
    return


@app.cell
def _(
    mo,
    perturb_changed,
    perturb_original,
    probe_model,
    probe_perturbation,
    run_intervention,
):
    mo.stop(
        not run_intervention.value,
        mo.md(
            '<p class="margin-note">Run this after choosing a model above. The model cache is '
            "reused when available.</p>"
        ),
    )
    try:
        perturb_probe_result = probe_perturbation(
            probe_model.value,
            perturb_original.value,
            perturb_changed.value,
        )
    except Exception as intervention_error:
        mo.stop(True, mo.callout(f"Intervention failed: {intervention_error}", kind="danger"))
    return (perturb_probe_result,)


@app.cell
def _(go, make_subplots, perturb_probe_result):
    intervention_figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("BOS present · sink available", "BOS removed"),
        horizontal_spacing=0.1,
    )
    intervention_scale = [[0, "#f9f4e9"], [0.4, "#efbd4d"], [1, "#ed6a5a"]]
    intervention_max = max(
        float(perturb_probe_result.difference_with_bos.max()),
        float(perturb_probe_result.difference_without_bos.max()),
        1e-9,
    )
    intervention_figure.add_trace(
        go.Heatmap(
            z=perturb_probe_result.difference_with_bos.T,
            colorscale=intervention_scale,
            zmin=0,
            zmax=intervention_max,
            showscale=False,
        ),
        row=1,
        col=1,
    )
    intervention_figure.add_trace(
        go.Heatmap(
            z=perturb_probe_result.difference_without_bos.T,
            colorscale=intervention_scale,
            zmin=0,
            zmax=intervention_max,
            colorbar={"title": "hidden L2 change", "thickness": 12},
        ),
        row=1,
        col=2,
    )
    intervention_figure.update_xaxes(title_text="layer")
    intervention_figure.update_yaxes(title_text="token", autorange="reversed", row=1, col=1)
    intervention_figure.update_yaxes(autorange="reversed", row=1, col=2)
    intervention_figure.update_layout(
        height=430,
        margin={"l": 48, "r": 36, "t": 72, "b": 48},
        paper_bgcolor="#fbf7ee",
        plot_bgcolor="#fbf7ee",
        font={"family": "Avenir Next, sans-serif", "color": "#24343d"},
    )
    None
    return (intervention_figure,)


@app.cell
def _(intervention_figure, mo, perturb_probe_result):
    bos_reduction = 100 * (
        1
        - perturb_probe_result.downstream_with_bos
        / max(perturb_probe_result.downstream_without_bos, 1e-9)
    )
    mo.vstack(
        [
            mo.hstack(
                [
                    mo.stat(
                        f"{perturb_probe_result.downstream_with_bos:.2f}",
                        "downstream change · BOS",
                    ),
                    mo.stat(
                        f"{perturb_probe_result.downstream_without_bos:.2f}",
                        "downstream change · no BOS",
                    ),
                    mo.stat(f"{bos_reduction:+.0f}%", "change contained by BOS"),
                ],
                widths="equal",
                gap=1,
            ),
            mo.ui.plotly(intervention_figure),
            mo.md(
                "This is a small-scale version of the paper's perturbation analysis. A positive "
                "containment value means the lexical change affected downstream tokens less when "
                "the first-token sink candidate was available. Results are prompt- and model-specific."
            ),
        ],
        gap=1.1,
    )
    return


@app.cell
def _(mo):
    mo.Html(
        """
        <div class="section-rule"><span class="number">05</span><div>
          <div class="eyebrow">Add our own intervention</div>
          <h2>Can punctuation impersonate BOS?</h2>
          <p class="citation-note">
            Citation note: <a href="https://arxiv.org/abs/2309.17453" target="_blank">Xiao et al. (2024)</a>
            showed initial tokens can act as useful attention sinks for streaming inference.
            This section asks a new inference-time question: can a different first token preserve
            that sink without changing predictions much?
          </p>
        </div></div>
        """
    )
    return


@app.cell
def _(mo):
    swap_candidates = (
        ("Native BOS", ""),
        ("Period", "."),
        ("Comma", ","),
        ("Newline", "\n"),
        ("Exclamation", "!"),
        ("The", "The"),
    )
    swap_threshold = mo.ui.slider(
        0.1,
        0.8,
        value=0.3,
        step=0.05,
        show_value=True,
        label="Strong-sink head threshold",
    )
    run_swap = mo.ui.run_button(
        label="Swap six first tokens in one GPU batch",
        kind="success",
        full_width=True,
    )
    mo.vstack(
        [
            mo.callout(
                "The paper's packing ablations suggest the first position can matter independently "
                "of BOS identity, except when training fixes BOS as the permanent first token. We "
                "test a new prompt-local question: which one-token replacements keep the sink while "
                "disturbing the next-token distribution as little as possible?",
                kind="info",
            ),
            swap_threshold,
            run_swap,
            mo.md(
                '<p class="margin-note">This is our extension: six variants, identical prompt, '
                "one accelerator forward.</p>"
            ),
        ],
        gap=1,
    )
    return run_swap, swap_candidates, swap_threshold


@app.cell
def _(mo):
    mo.Html(
        """
        <div class="reference-result">
          <div class="eyebrow" style="color:#86c8ba">Local preview snapshot · SmolLM2 135M</div>
          <div style="font:1.25rem/1.4 Georgia,serif; margin-top:.45rem">
            On the default prompt, newline behaved like a surprisingly good first-token substitute:
            it kept nearly all BOS sink attention while barely moving the next-token distribution.
          </div>
          <div class="reference-grid">
            <div><div class="reference-value">≈100%</div><small>BOS mean-sink attention preserved by newline</small></div>
            <div><div class="reference-value">0.0017</div><small>Jensen–Shannon drift from native BOS</small></div>
            <div><div class="reference-value">6</div><small>first-token variants evaluated in one accelerator forward</small></div>
          </div>
        </div>
        """
    )
    return


@app.cell
def _(mo, probe_model, probe_prompt, run_sink_swap, run_swap, swap_candidates):
    mo.stop(
        not run_swap.value,
        mo.md(
            '<p class="margin-note">Run the sink swap after choosing a model and prompt above.</p>'
        ),
    )
    try:
        sink_swap_result = run_sink_swap(
            probe_model.value,
            probe_prompt.value,
            swap_candidates,
        )
    except Exception as swap_error:
        swap_error_text = str(swap_error)
        swap_oom_hint = (
            " If this is a GPU memory issue, switch to the 135M preview model or shorten the prompt; "
            "the experiment still runs as a single six-row batch."
            if "memory" in swap_error_text.lower() or "oom" in swap_error_text.lower()
            else ""
        )
        mo.stop(True, mo.callout(f"Sink swap failed: {swap_error}{swap_oom_hint}", kind="danger"))
    return (sink_swap_result,)


@app.cell
def _(go, make_subplots, np, sink_swap_result, swap_threshold):
    swap_fractions = sink_swap_result.sink_fractions(swap_threshold.value)
    swap_layer_profiles = sink_swap_result.sink_profiles.mean(axis=2)
    swap_color_max = max(0.3, float(np.max(swap_layer_profiles)))
    swap_figure = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.46, 0.54],
        vertical_spacing=0.16,
        subplot_titles=(
            "Preserve the sink without moving the prediction",
            "Where each replacement attracts attention",
        ),
    )
    swap_figure.add_trace(
        go.Scatter(
            x=sink_swap_result.mean_sink_scores,
            y=sink_swap_result.js_divergences,
            mode="markers+text",
            text=sink_swap_result.candidate_names,
            textposition=[
                "bottom left",
                "top center",
                "top center",
                "top right",
                "top center",
                "middle left",
            ],
            marker={
                "size": 18 + 28 * swap_fractions,
                "color": swap_fractions,
                "cmin": 0,
                "cmax": 1,
                "colorscale": [[0, "#efbd4d"], [1, "#ed6a5a"]],
                "line": {"color": "#24343d", "width": 1.5},
                "showscale": False,
            },
            customdata=np.column_stack(
                [
                    sink_swap_result.candidate_tokens,
                    sink_swap_result.top_predictions,
                    swap_fractions,
                ]
            ),
            hovertemplate=(
                "%{text} · token %{customdata[0]}<br>mean sink %{x:.3f}"
                "<br>prediction drift %{y:.4f}<br>strong heads %{customdata[2]:.0%}"
                "<br>top next token %{customdata[1]}<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )
    swap_figure.add_trace(
        go.Heatmap(
            z=swap_layer_profiles,
            x=list(range(sink_swap_result.layers)),
            y=list(sink_swap_result.candidate_names),
            colorscale=[[0, "#f9f4e9"], [0.45, "#efbd4d"], [1, "#ed6a5a"]],
            zmin=0,
            zmax=swap_color_max,
            colorbar={
                "title": {"text": "attention<br>to token 0", "side": "top"},
                "thickness": 10,
                "len": 0.42,
                "y": 0.21,
            },
            hovertemplate="%{y}<br>layer %{x}<br>mean attention %{z:.3f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    swap_figure.update_xaxes(title_text="mean attention to token 0", row=1, col=1)
    swap_figure.update_yaxes(title_text="JS drift from BOS", rangemode="tozero", row=1, col=1)
    swap_figure.update_xaxes(title_text="layer", row=2, col=1)
    swap_figure.update_yaxes(title_text="replacement", autorange="reversed", row=2, col=1)
    swap_figure.update_layout(
        height=760,
        showlegend=False,
        margin={"l": 82, "r": 52, "t": 68, "b": 52},
        paper_bgcolor="#fbf7ee",
        plot_bgcolor="#fbf7ee",
        font={"family": "Avenir Next, sans-serif", "color": "#24343d"},
    )
    swap_figure.update_annotations(font={"size": 16})
    None
    return swap_figure, swap_fractions


@app.cell
def _(mo, np, sink_swap_result, swap_figure, swap_fractions):
    baseline_score = float(sink_swap_result.mean_sink_scores[0])
    eligible = np.flatnonzero(
        sink_swap_result.mean_sink_scores[1:] >= 0.8 * max(baseline_score, 1e-9)
    ) + 1
    if not len(eligible):
        eligible = np.arange(1, sink_swap_result.batch_size)
    best_swap = int(eligible[np.argmin(sink_swap_result.js_divergences[eligible])])
    preserved = 100 * float(
        sink_swap_result.mean_sink_scores[best_swap] / max(baseline_score, 1e-9)
    )
    device_kind = "success" if sink_swap_result.device != "cpu" else "warn"
    mo.vstack(
        [
            mo.hstack(
                [
                    mo.stat(sink_swap_result.device.upper(), "accelerator backend"),
                    mo.stat(
                        f"{sink_swap_result.batch_size} variants · 1 forward",
                        "accelerator batch",
                    ),
                    mo.stat(
                        f"{sink_swap_result.elapsed_seconds * 1000:.0f} ms",
                        "synchronized inference",
                    ),
                ],
                widths="equal",
                gap=1,
            ),
            mo.ui.plotly(swap_figure),
            mo.callout(
                f"**Prompt-local finding on this model:** {sink_swap_result.candidate_names[best_swap]} "
                f"preserves {preserved:.0f}% of BOS's mean sink attention while moving the "
                f"next-token distribution by only "
                f"JS={sink_swap_result.js_divergences[best_swap]:.4f}. Its top prediction is "
                f"`{sink_swap_result.top_predictions[best_swap]}`. Bubble size reflects the "
                f"{swap_fractions[best_swap]:.0%} of heads above the chosen threshold.",
                kind=device_kind,
            ),
            mo.md(
                "The ideal substitute sits **far right and near the floor**: it attracts "
                "attention like BOS without changing the model's behavior much. This turns the "
                "paper's position-vs-identity question into an editable inference-time experiment."
            ),
        ],
        gap=1.1,
    )
    return


@app.cell
def _(mo):
    mo.Html(
        """
        <div class="section-rule"><span class="number">06</span><div>
          <div class="eyebrow">Test causality, not correlation</div>
          <h2>Pull the plug on one head.</h2>
          <p class="citation-note">
            Citation note: <a href="https://arxiv.org/abs/2504.02732" target="_blank">Barbero et al. (2025)</a>
            interpret some sink heads as approximate no-op routes. The head-gate here is our local
            causal check: does removing a selected sink head perturb the prediction less than removing
            a contrast head?
          </p>
        </div></div>
        """
    )
    return


@app.cell
def _(mo):
    run_gate = mo.ui.run_button(
        label="Gate baseline, sink head, and contrast head in one batch",
        kind="warn",
        full_width=True,
    )
    mo.vstack(
        [
            mo.callout(
                "Run the live probe in section 02 first. This experiment takes the layer and head "
                "you selected there, finds the most different head in the same layer, and zeros "
                "each head's contribution immediately before the output projection.",
                kind="info",
            ),
            run_gate,
            mo.md(
                '<p class="margin-note">Three identical prompts · untouched baseline · selected '
                "head gated · contrast head gated.</p>"
            ),
        ],
        gap=1,
    )
    return (run_gate,)


@app.cell
def _(mo):
    mo.Html(
        """
        <div class="reference-result">
          <div class="eyebrow" style="color:#86c8ba">Local preview snapshot · causal check</div>
          <div style="font:1.25rem/1.4 Georgia,serif; margin-top:.45rem">
            On the default prompt, gating the selected sink-heavy head disturbed the prediction less
            than gating a contrast head in the same layer. This supports lower functional impact on
            that prompt, not a universal claim that every sink head is harmless.
          </div>
          <div class="reference-grid">
            <div><div class="reference-value">97%</div><small>selected head attention to token zero</small></div>
            <div><div class="reference-value">3.8×</div><small>less prediction drift than the contrast-head gate</small></div>
            <div><div class="reference-value">3 rows</div><small>baseline, sink-head gate, contrast-head gate in one batch</small></div>
          </div>
        </div>
        """
    )
    return


@app.cell
def _(
    measured_probe,
    mo,
    np,
    probe_model,
    probe_prompt,
    run_gate,
    run_head_gate,
    selected_head,
    selected_layer,
):
    mo.stop(
        not run_gate.value,
        mo.md(
            '<p class="margin-note">Choose a measured layer and head above, then run the gate.</p>'
        ),
    )
    gate_layer = selected_layer.value
    gate_head = selected_head.value
    layer_sink_scores = measured_probe.sink_profile[gate_layer]
    if layer_sink_scores[gate_head] >= float(np.median(layer_sink_scores)):
        contrast_head = int(np.argmin(layer_sink_scores))
    else:
        contrast_head = int(np.argmax(layer_sink_scores))
    try:
        head_gate_result = run_head_gate(
            probe_model.value,
            probe_prompt.value,
            layer=gate_layer,
            sink_head=gate_head,
            comparison_head=contrast_head,
            sink_score=float(layer_sink_scores[gate_head]),
            comparison_score=float(layer_sink_scores[contrast_head]),
        )
    except Exception as gate_error:
        gate_error_text = str(gate_error)
        gate_oom_hint = (
            " If this is a GPU memory issue, rerun with the 135M preview model or a shorter prompt; "
            "the gate keeps three rows resident at once."
            if "memory" in gate_error_text.lower() or "oom" in gate_error_text.lower()
            else ""
        )
        mo.stop(True, mo.callout(f"Head gate failed: {gate_error}{gate_oom_hint}", kind="danger"))
    return (head_gate_result,)


@app.cell
def _(go, head_gate_result, make_subplots):
    gate_labels = (
        f"selected H{head_gate_result.sink_head}",
        f"contrast H{head_gate_result.comparison_head}",
    )
    gate_colors = ("#ed6a5a", "#3d8b7d")
    gate_js_max = max(float(head_gate_result.js_divergences[1:].max()), 1e-9)
    gate_hidden_max = max(float(head_gate_result.final_hidden_l2[1:].max()), 1e-9)
    _gate_contribution_max = max(
        float(head_gate_result.projected_contribution_l2[1:].max()), 1e-9
    )
    gate_figure = make_subplots(
        rows=3,
        cols=1,
        row_heights=[0.34, 0.33, 0.33],
        vertical_spacing=0.14,
        subplot_titles=(
            "Prediction drift after gating",
            "Final-token representation change",
            "Baseline projected head contribution",
        ),
    )
    gate_figure.add_trace(
        go.Bar(
            x=list(gate_labels),
            y=head_gate_result.js_divergences[1:],
            marker_color=list(gate_colors),
            text=[f"{value:.5f}" for value in head_gate_result.js_divergences[1:]],
            textposition="outside",
            cliponaxis=False,
        ),
        row=1,
        col=1,
    )
    gate_figure.add_trace(
        go.Bar(
            x=list(gate_labels),
            y=head_gate_result.final_hidden_l2[1:],
            marker_color=list(gate_colors),
            text=[f"{value:.3f}" for value in head_gate_result.final_hidden_l2[1:]],
            textposition="outside",
            cliponaxis=False,
        ),
        row=2,
        col=1,
    )
    gate_figure.add_trace(
        go.Bar(
            x=list(gate_labels),
            y=head_gate_result.projected_contribution_l2[1:],
            marker_color=list(gate_colors),
            text=[f"{value:.3f}" for value in head_gate_result.projected_contribution_l2[1:]],
            textposition="outside",
            cliponaxis=False,
        ),
        row=3,
        col=1,
    )
    gate_figure.update_yaxes(
        title_text="JS divergence", range=[0, gate_js_max * 1.24], row=1, col=1
    )
    gate_figure.update_yaxes(
        title_text="hidden-state L2", range=[0, gate_hidden_max * 1.24], row=2, col=1
    )
    gate_figure.update_yaxes(
        title_text="projected L2",
        range=[0, _gate_contribution_max * 1.24],
        row=3,
        col=1,
    )
    gate_figure.update_layout(
        height=820,
        showlegend=False,
        margin={"l": 72, "r": 36, "t": 70, "b": 52},
        paper_bgcolor="#fbf7ee",
        plot_bgcolor="#fbf7ee",
        font={"family": "Avenir Next, sans-serif", "color": "#24343d"},
    )
    gate_figure.update_annotations(font={"size": 16})
    None
    return (gate_figure,)


@app.cell
def _(gate_figure, head_gate_result, mo):
    selected_drift = float(head_gate_result.js_divergences[1])
    contrast_drift = float(head_gate_result.js_divergences[2])
    selected_hidden = float(head_gate_result.final_hidden_l2[1])
    contrast_hidden = float(head_gate_result.final_hidden_l2[2])
    _selected_contribution = float(head_gate_result.projected_contribution_l2[1])
    _contrast_contribution = float(head_gate_result.projected_contribution_l2[2])
    if selected_drift < contrast_drift:
        drift_ratio = contrast_drift / max(selected_drift, 1e-12)
        causal_message = (
            f"Gating the selected high-sink head changes the next-token distribution "
            f"{drift_ratio:.1f}× less than gating the contrast head. This is local intervention "
            "evidence that the selected head has lower final-token impact than the contrast head "
            "on this prompt."
        )
        causal_kind = "success"
    else:
        drift_ratio = selected_drift / max(contrast_drift, 1e-12)
        causal_message = (
            f"Gating the selected head changes the prediction {drift_ratio:.1f}× more than the "
            "contrast head. This prompt challenges the simple claim that every strong sink is quiet."
        )
        causal_kind = "warn"
    mo.vstack(
        [
            mo.hstack(
                [
                    mo.stat(
                        f"{head_gate_result.sink_score:.0%}",
                        f"selected H{head_gate_result.sink_head} sink score",
                    ),
                    mo.stat(
                        f"{head_gate_result.comparison_score:.0%}",
                        f"contrast H{head_gate_result.comparison_head} sink score",
                    ),
                    mo.stat(
                        f"{head_gate_result.elapsed_seconds * 1000:.0f} ms",
                        f"{head_gate_result.device.upper()} · 3-row forward",
                    ),
                    mo.stat(
                        f"{_selected_contribution / max(_contrast_contribution, 1e-9):.2f}×",
                        "selected / contrast projected L2",
                    ),
                ],
                widths="equal",
                gap=1,
            ),
            mo.ui.plotly(gate_figure),
            mo.callout(causal_message, kind=causal_kind),
            mo.md(
                f"The selected gate changes final-token hidden state by **{selected_hidden:.3f}** "
                f"versus **{contrast_hidden:.3f}** for the contrast gate. Its baseline projected "
                f"head contribution is **{_selected_contribution:.3f}** versus "
                f"**{_contrast_contribution:.3f}** for the contrast head. This intervention zeros "
                "one head before `o_proj`; it does not retrain or edit model weights."
            ),
        ],
        gap=1.1,
    )
    return


@app.cell
def _(mo):
    mo.Html(
        """
        <div class="section-rule"><span class="number">07</span><div>
          <div class="eyebrow">Test the paper's prediction</div>
          <h2>At scale, the paper sees more strong sinks.</h2>
          <p class="citation-note">
            Citation note: this panel quotes the scale/context trends from
            <a href="https://arxiv.org/abs/2504.02732" target="_blank">Barbero et al. (2025)</a>.
            It is paper evidence, not a live reproduction on this laptop.
          </p>
        </div></div>
        """
    )
    return


@app.cell
def _(go):
    scale_figure = go.Figure(
        data=go.Scatter(
            x=[8, 70, 405],
            y=[45.97, 73.49, 78.29],
            mode="lines+markers+text",
            line={"color": "#24343d", "width": 3},
            marker={
                "color": ["#efbd4d", "#ed6a5a", "#3d8b7d"],
                "size": [22, 38, 60],
                "line": {"color": "#fbf7ee", "width": 2},
            },
            text=["1,024 heads", "5,120 heads", "16,128 heads"],
            textposition=["bottom right", "top center", "bottom left"],
            customdata=[[32, 32], [80, 64], [126, 128]],
            hovertemplate=(
                "LLaMA 3.1 %{x}B<br>%{customdata[0]} layers · %{customdata[1]} heads/layer"
                "<br>strong-sink heads %{y:.2f}%<extra></extra>"
            ),
        )
    )
    scale_figure.update_xaxes(type="log", title="model parameters (billions)")
    scale_figure.update_yaxes(title="heads forming strong sinks", ticksuffix="%", range=[35, 88])
    scale_figure.update_layout(
        title="LLaMA 3.1 family · reported by the paper",
        height=430,
        margin={"l": 62, "r": 38, "t": 72, "b": 58},
        paper_bgcolor="#fbf7ee",
        plot_bgcolor="#fbf7ee",
        font={"family": "Avenir Next, sans-serif", "color": "#24343d"},
    )
    None
    return (scale_figure,)


@app.cell
def _(mo, scale_figure):
    mo.hstack(
        [
            mo.ui.plotly(scale_figure),
            mo.vstack(
                [
                    mo.md(
                        """
                        ### Two paper patterns, both directionally supported

                        **More depth → more strong-sink heads.** Deep models accumulate more mixing,
                        so the paper argues more heads benefit from a low-impact attention target.

                        **Longer training context → stronger sinks.** In the paper's controlled
                        120M-model runs, every setup processed 5B tokens; models trained on longer
                        contexts developed substantially more sinks, while very short-context models
                        developed almost none.
                        """
                    ),
                    mo.callout(
                        "Table 1 uses a strict 0.80 head threshold over 170 prompts, taking the "
                        "first 64 tokens. The context-length result stays qualitative here because "
                        "its exact points are only shown in Figure 5.",
                        kind="neutral",
                    ),
                ],
                gap=1.2,
            ),
        ],
        widths=[1.55, 1],
        align="center",
        gap=2,
    )
    return


@app.cell
def _(mo):
    mo.Html(
        """
        <div class="section-rule"><span class="number">08</span><div>
          <div class="eyebrow">Ask what is actually special</div>
          <h2>Position usually wins. Fixed-BOS training can flip the answer.</h2>
          <p class="citation-note">
            Citation note: the BOS/data-packing comparison comes from
            <a href="https://arxiv.org/abs/2504.02732" target="_blank">Barbero et al. (2025)</a>,
            while <a href="https://arxiv.org/abs/2410.10781" target="_blank">Gu et al. (2025)</a>
            emphasize that sink emergence depends on training setup, data distribution, and loss.
          </p>
        </div></div>
        """
    )
    return


@app.cell
def _(mo):
    packing_results = {
        "causal_shared": {
            "label": "Causal · shared BOS/EOS token",
            "present_sink": 65.10,
            "removed_sink": 65.15,
            "present_loss": 2.69,
            "removed_loss": 2.70,
            "fixed": False,
        },
        "causal_fixed": {
            "label": "Causal · permanently fixed BOS",
            "present_sink": 90.84,
            "removed_sink": 0.05,
            "present_loss": 2.69,
            "removed_loss": 7.56,
            "fixed": True,
        },
        "intradoc_tokens": {
            "label": "Intra-document · BOS + EOS",
            "present_sink": 83.33,
            "removed_sink": 50.24,
            "present_loss": 2.67,
            "removed_loss": 2.68,
            "fixed": False,
        },
        "intradoc_fixed": {
            "label": "Intra-document · permanently fixed BOS",
            "present_sink": 90.56,
            "removed_sink": 0.00,
            "present_loss": 2.67,
            "removed_loss": 7.78,
            "fixed": True,
        },
    }
    packing_regime = mo.ui.dropdown(
        {result["label"]: key for key, result in packing_results.items()},
        value="Causal · permanently fixed BOS",
        label="Training and packing regime",
        full_width=True,
    )
    packing_panel = mo.vstack(
        [
            mo.md(
                "The authors trained ~120M-parameter models for 30B tokens, then changed whether "
                "BOS was available at inference. Pick a regime to inspect the dependency."
            ),
            packing_regime,
        ],
        gap=1,
    ).style(
        {
            "background": "#eadfc988",
            "border": "1px solid #24343d24",
            "padding": "1.25rem",
            "border-radius": ".4rem 1.2rem 1.2rem 1.2rem",
        }
    )
    packing_panel
    return packing_regime, packing_results


@app.cell
def _(go, make_subplots, packing_regime, packing_results):
    chosen_packing = packing_results[packing_regime.value]
    packing_figure = make_subplots(specs=[[{"secondary_y": True}]])
    packing_figure.add_trace(
        go.Bar(
            x=["BOS available", "BOS removed"],
            y=[chosen_packing["present_sink"], chosen_packing["removed_sink"]],
            name="sink metric",
            marker_color=["#3d8b7d", "#ed6a5a"],
            text=[
                f"{chosen_packing['present_sink']:.2f}%",
                f"{chosen_packing['removed_sink']:.2f}%",
            ],
            textposition="outside",
        ),
        secondary_y=False,
    )
    packing_figure.add_trace(
        go.Scatter(
            x=["BOS available", "BOS removed"],
            y=[chosen_packing["present_loss"], chosen_packing["removed_loss"]],
            name="validation loss",
            mode="lines+markers+text",
            line={"color": "#24343d", "width": 3},
            marker={"size": 11},
            text=[
                f"loss {chosen_packing['present_loss']:.2f}",
                f"loss {chosen_packing['removed_loss']:.2f}",
            ],
            textposition="top center",
        ),
        secondary_y=True,
    )
    packing_figure.update_yaxes(
        title_text="heads forming sinks", ticksuffix="%", range=[0, 108], secondary_y=False
    )
    packing_figure.update_yaxes(title_text="validation loss", secondary_y=True)
    packing_figure.update_layout(
        title=chosen_packing["label"],
        height=400,
        barmode="group",
        legend={"orientation": "h", "y": 1.16, "x": 0},
        margin={"l": 58, "r": 58, "t": 82, "b": 48},
        paper_bgcolor="#fbf7ee",
        plot_bgcolor="#fbf7ee",
        font={"family": "Avenir Next, sans-serif", "color": "#24343d"},
    )
    None
    return chosen_packing, packing_figure


@app.cell
def _(chosen_packing, mo, packing_figure):
    sink_drop = chosen_packing["present_sink"] - chosen_packing["removed_sink"]
    loss_jump = chosen_packing["removed_loss"] - chosen_packing["present_loss"]
    interpretation = (
        "This model learned to depend on the permanently fixed BOS. Removing it destroys the sink "
        "and predictive performance together."
        if chosen_packing["fixed"]
        else "This model was not forced to use one permanent BOS. A sink survives on the first "
        "available token and validation loss barely moves."
    )
    mo.vstack(
        [
            mo.ui.plotly(packing_figure),
            mo.hstack(
                [
                    mo.stat(f"{sink_drop:.2f} pp", "sink reduction after removal"),
                    mo.stat(f"{loss_jump:+.2f}", "validation-loss change"),
                ],
                widths="equal",
                gap=1,
            ),
            mo.callout(interpretation, kind="warn" if chosen_packing["fixed"] else "success"),
            mo.md(
                "**Takeaway:** when BOS is fixed at position zero during training, the model uses "
                "that token as its sink. Otherwise, it still forms a weaker sink on whichever "
                "token occupies the first position. Position is often the stable affordance, but "
                "fixed-BOS training can make token identity matter."
            ),
        ],
        gap=1.1,
    )
    return


@app.cell
def _(mo):
    mo.Html(
        """
        <div class="section-rule"><span class="number">★</span><div>
          <div class="eyebrow">Evidence ledger</div>
          <h2>What each experiment earns—and what it does not.</h2>
        </div></div>
        <table class="evidence-table">
          <thead>
            <tr>
              <th>Question</th>
              <th>Measurement</th>
              <th>Verdict</th>
              <th>Caveat</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Can attention sinks reduce information mixing?</td>
              <td>Toy recurrence with controllable sink strength and sink value norm.</td>
              <td>Illustrates the paper's mechanism before heavy model runs.</td>
              <td>Not a trained-transformer Jacobian or pre-training reproduction.</td>
            </tr>
            <tr>
              <td>Does an off-the-shelf LM show prompt-local first-token sinks?</td>
              <td>Layer × head attention to token zero plus BOS value-norm ratio.</td>
              <td>Default SmolLM2 preview is consistent with the no-op interpretation.</td>
              <td>Local ε=0.30 threshold is exploratory; paper Table 1 uses ε=0.80.</td>
            </tr>
            <tr>
              <td>Is the result robust beyond one cute prompt?</td>
              <td>5–40 prompts batched as one accelerator forward with genre bootstrap intervals.</td>
              <td>Shows whether sink-heavy layers survive genre changes.</td>
              <td>Still much smaller than the paper's 170-prompt fixed-window evaluation.</td>
            </tr>
            <tr>
              <td>Can another first token impersonate BOS?</td>
              <td>Six one-token replacements compared by mean sink attention and JS prediction drift.</td>
              <td>A prompt-local inference-time extension of the paper's position-vs-identity question.</td>
              <td>Single-prompt result; should be repeated across the suite for a stronger claim.</td>
            </tr>
            <tr>
              <td>Does a selected sink head matter less functionally?</td>
              <td>One 3-row batch gates baseline, selected head, and contrast head before <code>o_proj</code>.</td>
              <td>Tests functional impact instead of only attention correlation.</td>
              <td>Whole-head gate, not a BOS-path-only intervention.</td>
            </tr>
          </tbody>
        </table>
        """
    )
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "What this prototype does—and does not—claim": mo.md(
                """
                The synthetic experiment isolates the proposed mechanism; it is not a reproduction
                of the paper's pre-training results. The real-model sink score is prompt-local, while
                the paper aggregates fixed-length windows over many prompts. The genre confidence
                intervals quantify variation within this notebook's smaller suite, not population-level
                uncertainty. A high score alone does not prove that a head implements a no-op.
                """
            ),
            "What we add beyond the paper": mo.md(
                """
                **Sink swap** is an inference-time extension: replace BOS with six one-token
                candidates in an otherwise identical prompt, then compare attention captured at
                position zero with Jensen–Shannon drift in the next-token distribution. It asks not
                only whether identity matters, but which replacement best preserves behavior.

                **Head gate** is a causal extension: zero the selected head's contribution just
                before its output projection and compare it against a contrast head in the same
                layer. This tests whether high sink attention actually corresponds to low functional
                impact instead of merely correlating with it.
                """
            ),
            "How the GPU is used": mo.md(
                """
                The prompt suite performs one padded forward for up to 40 prompts. Sink swap performs
                one forward for six first-token variants. Input tensors and masks move to CUDA/MPS
                before synchronized timing; attention summaries move to CPU only after inference.
                Head gate runs baseline, selected-head gate, and contrast-head gate as one three-row
                accelerator batch.
                CUDA models load in BF16 when supported, otherwise FP16. The accelerator and measured
                runtime are reported directly in the notebook.
                """
            ),
            "Source papers and related work": mo.md(
                """
                - [Barbero et al., *Why do LLMs attend to the first token?*](https://www.alphaxiv.org/abs/2504.02732)
                  is the main paper this notebook explains.
                - [Xiao et al., *Efficient Streaming Language Models with Attention Sinks*](https://arxiv.org/abs/2309.17453)
                  introduced the attention-sink framing in the StreamingLLM setting.
                - [Gu et al., *When Attention Sink Emerges in Language Models*](https://arxiv.org/abs/2410.10781)
                  studies when sinks emerge during pre-training and motivates sink-rate style measurements.
                """
            ),
        },
        multiple=True,
    )
    return


if __name__ == "__main__":
    app.run()
