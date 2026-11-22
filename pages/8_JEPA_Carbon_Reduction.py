"""
Chapter 08 — JEPA for Carbon-Efficient Multimodal LLM Systems

Joint-Embedding Predictive Architecture (JEPA) as a representation-learning
front-end for multimodal LLM pipelines: theoretical foundations, proposed
architecture, hypothetical energy modelling, and experimental protocol.

NOTE: JEPA's potential energy advantage remains a *testable hypothesis*,
not an established empirical result. This chapter presents the research
framework, distinguishes proven concepts from open questions, and provides
a simulation tool for exploring the hypothesis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.constants import (
    GRID_CARBON_INTENSITY_KG_PER_KWH,
    CO2_KG_PER_CAR_MILE,
    CO2_KG_PER_TREE_PER_YEAR,
    JEPA_PAPERS,
)
from utils.theme import EDITORIAL_CSS, chapter, divider, finding, quote, metric_card, formula_box

st.set_page_config(
    page_title="JEPA Carbon Reduction",
    layout="wide",
)

# ── Apply Editorial Theme ─────────────────────────────────────────────────────
st.markdown(EDITORIAL_CSS, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(
    chapter(8, "JEPA FOR CARBON-EFFICIENT MULTIMODAL LLM SYSTEMS",
            "Joint-Embedding Predictive Architecture as a perception front-end"),
    unsafe_allow_html=True,
)

st.markdown(
    "<h1>Reducing Carbon Emissions in LLM Systems <em>Using JEPA</em></h1>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <p class="hero-subtitle">
    Joint-Embedding Predictive Architecture (JEPA) learns to predict representations
    in a latent space rather than reconstructing every low-level detail of an input.
    This chapter investigates how JEPA could be integrated into multimodal LLM pipelines
    to reduce redundant visual processing, compress perception-stage computation, and
    lower operational carbon emissions.
    </p>
    """,
    unsafe_allow_html=True,
)

# ── Epistemic status badge ────────────────────────────────────────────────────
st.markdown(
    """
    <div style="display: inline-flex; align-items: center; gap: 8px; padding: 8px 20px;
                border-radius: 100px; border: 1px solid var(--accent-red);
                font-family: var(--mono); font-size: 12px; color: var(--accent-red);
                margin-bottom: 2rem; background: rgba(185,28,28,0.05);">
        <span style="width: 8px; height: 8px; border-radius: 50%;
                     background: var(--accent-red); display: inline-block;"></span>
        RESEARCH HYPOTHESIS &mdash; NOT AN ESTABLISHED EMPIRICAL RESULT
    </div>
    """,
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — WHY LLMs PRODUCE CARBON EMISSIONS
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(divider(), unsafe_allow_html=True)

st.markdown(
    "<div class='chapter-marker'>SECTION 1 — FOUNDATIONAL CONTEXT</div>"
    "<h3>Why LLM Systems Produce Carbon Emissions</h3>",
    unsafe_allow_html=True,
)

s1_col1, s1_col2 = st.columns(2)

with s1_col1:
    st.markdown(
        """
        LLM-related electricity consumption occurs across two distinct operational phases:

        **Training** — Processing large corpora and updating billions of parameters via
        backpropagation, typically requiring thousands of GPU-hours on accelerators such as
        NVIDIA A100 or H100.

        **Inference** — Processing user prompts and autoregressively generating responses.
        For commercially deployed models with widespread adoption, inference constitutes
        80% to 90% of full lifetime energy consumption.

        The resulting greenhouse gas emissions depend on more than model scale alone.
        Relevant factors include hardware efficiency, accelerator utilization rate,
        data-centre cooling overhead (PUE), and the carbon intensity of the regional
        electricity supply.
        """
    )

with s1_col2:
    st.markdown(
        formula_box(
            "Operational Carbon Accounting",
            r"\text{Operational CO}_{2}\text{e} = E_{\text{consumed}} \;(\text{kWh}) \;\times\; I_{\text{grid}} \;(\text{kgCO}_{2}\text{e/kWh})",
            "The simplified operational emissions estimate used throughout this dashboard. "
            "Reducing electricity consumption reduces operational emissions when grid carbon "
            "intensity is held constant.",
            tag="CORE FORMULA",
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        formula_box(
            "Time-Varying Carbon Intensity",
            r"\text{CO}_{2}\text{e} = \sum_{t} E_t \times I_t",
            "For time-varying electricity supply with dynamic carbon intensity, emissions "
            "are summed across discrete time intervals. Lifecycle emissions and embodied hardware "
            "emissions are separate accounting categories.",
            tag="DYNAMIC VARIANT",
        ),
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — WHAT IS JEPA?
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(divider(), unsafe_allow_html=True)

st.markdown(
    "<div class='chapter-marker'>SECTION 2 — ARCHITECTURAL PRIMER</div>"
    "<h3>What is Joint-Embedding Predictive Architecture?</h3>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    **JEPA** (Joint-Embedding Predictive Architecture) is a family of learning architectures
    associated with Yann LeCun's proposal for learning predictive representations of the world.
    Unlike conventional reconstruction-based objectives where a model is trained to reproduce
    missing input details (such as pixels), a JEPA-style objective trains the model to predict
    a *representation* (an embedding) of a target region from a representation of the available context.
    """
)

# ── The Three JEPA Components ─────────────────────────────────────────────────
st.markdown(
    "<div class='chapter-marker'>CORE COMPONENTS</div>",
    unsafe_allow_html=True,
)

jepa_c1, jepa_c2, jepa_c3 = st.columns(3)

with jepa_c1:
    st.markdown(
        """
        <div class="compare-panel" style="min-height: 200px;">
            <div class="compare-tag modern">COMPONENT A</div>
            <h4 style="font-family: var(--serif); font-size: 1.15rem; color: var(--ink); margin: 0 0 0.5rem 0;">
                Context Encoder
            </h4>
            <p style="font-size: 0.9rem; color: var(--graphite); line-height: 1.6; margin: 0;">
                Encodes the visible or available context regions of the input
                (e.g., unmasked image patches, selected video frames) into a
                latent representation vector. Typically implemented as a
                Vision Transformer (ViT).
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with jepa_c2:
    st.markdown(
        """
        <div class="compare-panel" style="min-height: 200px;">
            <div class="compare-tag modern">COMPONENT B</div>
            <h4 style="font-family: var(--serif); font-size: 1.15rem; color: var(--ink); margin: 0 0 0.5rem 0;">
                Target Encoder
            </h4>
            <p style="font-size: 0.9rem; color: var(--graphite); line-height: 1.6; margin: 0;">
                Produces target representations for selected (masked) regions
                of the input. Updated via exponential moving average (EMA) of
                the context encoder parameters rather than gradient descent,
                providing stable learning targets.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with jepa_c3:
    st.markdown(
        """
        <div class="compare-panel" style="min-height: 200px;">
            <div class="compare-tag modern">COMPONENT C</div>
            <h4 style="font-family: var(--serif); font-size: 1.15rem; color: var(--ink); margin: 0 0 0.5rem 0;">
                Predictor
            </h4>
            <p style="font-size: 0.9rem; color: var(--graphite); line-height: 1.6; margin: 0;">
                Predicts the target representation from the context representation.
                The prediction is compared against the target encoder's output
                in latent space rather than in pixel space, avoiding costly
                reconstruction computations.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Core Equation ─────────────────────────────────────────────────────────────
st.markdown(
    formula_box(
        "JEPA Prediction Objective",
        r"\hat{y} = P\bigl(E_c(x_c)\bigr)",
        "Where E_c is the context encoder, x_c is the context input, P is the predictor, "
        "and y&#x0302; is the predicted target representation. During training, the prediction is compared "
        "with E_t(x_t) (the target encoder's representation of the masked region) using a distance "
        "loss in embedding space &mdash; never requiring pixel-level reconstruction.",
        tag="LATENT PREDICTION",
    ),
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — HOW JEPA COULD HELP
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(divider(), unsafe_allow_html=True)

st.markdown(
    "<div class='chapter-marker'>SECTION 3 — EFFICIENCY HYPOTHESES</div>"
    "<h3>How JEPA Could Reduce Energy Use in LLM Systems</h3>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <p style="color: var(--graphite); line-height: 1.7; font-size: 0.95rem;">
    The strongest near-term research opportunity is <strong>not</strong> to replace
    a text-generating LLM with JEPA. Rather, it is to investigate JEPA as a
    representation-learning or perception component in a <strong>multimodal LLM pipeline</strong>,
    compressing visual or temporal information before it reaches the language model.
    </p>
    """,
    unsafe_allow_html=True,
)

hyp_c1, hyp_c2 = st.columns(2)

with hyp_c1:
    st.markdown(
        """
        <div class="compare-panel" style="margin-bottom: 1rem;">
            <div class="compare-tag legacy">HYPOTHESIS A</div>
            <h4 style="font-family: var(--serif); font-size: 1.1rem; color: var(--ink); margin: 0 0 0.5rem 0;">
                Avoiding Unnecessary Low-Level Reconstruction
            </h4>
            <p style="font-size: 0.9rem; color: var(--graphite); line-height: 1.6;">
                A JEPA objective predicts latent representations rather than requiring
                pixel-level reconstruction as its learning target. This may avoid some
                reconstruction-related computation in a comparable representation-learning setup.
            </p>
            <p style="font-size: 0.85rem; color: var(--accent-red); line-height: 1.5; margin-top: 0.5rem; font-style: italic;">
                Caution: The target encoder, context encoder, predictor, data pipeline, and training
                schedule all consume resources. The objective alone does not establish a net energy saving.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="compare-panel" style="margin-bottom: 1rem;">
            <div class="compare-tag legacy">HYPOTHESIS C</div>
            <h4 style="font-family: var(--serif); font-size: 1.1rem; color: var(--ink); margin: 0 0 0.5rem 0;">
                Potential Inference Savings
            </h4>
            <p style="font-size: 0.9rem; color: var(--graphite); line-height: 1.6;">
                If the JEPA-based front end reduces the number of visual tokens, frames, or
                other inputs that the downstream model must process, it may reduce downstream
                computation and memory use during inference.
            </p>
            <p style="font-size: 0.85rem; color: var(--accent-red); line-height: 1.5; margin-top: 0.5rem; font-style: italic;">
                However: JEPA does not reduce autoregressive text token generation. The encoder
                itself consumes energy. Compression may reduce accuracy. End-to-end latency may
                increase if preprocessing overhead exceeds downstream savings.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with hyp_c2:
    st.markdown(
        """
        <div class="compare-panel" style="margin-bottom: 1rem;">
            <div class="compare-tag legacy">HYPOTHESIS B</div>
            <h4 style="font-family: var(--serif); font-size: 1.1rem; color: var(--ink); margin: 0 0 0.5rem 0;">
                Compressing Multimodal Information Before the LLM
            </h4>
            <p style="font-size: 0.9rem; color: var(--graphite); line-height: 1.6;">
                Video or image inputs can contain substantial redundant information.
                A JEPA-based encoder could produce compact semantic features, select informative
                segments, or support downstream retrieval before information is passed to the
                autoregressive language model.
            </p>
            <p style="font-size: 0.85rem; color: var(--accent-red); line-height: 1.5; margin-top: 0.5rem; font-style: italic;">
                This is a proposed architecture, not a result already proven to reduce carbon emissions.
                It must be compared with a conventional multimodal pipeline using the same tasks and
                quality requirements.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="compare-panel" style="margin-bottom: 1rem;">
            <div class="compare-tag modern">HYPOTHESIS D</div>
            <h4 style="font-family: var(--serif); font-size: 1.1rem; color: var(--ink); margin: 0 0 0.5rem 0;">
                Combining JEPA with Established Efficiency Techniques
            </h4>
            <p style="font-size: 0.9rem; color: var(--graphite); line-height: 1.6;">
                JEPA could be evaluated alongside quantization (INT8/INT4), knowledge distillation,
                sparse Mixture-of-Experts, context/token reduction, and dynamic model routing.
                These techniques are complementary research options; they should not be presented
                as inherent features of JEPA.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — PROPOSED JEPA–LLM ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(divider(), unsafe_allow_html=True)

st.markdown(
    "<div class='chapter-marker'>SECTION 4 — PROPOSED SYSTEM ARCHITECTURE</div>"
    "<h3>JEPA-Enhanced Multimodal LLM Pipeline</h3>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <p style="color: var(--graphite); line-height: 1.7; font-size: 0.95rem; margin-bottom: 1.5rem;">
        The core research question is whether inserting a JEPA perception stage
        before the language model reduces <strong>total system energy</strong>
        while preserving task performance. The following diagram illustrates the
        proposed five-stage pipeline.
    </p>
    """,
    unsafe_allow_html=True,
)

# ── Architecture Pipeline Diagram ─────────────────────────────────────────────
pipeline_stages = [
    ("01", "Multimodal Input", "Raw images, video frames, or other non-text modalities enter the system."),
    ("02", "JEPA Encoder", "Context encoder extracts latent semantic representations, discarding low-level redundancy."),
    ("03", "Selection and Compression", "Retains only task-relevant information, reducing the visual token count passed downstream."),
    ("04", "Compact Context Injection", "Compressed representations are projected into the LLM's embedding space via a learned adapter."),
    ("05", "Language Model Response", "The same autoregressive LLM generates text, but over a shorter, information-dense context window."),
]

pipe_cols = st.columns(5)
for i, (num, title, desc) in enumerate(pipeline_stages):
    with pipe_cols[i]:
        st.markdown(
            f"""
            <div style="border: 1px solid var(--divider); border-radius: 2px; padding: 1.2rem 0.8rem;
                        min-height: 220px; display: flex; flex-direction: column; justify-content: flex-start;
                        position: relative;">
                <div style="font-family: var(--serif); font-size: 2.5rem; font-weight: 300;
                            color: rgba(17,17,17,0.08); line-height: 1; margin-bottom: 0.5rem;">{num}</div>
                <div style="font-family: var(--sans); font-weight: 600; font-size: 0.9rem;
                            color: var(--ink); margin-bottom: 0.4rem;">{title}</div>
                <div style="font-family: var(--sans); font-size: 0.82rem; color: var(--graphite);
                            line-height: 1.5;">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown(
    """
    <div style="text-align: center; font-family: var(--mono); font-size: 11px; color: var(--graphite);
                letter-spacing: 0.05em; text-transform: uppercase; margin: 1rem 0 0.5rem 0;">
        Pipeline Flow: Input &rarr; JEPA Encoder &rarr; Selection / Compression &rarr; Compact Context &rarr; LLM Response
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    finding(
        "The research question is whether the added JEPA perception stage reduces <strong>total end-to-end system energy</strong> "
        "(including JEPA's own computational cost) while preserving downstream task quality. "
        "This cannot be established through architectural reasoning alone &mdash; it requires controlled empirical measurement.",
        label="OPEN RESEARCH QUESTION",
        warning=True,
    ),
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — COMPARATIVE BASELINE vs. JEPA ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(divider(), unsafe_allow_html=True)

st.markdown(
    "<div class='chapter-marker'>SECTION 5 — COMPARATIVE ANALYSIS</div>"
    "<h3>Baseline vs. JEPA Pipeline: Side-by-Side</h3>",
    unsafe_allow_html=True,
)

comp_col1, comp_col2 = st.columns(2)

with comp_col1:
    st.markdown(
        """
        <div class="compare-panel" style="border-left: 3px solid var(--accent-red);">
            <div class="compare-tag legacy">CONVENTIONAL BASELINE</div>
            <h4 style="font-family: var(--serif); font-size: 1.1rem; color: var(--ink); margin: 0 0 0.75rem 0;">
                Standard Multimodal Pipeline
            </h4>
            <div style="font-size: 0.88rem; color: var(--graphite); line-height: 1.7;">
                <p><strong>1. Visual Encoder:</strong> Full-resolution ViT or CLIP-style encoder processes
                all input frames/patches at native resolution.</p>
                <p><strong>2. Token Projection:</strong> All visual tokens (e.g., 256&ndash;576 per image)
                projected into LLM embedding space.</p>
                <p><strong>3. LLM Prefill:</strong> Full quadratic self-attention over combined
                text + visual tokens. For video: potentially thousands of visual tokens.</p>
                <p><strong>4. Autoregressive Decode:</strong> Standard token-by-token generation.</p>
            </div>
            <div style="border-top: 1px solid var(--divider); margin-top: 0.75rem; padding-top: 0.75rem;
                        font-family: var(--mono); font-size: 11px; color: var(--accent-red);">
                TOTAL ENERGY = E(encoder) + E(full_prefill) + E(decode)
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with comp_col2:
    st.markdown(
        """
        <div class="compare-panel" style="border-left: 3px solid var(--accent-green);">
            <div class="compare-tag modern">JEPA-ENHANCED PIPELINE</div>
            <h4 style="font-family: var(--serif); font-size: 1.1rem; color: var(--ink); margin: 0 0 0.75rem 0;">
                Latent-Compressed Multimodal Pipeline
            </h4>
            <div style="font-size: 0.88rem; color: var(--graphite); line-height: 1.7;">
                <p><strong>1. JEPA Encoder:</strong> Context encoder processes input, predicting
                latent representations rather than reconstructing raw pixels.</p>
                <p><strong>2. Semantic Selection:</strong> Only task-relevant representations retained;
                redundant frames/regions discarded in latent space.</p>
                <p><strong>3. LLM Prefill:</strong> Reduced visual token count &rarr; shorter
                prefill sequence &rarr; lower quadratic attention cost.</p>
                <p><strong>4. Autoregressive Decode:</strong> Identical decode phase (unchanged).</p>
            </div>
            <div style="border-top: 1px solid var(--divider); margin-top: 0.75rem; padding-top: 0.75rem;
                        font-family: var(--mono); font-size: 11px; color: var(--accent-green);">
                TOTAL ENERGY = E(jepa) + E(reduced_prefill) + E(decode)
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Energy comparison formulas ────────────────────────────────────────────────
st.markdown(
    formula_box(
        "Baseline Energy Budget",
        r"E_{\text{baseline}} = E_{\text{encoder}} + E_{\text{prefill}}(N_{\text{visual}}) + E_{\text{decode}}",
        "Total pipeline energy for a conventional multimodal system where N_visual is the full "
        "visual token count passed to the LLM's self-attention mechanism.",
        tag="BASELINE",
    ),
    unsafe_allow_html=True,
)
st.markdown(
    formula_box(
        "JEPA-Enhanced Energy Budget",
        r"E_{\text{JEPA}} = E_{\text{jepa\_enc}} + E_{\text{prefill}}(N_{\text{compressed}}) + E_{\text{decode}}",
        "If N_compressed &lt; N_visual, the quadratic attention cost during prefill is reduced. "
        "The net saving depends on whether this reduction exceeds the JEPA encoder overhead.",
        tag="JEPA SYSTEM",
    ),
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — HYPOTHETICAL ENERGY SIMULATOR
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(divider(), unsafe_allow_html=True)

st.markdown(
    "<div class='chapter-marker'>SECTION 6 — HYPOTHETICAL ENERGY MODELLING</div>"
    "<h3>Interactive JEPA Energy Reduction Simulator</h3>",
    unsafe_allow_html=True,
)

st.markdown(
    finding(
        "The numbers below are <strong>hypothetical illustrations</strong> for exploring the research hypothesis. "
        "They must not be cited as measured JEPA results. A controlled experiment following the protocol "
        "in Section 8 is required to produce defensible energy and emissions measurements.",
        label="IMPORTANT DISCLAIMER",
        warning=True,
    ),
    unsafe_allow_html=True,
)

sim_c1, sim_c2 = st.columns([1, 2])

with sim_c1:
    st.markdown("**Baseline Configuration:**")

    baseline_energy_per_1k = st.slider(
        "Baseline energy per 1,000 requests (kWh):",
        min_value=2.0,
        max_value=30.0,
        value=10.0,
        step=0.5,
        help="Total pipeline energy for a conventional multimodal system.",
    )

    jepa_visual_reduction_pct = st.slider(
        "JEPA visual token reduction (%):",
        min_value=10,
        max_value=80,
        value=40,
        step=5,
        help="Percentage of visual tokens eliminated by JEPA compression before the LLM.",
    )

    jepa_overhead_pct = st.slider(
        "JEPA encoder overhead (% of baseline):",
        min_value=2,
        max_value=25,
        value=8,
        step=1,
        help="Additional energy consumed by the JEPA encoder stage itself.",
    )

    visual_pct_of_total = st.slider(
        "Visual processing share of total energy (%):",
        min_value=10,
        max_value=70,
        value=35,
        step=5,
        help="Fraction of total pipeline energy attributable to visual token processing in the LLM.",
    )

    carbon_intensity = st.number_input(
        "Grid carbon intensity (kgCO2e/kWh):",
        min_value=0.01,
        max_value=1.5,
        value=float(GRID_CARBON_INTENSITY_KG_PER_KWH),
        step=0.01,
    )

    daily_requests = st.select_slider(
        "Daily request volume:",
        options=[1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000],
        value=100_000,
        format_func=lambda x: f"{x:,.0f}",
    )

with sim_c2:
    # ── Calculations ──────────────────────────────────────────────────────────
    visual_energy = baseline_energy_per_1k * (visual_pct_of_total / 100.0)
    non_visual_energy = baseline_energy_per_1k - visual_energy

    # JEPA reduces visual processing; adds its own overhead
    jepa_visual_saving = visual_energy * (jepa_visual_reduction_pct / 100.0)
    jepa_overhead = baseline_energy_per_1k * (jepa_overhead_pct / 100.0)
    jepa_net_visual = visual_energy - jepa_visual_saving + jepa_overhead
    jepa_total = non_visual_energy + jepa_net_visual

    energy_reduction_pct = ((baseline_energy_per_1k - jepa_total) / baseline_energy_per_1k) * 100

    # Annualized
    annual_requests = daily_requests * 365
    annual_baseline_kwh = (baseline_energy_per_1k / 1000.0) * annual_requests
    annual_jepa_kwh = (jepa_total / 1000.0) * annual_requests
    annual_saved_kwh = annual_baseline_kwh - annual_jepa_kwh
    annual_baseline_co2_kg = annual_baseline_kwh * carbon_intensity
    annual_jepa_co2_kg = annual_jepa_kwh * carbon_intensity
    annual_saved_co2_kg = annual_baseline_co2_kg - annual_jepa_co2_kg

    # ── Stacked Bar Comparison ────────────────────────────────────────────────
    fig_comp = go.Figure()

    fig_comp.add_trace(go.Bar(
        name="Non-Visual (Text/Decode)",
        x=["Conventional Baseline", "JEPA-Enhanced"],
        y=[non_visual_energy, non_visual_energy],
        marker_color="#4B5563",
        text=[f"{non_visual_energy:.1f}", f"{non_visual_energy:.1f}"],
        textposition="inside",
        textfont=dict(color="#FFFFFF", size=11, family="JetBrains Mono, monospace"),
    ))
    fig_comp.add_trace(go.Bar(
        name="Visual Processing (LLM Prefill)",
        x=["Conventional Baseline", "JEPA-Enhanced"],
        y=[visual_energy, visual_energy - jepa_visual_saving],
        marker_color="#B91C1C",
        text=[f"{visual_energy:.1f}", f"{visual_energy - jepa_visual_saving:.1f}"],
        textposition="inside",
        textfont=dict(color="#FFFFFF", size=11, family="JetBrains Mono, monospace"),
    ))
    fig_comp.add_trace(go.Bar(
        name="JEPA Encoder Overhead",
        x=["Conventional Baseline", "JEPA-Enhanced"],
        y=[0, jepa_overhead],
        marker_color="#D97706",
        text=["", f"+{jepa_overhead:.1f}"],
        textposition="inside",
        textfont=dict(color="#FFFFFF", size=11, family="JetBrains Mono, monospace"),
    ))

    fig_comp.update_layout(
        barmode="stack",
        title=dict(
            text="Energy Decomposition per 1,000 Requests (kWh) — Hypothetical",
            font=dict(family="Fraunces, Georgia, serif", size=16, color="#000000"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(247,245,240,0.5)",
        font=dict(family="Inter, sans-serif", color="#000000", size=12),
        xaxis=dict(
            gridcolor="rgba(17,17,17,0.08)",
            tickfont=dict(family="JetBrains Mono, monospace", size=11, color="#000000"),
        ),
        yaxis=dict(
            title=dict(text="Energy (kWh per 1,000 requests)", font=dict(color="#000000")),
            gridcolor="rgba(17,17,17,0.08)",
            tickfont=dict(family="JetBrains Mono, monospace", size=11, color="#000000"),
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(17,17,17,0.15)",
            borderwidth=1,
            font=dict(size=11, color="#000000"),
        ),
        margin=dict(l=60, r=30, t=50, b=50),
        hoverlabel=dict(
            bgcolor="#000000",
            font_size=12,
            font_family="Inter, sans-serif",
            font_color="#FFFFFF",
        ),
    )

    st.plotly_chart(fig_comp, use_container_width=True)

    # ── Summary Metrics ───────────────────────────────────────────────────────
    if energy_reduction_pct > 0:
        reduction_label = "NET ENERGY REDUCTION"
        color_cls = "green"
    else:
        reduction_label = "NET ENERGY INCREASE"
        color_cls = "red"

    met_c1, met_c2, met_c3, met_c4 = st.columns(4)

    with met_c1:
        st.markdown(
            metric_card(f"{energy_reduction_pct:.1f}%", reduction_label,
                        "Hypothetical net change in total pipeline energy.",
                        color_class=color_cls),
            unsafe_allow_html=True,
        )

    with met_c2:
        st.markdown(
            metric_card(f"{annual_saved_kwh:,.0f}", "KWH SAVED ANNUALLY",
                        f"At {daily_requests:,} requests/day over 365 days."),
            unsafe_allow_html=True,
        )

    with met_c3:
        st.markdown(
            metric_card(f"{annual_saved_co2_kg:,.0f}", "KG CO2e AVOIDED",
                        f"Using grid intensity of {carbon_intensity} kgCO2e/kWh.",
                        color_class="green" if annual_saved_co2_kg > 0 else "red"),
            unsafe_allow_html=True,
        )

    with met_c4:
        trees = annual_saved_co2_kg / CO2_KG_PER_TREE_PER_YEAR if annual_saved_co2_kg > 0 else 0
        st.markdown(
            metric_card(f"{trees:,.0f}", "TREE-YEARS EQUIVALENT",
                        "Mature trees growing for one year to sequester equivalent carbon."),
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — EXPERIMENTAL PROTOCOL
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(divider(), unsafe_allow_html=True)

st.markdown(
    "<div class='chapter-marker'>SECTION 7 — EXPERIMENTAL PROTOCOL</div>"
    "<h3>How to Test the Carbon-Reduction Hypothesis</h3>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    A controlled experiment is required to determine whether a JEPA-enhanced multimodal
    LLM system actually reduces energy and emissions. The following table specifies the
    necessary experimental controls:
    """
)

protocol_data = pd.DataFrame({
    "Experimental Factor": [
        "Input Dataset",
        "Perception Stage",
        "Language Model",
        "Task and Prompts",
        "Output Quality",
        "Energy and Emissions",
        "Hardware Platform",
        "Numerical Precision",
    ],
    "Conventional Baseline": [
        "Same dataset",
        "Standard visual/video encoder (CLIP, ViT)",
        "Same LLM (where technically feasible)",
        "Same task prompts",
        "Measured (accuracy, F1, human eval)",
        "Measured end-to-end (CodeCarbon/NVML)",
        "Same GPU and configuration",
        "Same precision (FP16 or INT8)",
    ],
    "JEPA-Enhanced System": [
        "Same dataset",
        "JEPA-based encoder (I-JEPA or V-JEPA)",
        "Same LLM",
        "Same task prompts",
        "Measured (accuracy, F1, human eval)",
        "Measured end-to-end (CodeCarbon/NVML)",
        "Same GPU and configuration",
        "Same precision (FP16 or INT8)",
    ],
})

st.dataframe(protocol_data, use_container_width=True, hide_index=True)

# ── Metrics to Record ─────────────────────────────────────────────────────────
st.markdown(
    "<div class='chapter-marker'>REQUIRED METRICS</div>",
    unsafe_allow_html=True,
)

metrics_c1, metrics_c2 = st.columns(2)

with metrics_c1:
    st.markdown(
        """
        **Primary Energy Metrics:**
        - Electricity consumption (kWh) for the full pipeline including preprocessing
        - GPU energy via NVML hardware registers
        - Operational CO2e using location- and time-appropriate carbon-intensity factors
        """
    )

with metrics_c2:
    st.markdown(
        """
        **Quality and Resource Metrics:**
        - Task quality (accuracy, F1, human evaluation scores)
        - End-to-end latency (seconds)
        - GPU model, memory utilization, batch size, precision, data-centre PUE
        """
    )

# ── Formulae ──────────────────────────────────────────────────────────────────
st.markdown(
    formula_box(
        "Energy Reduction",
        r"\text{Reduction (\%)} = \frac{E_{\text{baseline}} - E_{\text{JEPA}}}{E_{\text{baseline}}} \times 100",
        "Percentage reduction in total end-to-end pipeline energy consumption. "
        "A positive value indicates the JEPA system uses less energy.",
        tag="EVALUATION METRIC",
    ),
    unsafe_allow_html=True,
)

st.markdown(
    formula_box(
        "Operational Emissions",
        r"\text{CO}_{2}\text{e} = E \;(\text{kWh}) \times I_{\text{grid}} \;(\text{kgCO}_{2}\text{e/kWh})",
        "Direct operational carbon dioxide equivalent from electricity consumption. "
        "Uses location-based or time-varying grid carbon intensity factors.",
        tag="CARBON ACCOUNTING",
    ),
    unsafe_allow_html=True,
)

st.markdown(
    formula_box(
        "Emissions Reduction",
        r"\text{Reduction (\%)} = \frac{C_{\text{baseline}} - C_{\text{JEPA}}}{C_{\text{baseline}}} \times 100",
        "Percentage reduction in operational CO2e emissions. When grid carbon intensity is constant, "
        "this equals the energy reduction percentage.",
        tag="EVALUATION METRIC",
    ),
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 — RESEARCH LITERATURE
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(divider(), unsafe_allow_html=True)

st.markdown(
    "<div class='chapter-marker'>SECTION 8 — ACADEMIC FOUNDATIONS</div>"
    "<h3>Foundational Research Literature</h3>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <p style="color: var(--graphite); line-height: 1.7; font-size: 0.95rem; margin-bottom: 1rem;">
        The following papers provide relevant theoretical and empirical foundations.
        They support the concepts of JEPA, predictive representation learning, and the
        importance of measuring energy and carbon emissions. They do <strong>not</strong>
        collectively prove that a JEPA-LLM pipeline produces lower emissions.
    </p>
    """,
    unsafe_allow_html=True,
)

for idx, paper in enumerate(JEPA_PAPERS):
    st.markdown(
        f"""
        <div style="border-left: 2px solid var(--ink); padding: 1rem 0 1rem 1.5rem;
                    margin-bottom: 1.2rem; background: var(--smoke); border-radius: 0 2px 2px 0;">
            <div style="font-family: var(--mono); font-size: 11px; text-transform: uppercase;
                        letter-spacing: 0.05em; color: var(--accent-green); margin-bottom: 0.3rem;">
                PAPER {idx + 1} &mdash; {paper['key']}
            </div>
            <div style="font-family: var(--sans); font-weight: 600; font-size: 0.95rem; color: var(--ink);">
                <a href="{paper['url']}" target="_blank" style="color: var(--ink); text-decoration: underline;">
                    {paper['title']}
                </a>
            </div>
            <div style="font-family: var(--mono); font-size: 12px; color: var(--graphite);
                        margin: 0.3rem 0 0.5rem 0;">
                {paper['authors']}
            </div>
            <div style="font-family: var(--sans); font-size: 0.88rem; color: var(--graphite); line-height: 1.5;">
                <strong>Research Relevance:</strong> {paper['relevance']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 9 — LIMITATIONS & RESEARCH CAUTIONS
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(divider(), unsafe_allow_html=True)

st.markdown(
    "<div class='chapter-marker'>SECTION 9 — DELIMITATIONS AND CAUTIONS</div>"
    "<h3>Limitations and Research Integrity Notes</h3>",
    unsafe_allow_html=True,
)

lim_c1, lim_c2 = st.columns(2)

with lim_c1:
    st.markdown(
        """
        - JEPA was developed as a **representation-learning approach**, not as a direct
          replacement for autoregressive text-generation models.
        - JEPA encoders and predictors **consume energy** and must be included in the
          system boundary of any carbon accounting.
        - A smaller representation does **not necessarily** mean lower total energy
          consumption; overhead may exceed savings.
        - A compressed representation **may discard information** and reduce task accuracy.
        """
    )

with lim_c2:
    st.markdown(
        """
        - Training energy, inference energy, hardware manufacture, and data-centre
          overhead are **distinct accounting categories** that must not be conflated.
        - Carbon emissions depend on the **electricity mix and accounting method**
          used (location-based vs. market-based).
        - A claim of lower emissions requires **comparable task quality** and an
          end-to-end measurement, not merely fewer tokens or model operations.
        - **Video-based multimodal LLMs** represent the strongest test case due to
          high temporal redundancy in video data.
        """
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 10 — RESEARCH HYPOTHESIS & RECOMMENDED SCOPE
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(divider(), unsafe_allow_html=True)

st.markdown(
    "<div class='chapter-marker'>SECTION 10 — THESIS CONTRIBUTION</div>"
    "<h3>Research Hypothesis and Recommended Scope</h3>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="compare-panel" style="border-left: 3px solid var(--ink); padding: 2rem;">
        <div style="font-family: var(--mono); font-size: 11px; text-transform: uppercase;
                    letter-spacing: 0.05em; color: var(--graphite); margin-bottom: 1rem;">
            SUGGESTED RESEARCH TITLE
        </div>
        <h3 style="font-family: var(--serif); font-size: 1.3rem; font-weight: 400; color: var(--ink);
                   line-height: 1.3; margin: 0 0 1.5rem 0;">
            Reducing the Carbon Footprint of Large Language Model Systems
            Through Joint-Embedding Predictive Architectures (JEPA):
            An Energy-Efficiency Study
        </h3>
        <div style="font-family: var(--mono); font-size: 11px; text-transform: uppercase;
                    letter-spacing: 0.05em; color: var(--graphite); margin-bottom: 0.75rem;">
            HYPOTHESIS
        </div>
        <p style="font-family: var(--sans); font-size: 0.95rem; line-height: 1.7; color: var(--graphite);">
            Integrating Joint-Embedding Predictive Architectures into multimodal LLM
            pipelines may reduce redundant visual processing by learning compact semantic
            representations. This could lower downstream computation, electricity consumption,
            and operational carbon emissions while maintaining comparable task performance.
            The hypothesis must be tested through controlled end-to-end measurements.
        </p>
        <div style="font-family: var(--mono); font-size: 11px; text-transform: uppercase;
                    letter-spacing: 0.05em; color: var(--graphite); margin: 1.5rem 0 0.75rem 0;">
            RESEARCH QUESTIONS
        </div>
        <ol style="font-size: 0.92rem; color: var(--graphite); line-height: 1.7;">
            <li>Can JEPA reduce the computational requirements of multimodal data processing?</li>
            <li>Can a JEPA-based front end reduce the visual tokens, frames, or processing
                operations required by a downstream LLM?</li>
            <li>Can a JEPA-LLM hybrid achieve lower end-to-end energy use and operational
                CO2e while maintaining comparable accuracy and latency?</li>
        </ol>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="finding-box">
        <div class="finding-label">RECOMMENDED PROJECT SCOPE</div>
        <p>
            A practical prototype should focus on <strong>video question answering</strong>
            as the evaluation task. Video contains large amounts of temporally redundant
            information, making it the most promising test case for JEPA-based compression:
        </p>
        <ol style="line-height: 1.8;">
            <li>Use a public video dataset (e.g., MSRVTT-QA, ActivityNet-QA) with a fixed set of evaluation questions.</li>
            <li>Build a conventional video-encoder-plus-LLM baseline (e.g., CLIP-ViT + Llama-2-7B).</li>
            <li>Build a JEPA-based video representation pipeline (V-JEPA encoder) connected to the same LLM.</li>
            <li>Measure energy for both preprocessing and inference stages, not just the LLM call alone.</li>
            <li>Compare accuracy, latency, energy per request, and operational CO2e.</li>
            <li>Report limitations, run multiple trials, and present means with variability.</li>
        </ol>
    </div>
    """,
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  CONCLUSION
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(divider(), unsafe_allow_html=True)

st.markdown(
    quote(
        "JEPA offers a promising research direction for improving the efficiency of multimodal AI systems "
        "because it learns predictive latent representations rather than requiring reconstruction of every "
        "low-level input detail. This could help reduce redundant visual processing before information "
        "reaches an LLM. However, JEPA's potential energy advantage must be demonstrated empirically. "
        "A fair comparison should measure the complete pipeline, include JEPA's own computational cost, "
        "and verify that task quality remains comparable. The defensible conclusion is therefore that "
        "JEPA may help reduce the operational energy and carbon footprint of multimodal LLM systems, "
        "but this remains a testable hypothesis rather than an established result.",
        attribution="Project Conclusion Statement",
    ),
    unsafe_allow_html=True,
)
