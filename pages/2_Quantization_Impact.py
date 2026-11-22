"""
Chapter 02 — Precision & Quantization Dynamics

Evaluating the efficiency gains and semantic degradation of post-training
quantization (FP16 → INT8 → INT4) on Llama-2-7B.
"""

import streamlit as st
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.charts import (
    bar_quantization_energy,
    bar_quantization_co2,
    line_quality_vs_precision,
)
from utils.constants import QUANTIZATION_COLORS, QUANT_ORDER
from utils.theme import EDITORIAL_CSS, chapter, divider, finding

st.set_page_config(
    page_title="Quantization Dynamics",
    layout="wide",
)

# ── Apply Editorial Theme ─────────────────────────────────────────────────────
st.markdown(EDITORIAL_CSS, unsafe_allow_html=True)


# ── Load Data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    return pd.read_csv(os.path.join(data_dir, "quantization_comparison.csv"))


df = load_data()


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    chapter(2, "POST-TRAINING QUANTIZATION", "Compression efficiency vs. semantic degradation trade-offs"),
    unsafe_allow_html=True,
)
st.markdown("<h1>Precision and Quantization Dynamics</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='hero-subtitle'>"
    "Quantifying the energetic and environmental benefits of compressing weight matrices from 16-bit "
    "floating point (FP16) down to 8-bit (INT8) and 4-bit (INT4) representations on Llama-2-7B."
    "</p>",
    unsafe_allow_html=True,
)


# ── Technical Background ──────────────────────────────────────────────────────
with st.expander("Technical Background: Weight Quantization Principles", expanded=False):
    st.markdown(
        """
        **Post-Training Quantization (PTQ)** maps continuous 16-bit floating-point weights into discrete 
        low-bit integer representations (INT8 or NormalFloat4) without necessitating full architectural retraining:
        
        - **FP16 (Half Precision):** 16 bits per parameter (2 bytes). Standard baseline yielding maximum fidelity with uncompressed VRAM utilization (~14,032 MB).
        - **INT8 (8-Bit Quantized):** Vector-wise integer scaling via `BitsAndBytesConfig(load_in_8bit=True)`. Halves memory footprint (~7,570 MB) while preserving dynamic range.
        - **INT4 (4-Bit NormalFloat):** QLoRA/NF4 weight quantization with FP16 compute dequantization. Reduces VRAM to ~4,820 MB at the cost of higher rounding noise.
        """
    )


# ── Savings Metrics ───────────────────────────────────────────────────────────
st.markdown(
    "<div class='chapter-marker'>PRECISION REGIMES</div>"
    "<h3>Energy and Carbon Abatement by Precision Level</h3>",
    unsafe_allow_html=True,
)

avg_energy = df.groupby("precision")["energy_kwh"].mean()
fp16_energy = avg_energy.get("FP16", 1)

col1, col2, col3 = st.columns(3)

for col, precision in zip([col1, col2, col3], QUANT_ORDER):
    energy_wh = avg_energy.get(precision, 0) * 1000
    saving = (1 - avg_energy.get(precision, 0) / fp16_energy) * 100
    color = QUANTIZATION_COLORS[precision]

    with col:
        saving_text = f"Abatement: -{saving:.1f}% vs. FP16" if precision != "FP16" else "Baseline Benchmark"
        st.markdown(
            f"""
            <div class="metric-editorial">
                <div style="font-family: var(--mono); font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--graphite); margin-bottom: 0.5rem;">
                    {precision} Precision
                </div>
                <div class="metric-number" style="color: {color}; font-size: 2.2rem;">
                    {energy_wh:.3f} <span style="font-size: 1.1rem; color: var(--graphite);">Wh</span>
                </div>
                <div class="metric-unit">{saving_text}</div>
                <div class="metric-desc">Mean electrical draw per 200-token inference generation sequence.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ── Core Charts ───────────────────────────────────────────────────────────────
st.markdown(divider(), unsafe_allow_html=True)

st.markdown(
    "<div class='chapter-marker'>ENERGETIC TELEMETRY</div>"
    "<h3>Measured Energy and Emissions</h3>",
    unsafe_allow_html=True,
)

c1, c2 = st.columns(2)

with c1:
    fig_energy = bar_quantization_energy(df)
    st.plotly_chart(fig_energy, use_container_width=True)

with c2:
    fig_co2 = bar_quantization_co2(df)
    st.plotly_chart(fig_co2, use_container_width=True)


# ── Quality Trade-Off Analysis ────────────────────────────────────────────────
st.markdown(divider(), unsafe_allow_html=True)

st.markdown(
    "<div class='chapter-marker'>SEMANTIC QUALITY TRADEOFF</div>"
    "<h3>Accuracy Retention vs. Numerical Compression</h3>",
    unsafe_allow_html=True,
)

st.markdown(
    "<p style='color: var(--graphite); font-size: 0.95rem; line-height: 1.6;'>"
    "A central engineering concern in green computing is whether aggressive quantization harms downstream semantic performance. "
    "The evaluation below tracks empirical semantic fidelity (scored 0–100 across factual and reasoning queries) against precision regimes."
    "</p>",
    unsafe_allow_html=True,
)

fig_quality = line_quality_vs_precision(df)
st.plotly_chart(fig_quality, use_container_width=True)

# Quality drop stats
avg_quality = df.groupby("precision")["quality_score"].mean()
fp16_quality = avg_quality.get("FP16", 100)
int8_drop = fp16_quality - avg_quality.get("INT8", 0)
int4_drop = fp16_quality - avg_quality.get("INT4", 0)

st.markdown(
    finding(
        f"<strong>INT8 quantization emerges as the optimal drop-in production standard</strong>: it delivers a <strong>35.0% reduction in electricity and carbon emissions</strong> "
        f"with an insignificant quality degradation of merely <strong>{int8_drop:.1f} score points (-1.6%)</strong>. "
        f"Conversely, while INT4 yields greater nominal energy abatement, it incurs a steeper quality penalty of <strong>{int4_drop:.1f} points (-7.2%)</strong>, "
        "making it unsuitable for complex reasoning without domain-specific calibration.",
        label="PRODUCTION RECOMMENDATION",
    ),
    unsafe_allow_html=True,
)
