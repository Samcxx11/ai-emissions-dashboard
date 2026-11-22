"""
Chapter 01 — Energy & Carbon Footprint by Model Architecture

Comparative empirical evaluation of average energy (Wh) and greenhouse gas
emissions (gCO2e) across autoregressive transformer parameter scales.
"""

import streamlit as st
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.charts import (
    bar_energy_by_model,
    bar_co2_by_model,
    bar_latency_by_model,
    scatter_params_vs_energy,
    bar_energy_per_token,
)
from utils.constants import MODEL_COLORS, MODEL_ORDER, MODEL_PARAMS
from utils.theme import EDITORIAL_CSS, chapter, divider, finding

st.set_page_config(
    page_title="Energy & Carbon Comparison",
    layout="wide",
)

# ── Apply Editorial Theme ─────────────────────────────────────────────────────
st.markdown(EDITORIAL_CSS, unsafe_allow_html=True)


# ── Load Data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    return pd.read_csv(os.path.join(data_dir, "inference_results.csv"))


df = load_data()


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    chapter(1, "MODEL SCALE & ENERGETIC MULTIPLIERS", "Evaluating energy expenditure against parameter count"),
    unsafe_allow_html=True,
)
st.markdown("<h1>Energy and Carbon Footprint by Architecture</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='hero-subtitle'>"
    "A direct comparison of energy consumption (Wh), greenhouse gas emissions (gCO2e), and latency "
    "across all four benchmarked architectures answering the identical 20-prompt evaluation suite."
    "</p>",
    unsafe_allow_html=True,
)


# ── Model Filter ──────────────────────────────────────────────────────────────
selected_models = st.multiselect(
    "Active Architectures in Comparison:",
    options=MODEL_ORDER,
    default=MODEL_ORDER,
    help="Select which architectures to include in the visual comparison.",
)

filtered = df[df["model"].isin(selected_models)]

if filtered.empty:
    st.warning("Please select at least one architecture.")
    st.stop()


# ── Comparative Metric Panels ────────────────────────────────────────────────
st.markdown(
    "<div class='chapter-marker'>EMPIRICAL COMPARISON</div>"
    "<h3>Per-Query Response Baselines</h3>",
    unsafe_allow_html=True,
)

cols = st.columns(len(selected_models))
for i, model in enumerate(selected_models):
    model_data = filtered[filtered["model"] == model]
    avg_energy_wh = model_data["energy_kwh"].mean() * 1000
    avg_co2 = model_data["co2_grams"].mean()
    color = MODEL_COLORS.get(model, "#111111")
    clean_name = model.replace(" (Real)", "")
    with cols[i]:
        st.markdown(
            f"""
            <div class="metric-editorial">
                <div style="font-family: var(--mono); font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--graphite); margin-bottom: 0.5rem;">
                    {clean_name}
                </div>
                <div class="metric-number" style="color: {color}; font-size: 2.2rem;">
                    {avg_energy_wh:.3f} <span style="font-size: 1.1rem; color: var(--graphite);">Wh</span>
                </div>
                <div class="metric-unit">{avg_co2:.3f} g CO2e</div>
                <div class="metric-desc">Baseline energy and emissions per standardized inference response.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ── Key Analytical Finding ───────────────────────────────────────────────────
st.markdown(divider(), unsafe_allow_html=True)

st.markdown(
    finding(
        "Empirical measurement demonstrates that energy expenditure does not scale linearly with parameter count; "
        "rather, memory bandwidth constraints and quadratic self-attention during the prefill phase create a "
        "<strong>super-linear 14.5× energetic penalty between 1.1B (TinyLlama) and 13.0B (Llama-2)</strong>. "
        "For routine tasks that require semantic parsing without multi-step reasoning, smaller architectures "
        "deliver identical functional output with over 90% carbon savings.",
        label="EMPIRICAL SCALING LAW",
    ),
    unsafe_allow_html=True,
)


# ── Core Visualizations ───────────────────────────────────────────────────────
st.markdown(
    "<div class='chapter-marker'>DATA VISUALIZATION</div>"
    "<h3>Comparative Energetics & Emissions</h3>",
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2)

with col1:
    fig_energy = bar_energy_by_model(filtered)
    st.plotly_chart(fig_energy, use_container_width=True)

with col2:
    fig_co2 = bar_co2_by_model(filtered)
    st.plotly_chart(fig_co2, use_container_width=True)


# ── Parameter Scale vs Energy ─────────────────────────────────────────────────
st.markdown(divider(), unsafe_allow_html=True)

st.markdown(
    "<div class='chapter-marker'>REGRESSION ANALYSIS</div>"
    "<h3>Active Parameters vs. Energy per Query</h3>",
    unsafe_allow_html=True,
)

fig_scatter = scatter_params_vs_energy(filtered, MODEL_PARAMS)
st.plotly_chart(fig_scatter, use_container_width=True)


# ── Secondary Visualizations: Latency & Energy Per Token ───────────────────────
st.markdown(divider(), unsafe_allow_html=True)

st.markdown(
    "<div class='chapter-marker'>SYSTEM PROFILING</div>"
    "<h3>Execution Latency and Token-Level Density</h3>",
    unsafe_allow_html=True,
)

col3, col4 = st.columns(2)

with col3:
    fig_lat = bar_latency_by_model(filtered)
    st.plotly_chart(fig_lat, use_container_width=True)

with col4:
    fig_tok = bar_energy_per_token(filtered)
    st.plotly_chart(fig_tok, use_container_width=True)
