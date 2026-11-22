"""
Chapter 03 — Statistical Variance & Query Distributions

Empirical inspection of individual prompt variance, box-whisker dispersions,
and raw telemetry distributions across test sequences.
"""

import streamlit as st
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.charts import histogram_energy, box_energy_by_model, box_co2_by_model
from utils.constants import MODEL_COLORS, MODEL_ORDER
from utils.theme import EDITORIAL_CSS, chapter, divider, finding

st.set_page_config(
    page_title="Statistical Variance & Distributions",
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
    chapter(3, "DISTRIBUTIONAL ANALYSIS", "Statistical dispersion and individual query variance"),
    unsafe_allow_html=True,
)
st.markdown("<h1>Statistical Variance and Distributions</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='hero-subtitle'>"
    "A transparent inspection of raw measurement distributions across all 200 analytical inference queries, "
    "evaluating standard deviations, outlier sensitivity, and interquartile spreads."
    "</p>",
    unsafe_allow_html=True,
)


# ── Model Filter ──────────────────────────────────────────────────────────────
selected_models = st.multiselect(
    "Active Architectures:",
    options=MODEL_ORDER,
    default=MODEL_ORDER,
)

filtered = df[df["model"].isin(selected_models)]

if filtered.empty:
    st.warning("Please select at least one architecture.")
    st.stop()


# ── Statistical Overview ─────────────────────────────────────────────────────
st.markdown(
    "<div class='chapter-marker'>SUMMARY STATISTICS</div>"
    "<h3>Per-Architecture Dispersion Indices</h3>",
    unsafe_allow_html=True,
)

summary_stats = (
    filtered.groupby("model")
    .agg(
        mean_wh=("energy_kwh", lambda x: (x.mean() * 1000)),
        std_wh=("energy_kwh", lambda x: (x.std() * 1000)),
        min_wh=("energy_kwh", lambda x: (x.min() * 1000)),
        max_wh=("energy_kwh", lambda x: (x.max() * 1000)),
        mean_co2=("co2_grams", "mean"),
    )
    .reindex(selected_models)
    .reset_index()
)

cols = st.columns(len(selected_models))
for i, row in summary_stats.iterrows():
    clean_name = row['model'].replace(" (Real)", "")
    with cols[i]:
        st.markdown(
            f"""
            <div class="metric-editorial">
                <div style="font-family: var(--mono); font-size: 11px; text-transform: uppercase; color: var(--graphite); margin-bottom: 0.5rem;">
                    {clean_name}
                </div>
                <div class="metric-number" style="font-size: 1.8rem;">
                    {row['mean_wh']:.3f} <span style="font-size: 0.9rem; color: var(--graphite);">± {row['std_wh']:.3f}</span>
                </div>
                <div class="metric-unit">Range: {row['min_wh']:.3f} – {row['max_wh']:.3f} Wh</div>
                <div class="metric-desc">Standard deviation reflects output sequence length variance under deterministic greedy decoding.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ── Distribution Chart ────────────────────────────────────────────────────────
st.markdown(divider(), unsafe_allow_html=True)

st.markdown(
    "<div class='chapter-marker'>HISTOGRAM PROFILE</div>"
    "<h3>Cumulative Energy Distribution Across All Runs</h3>",
    unsafe_allow_html=True,
)

fig_hist = histogram_energy(filtered)
st.plotly_chart(fig_hist, use_container_width=True)


# ── Box & Whisker Charts ──────────────────────────────────────────────────────
st.markdown(divider(), unsafe_allow_html=True)

st.markdown(
    "<div class='chapter-marker'>INTERQUARTILE RANGE</div>"
    "<h3>Variance and Outlier Spread</h3>",
    unsafe_allow_html=True,
)

b1, b2 = st.columns(2)

with b1:
    fig_box_e = box_energy_by_model(filtered)
    st.plotly_chart(fig_box_e, use_container_width=True)

with b2:
    fig_box_c = box_co2_by_model(filtered)
    st.plotly_chart(fig_box_c, use_container_width=True)


# ── Raw Data Table ────────────────────────────────────────────────────────────
st.markdown(divider(), unsafe_allow_html=True)

st.markdown(
    "<div class='chapter-marker'>AUDIT LEDGER</div>"
    "<h3>Complete Raw Telemetry Dataset</h3>",
    unsafe_allow_html=True,
)

display_df = filtered[[
    "query_id", "model", "prompt", "tokens_generated", 
    "energy_kwh", "duration_s", "co2_grams", "gpu_memory_mb"
]].copy()

display_df["energy_wh"] = display_df["energy_kwh"] * 1000
display_df = display_df.drop(columns=["energy_kwh"])
display_df = display_df.rename(columns={
    "query_id": "ID",
    "model": "Architecture",
    "prompt": "Prompt",
    "tokens_generated": "Output Tokens",
    "duration_s": "Latency (s)",
    "co2_grams": "CO2e (g)",
    "gpu_memory_mb": "VRAM (MB)",
    "energy_wh": "Energy (Wh)",
})

st.dataframe(display_df, use_container_width=True, height=350)
