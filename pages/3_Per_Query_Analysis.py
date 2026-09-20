"""
Page 3 — Per-Query Analysis

Deep dive into individual query measurements: histograms, box plots,
and the raw data table for full transparency.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import streamlit as st
import pandas as pd
from utils.theme import EDITORIAL_CSS, apply_editorial_layout, chapter, divider, finding, quote, metric_card, big_number

from utils.charts import histogram_energy, box_energy_by_model, box_co2_by_model
from utils.constants import MODEL_COLORS, MODEL_ORDER
import plotly.express as px

st.set_page_config(page_title="Per-Query Analysis", page_icon="🔍", layout="wide")

st.markdown(EDITORIAL_CSS, unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    return pd.read_csv(os.path.join(data_dir, "inference_results.csv"))

df = load_data()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🔍 Per-Query Analysis")
st.markdown(
    "This page shows the raw measurement data — distributions, variance, "
    "and the full CSV table. Showing raw data plus process equals credibility."
)

# ── Filter ────────────────────────────────────────────────────────────────────
selected_models = st.multiselect(
    "Filter models:",
    options=MODEL_ORDER,
    default=MODEL_ORDER,
    key="per_query_filter",
)

filtered = df[df["model"].isin(selected_models)]

if filtered.empty:
    st.warning("Please select at least one model.")
    st.stop()

# ── Descriptive statistics ────────────────────────────────────────────────────
st.markdown(chapter(1, "Descriptive Statistics"), unsafe_allow_html=True)

for model in selected_models:
    model_df = filtered[filtered["model"] == model]
    energy_wh = model_df["energy_kwh"] * 1000
    color = MODEL_COLORS.get(model, "#3498db")

    with st.expander(f"**{model}** — {len(model_df)} queries", expanded=(len(selected_models) <= 2)):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(metric_card(f"{energy_wh.mean():.4f}", "Wh", "Mean Energy", ""), unsafe_allow_html=True)
        with col2:
            st.markdown(metric_card(f"{energy_wh.std():.4f}", "Wh", "Std Dev", ""), unsafe_allow_html=True)
        with col3:
            st.markdown(metric_card(f"{energy_wh.min():.4f}", "Wh", "Min", ""), unsafe_allow_html=True)
        with col4:
            st.markdown(metric_card(f"{energy_wh.max():.4f}", "Wh", "Max", ""), unsafe_allow_html=True)

        col1b, col2b, col3b, col4b = st.columns(4)
        with col1b:
            st.markdown(metric_card(f"{model_df['co2_grams'].mean():.4f}", "g", "Mean CO₂", ""), unsafe_allow_html=True)
        with col2b:
            st.markdown(metric_card(f"{model_df['duration_s'].mean():.2f}", "s", "Mean Latency", ""), unsafe_allow_html=True)
        with col3b:
            st.markdown(metric_card(f"{model_df['tokens_generated'].mean():.0f}", "", "Mean Tokens", ""), unsafe_allow_html=True)
        with col4b:
            st.markdown(metric_card(f"{len(model_df)}", "", "Total Queries", ""), unsafe_allow_html=True)

st.markdown(divider(), unsafe_allow_html=True)

# ── Histogram ─────────────────────────────────────────────────────────────────
st.markdown(chapter(2, "Energy Distribution"), unsafe_allow_html=True)

st.markdown(
    "This histogram shows how energy consumption is distributed across all "
    "queries. Notice how the distributions for different models are well-separated — "
    "confirming that model size has a consistent effect, not just an average one."
)

fig_hist = histogram_energy(filtered)
apply_editorial_layout(fig_hist)
st.plotly_chart(fig_hist, use_container_width=True)

st.markdown(divider(), unsafe_allow_html=True)

# ── Box plots ─────────────────────────────────────────────────────────────────
st.markdown(chapter(3, "Energy & CO₂ Spread"), unsafe_allow_html=True)

st.markdown(
    "Box plots show the median, quartiles, and outliers. "
    "Tight boxes = consistent measurements. Wide boxes or many outliers "
    "= high variance (potentially noisy hardware or variable prompt difficulty)."
)

col1, col2 = st.columns(2)

with col1:
    fig_box_energy = box_energy_by_model(filtered)
    apply_editorial_layout(fig_box_energy)
    st.plotly_chart(fig_box_energy, use_container_width=True)

with col2:
    fig_box_co2 = box_co2_by_model(filtered)
    apply_editorial_layout(fig_box_co2)
    st.plotly_chart(fig_box_co2, use_container_width=True)

# ── Insight ───────────────────────────────────────────────────────────────────
st.markdown(
    finding(
        "If a model's energy consumption varies wildly between queries, it means some queries are much more 'expensive' than others — suggesting that not all prompts are created equal. Longer, more complex prompts tend to generate more tokens and consume more energy.", 
        "Why variance matters"
    ),
    unsafe_allow_html=True,
)

st.markdown(divider(), unsafe_allow_html=True)

# ── Tokens vs Energy correlation ──────────────────────────────────────────────
st.markdown(chapter(4, "Tokens Generated vs Energy"), unsafe_allow_html=True)

filtered_copy = filtered.copy()
filtered_copy["energy_wh"] = filtered_copy["energy_kwh"] * 1000

fig_scatter = px.scatter(
    filtered_copy,
    x="tokens_generated",
    y="energy_wh",
    color="model",
    color_discrete_map=MODEL_COLORS,
    opacity=0.7,
    trendline="ols",
    labels={"tokens_generated": "Tokens Generated", "energy_wh": "Energy (Wh)"},
)
fig_scatter.update_layout(title="Tokens Generated vs Energy Consumed")
apply_editorial_layout(fig_scatter)
st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown(divider(), unsafe_allow_html=True)

# ── Raw data table ────────────────────────────────────────────────────────────
st.markdown(chapter(5, "Raw Data (CSV)"), unsafe_allow_html=True)

st.markdown(
    "Full transparency: here's every measurement. This is the actual data "
    "behind every chart in this dashboard. You can sort by any column."
)

display_df = filtered.copy()
display_df["energy_wh"] = (display_df["energy_kwh"] * 1000).round(4)

display_cols = ["query_id", "model", "parameters_b", "prompt", "tokens_generated",
     "energy_wh", "co2_grams", "duration_s"]
rename_map = {
    "query_id": "Query #",
    "model": "Model",
    "parameters_b": "Params (B)",
    "prompt": "Prompt",
    "tokens_generated": "Tokens",
    "energy_wh": "Energy (Wh)",
    "co2_grams": "CO₂ (g)",
    "duration_s": "Latency (s)",
}

if "energy_per_token" in display_df.columns:
    display_df["energy_per_token_uh"] = (display_df["energy_per_token"] * 1e6).round(2)
    display_cols.append("energy_per_token_uh")
    rename_map["energy_per_token_uh"] = "µWh/Token"

if "gpu_memory_mb" in display_df.columns:
    display_cols.append("gpu_memory_mb")
    rename_map["gpu_memory_mb"] = "GPU Mem (MB)"

display_df = display_df[display_cols].rename(columns=rename_map)

st.dataframe(display_df, use_container_width=True, height=400, hide_index=True)

# ── Download button ───────────────────────────────────────────────────────────
csv = filtered.to_csv(index=False)
st.download_button(
    label="⬇️ Download filtered data as CSV",
    data=csv,
    file_name="inference_results_filtered.csv",
    mime="text/csv",
)

st.markdown(divider(), unsafe_allow_html=True)
st.markdown(
    quote(
        "Reproducibility requires sharing actual measurements, not just aggregates. Raw data plus process = credibility.", 
        "Henderson et al. (JMLR 2020)"
    ),
    unsafe_allow_html=True
)
