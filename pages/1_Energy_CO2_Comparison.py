"""
Page 1 — Energy & CO₂ Comparison by Model

Bar charts comparing average energy and CO₂ per response across models,
plus a scatter plot showing the relationship between model size and energy.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import streamlit as st
import pandas as pd
from utils.theme import EDITORIAL_CSS, apply_editorial_layout, chapter, divider, finding, quote, metric_card, big_number

from utils.charts import (
    bar_energy_by_model,
    bar_co2_by_model,
    bar_latency_by_model,
    scatter_params_vs_energy,
    bar_energy_per_token,
)
from utils.constants import MODEL_COLORS, MODEL_ORDER, MODEL_PARAMS

st.set_page_config(page_title="Energy & CO₂ Comparison", page_icon="⚡", layout="wide")

st.markdown(EDITORIAL_CSS, unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    return pd.read_csv(os.path.join(data_dir, "inference_results.csv"))

df = load_data()

# ── Header ────────────────────────────────────────────────────────────────────
chapter(1, "Energy & CO₂ Comparison")
st.markdown(
    "Comparing average energy consumption and carbon emissions per response "
    "across all tested models. All models answered the same 20 questions "
    "on the same hardware for a fair comparison."
)

# ── Filter ────────────────────────────────────────────────────────────────────
selected_models = st.multiselect(
    "Filter models:",
    options=MODEL_ORDER,
    default=MODEL_ORDER,
    help="Select which models to include in the comparison.",
)

filtered = df[df["model"].isin(selected_models)]

if filtered.empty:
    st.warning("Please select at least one model.")
    st.stop()

# ── Quick comparison metrics ──────────────────────────────────────────────────
divider()
chapter(2, "Quick Comparison")

cols = st.columns(len(selected_models))
for i, model in enumerate(selected_models):
    model_data = filtered[filtered["model"] == model]
    avg_energy_wh = model_data["energy_kwh"].mean() * 1000
    avg_co2 = model_data["co2_grams"].mean()
    color = MODEL_COLORS.get(model, "#3498db")
    with cols[i]:
        metric_card(
            value=f"{avg_energy_wh:.3f}", 
            unit="Wh", 
            description=f"{model} • {avg_co2:.3f} g CO₂"
        )

# ── Energy chart ──────────────────────────────────────────────────────────────
divider()
chapter(3, "Energy per Response")

col1, col2 = st.columns(2)

with col1:
    fig_energy = bar_energy_by_model(filtered)
    apply_editorial_layout(fig_energy)
    st.plotly_chart(fig_energy, use_container_width=True)

with col2:
    fig_co2 = bar_co2_by_model(filtered)
    apply_editorial_layout(fig_co2)
    st.plotly_chart(fig_co2, use_container_width=True)

# ── Insight ───────────────────────────────────────────────────────────────────
if len(selected_models) >= 2:
    model_avgs = filtered.groupby("model")["energy_kwh"].mean()
    cheapest = model_avgs.idxmin()
    costliest = model_avgs.idxmax()
    ratio = model_avgs[costliest] / model_avgs[cheapest]

    finding(
        f"**{costliest}** uses **{ratio:.1f}× more energy** "
        f"per response than **{cheapest}**. That means every time you use "
        f"the larger model for a task the smaller one could handle, you're spending "
        f"{ratio:.1f}× more electricity — and emitting {ratio:.1f}× more CO₂.",
        "EFFICIENCY PENALTY"
    )

# ── Latency chart ─────────────────────────────────────────────────────────────
divider()
chapter(4, "Latency per Response")

fig_latency = bar_latency_by_model(filtered)
apply_editorial_layout(fig_latency)
st.plotly_chart(fig_latency, use_container_width=True)

# ── Scatter: params vs energy ─────────────────────────────────────────────────
divider()
chapter(5, "Model Size vs Energy — The Core Question")

st.markdown(
    "Does bigger always mean more energy? This scatter plot shows the "
    "relationship between model size (number of parameters) and average "
    "energy consumption per response."
)

fig_scatter = scatter_params_vs_energy(filtered, MODEL_PARAMS)
apply_editorial_layout(fig_scatter)
st.plotly_chart(fig_scatter, use_container_width=True)

finding(
    "The relationship between model size and energy is roughly "
    "**linear to super-linear** — doubling the parameters more "
    "than doubles the energy cost. This means the efficiency penalty of "
    "choosing an oversized model compounds at scale.",
    "SCALING LAW"
)

# ── Energy per Token ──────────────────────────────────────────────────────────
divider()
chapter(6, "Energy per Output Token")

st.markdown(
    "How much energy does each *token* of output cost? This normalizes "
    "for the fact that larger models tend to generate more tokens."
)

if "tokens_generated" in filtered.columns:
    fig_ept = bar_energy_per_token(filtered)
    apply_editorial_layout(fig_ept)
    st.plotly_chart(fig_ept, use_container_width=True)

# ── Summary table ─────────────────────────────────────────────────────────────
divider()
chapter(7, "Summary Table")

summary_agg = {
    "Parameters": ("parameters_b", "first"),
    "Avg_Energy_Wh": ("energy_kwh", lambda x: round(x.mean() * 1000, 4)),
    "Avg_CO2_g": ("co2_grams", lambda x: round(x.mean(), 4)),
    "Avg_Latency_s": ("duration_s", lambda x: round(x.mean(), 2)),
    "Avg_Tokens": ("tokens_generated", lambda x: round(x.mean(), 1)),
    "Queries": ("query_id", "count"),
}

if "energy_per_token" in filtered.columns:
    summary_agg["Avg_Energy_per_Token"] = ("energy_per_token", lambda x: round(x.mean() * 1e6, 4))

if "gpu_memory_mb" in filtered.columns:
    summary_agg["Avg_GPU_Memory_MB"] = ("gpu_memory_mb", lambda x: round(x.mean(), 0))

summary_table = (
    filtered.groupby("model")
    .agg(**summary_agg)
    .reindex([m for m in MODEL_ORDER if m in selected_models])
    .reset_index()
    .rename(columns={"model": "Model", "Parameters": "Params (B)"})
)

st.dataframe(summary_table, use_container_width=True, hide_index=True)
