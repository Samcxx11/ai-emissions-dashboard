"""
Page 1 — Energy & CO₂ Comparison by Model

Bar charts comparing average energy and CO₂ per response across models,
plus a scatter plot showing the relationship between model size and energy.
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

st.set_page_config(page_title="Energy & CO₂ Comparison", page_icon="⚡", layout="wide")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .section-header {
        font-size: 1.5rem; font-weight: 600; color: #ecf0f1;
        margin: 2rem 0 1rem 0; padding-bottom: 0.5rem;
        border-bottom: 2px solid #3498db;
    }
    .insight-box {
        background: linear-gradient(135deg, #1a1d23 0%, #2c3e50 100%);
        border-radius: 10px; padding: 1.2rem; margin: 1rem 0;
        border-left: 4px solid #f39c12;
    }
    .insight-box p { color: #bdc3c7; margin: 0; }
    .compare-metric {
        background: #1a1d23; border-radius: 10px; padding: 1rem;
        text-align: center; border: 1px solid #34495e;
    }
    .compare-metric .value { font-size: 1.8rem; font-weight: 700; }
    .compare-metric .label { font-size: 0.85rem; color: #95a5a6; margin-top: 0.3rem; }
</style>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    return pd.read_csv(os.path.join(data_dir, "inference_results.csv"))

df = load_data()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# ⚡ Energy & CO₂ Comparison")
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
st.markdown('<p class="section-header">📊 Quick Comparison</p>', unsafe_allow_html=True)

cols = st.columns(len(selected_models))
for i, model in enumerate(selected_models):
    model_data = filtered[filtered["model"] == model]
    avg_energy_wh = model_data["energy_kwh"].mean() * 1000
    avg_co2 = model_data["co2_grams"].mean()
    color = MODEL_COLORS.get(model, "#3498db")
    with cols[i]:
        st.markdown(
            f"""<div class="compare-metric">
                <div class="value" style="color: {color};">{avg_energy_wh:.3f} Wh</div>
                <div class="label">{model}<br>{avg_co2:.3f} g CO₂</div>
            </div>""",
            unsafe_allow_html=True,
        )

# ── Energy chart ──────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">🔋 Energy per Response</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    fig_energy = bar_energy_by_model(filtered)
    st.plotly_chart(fig_energy, use_container_width=True)

with col2:
    fig_co2 = bar_co2_by_model(filtered)
    st.plotly_chart(fig_co2, use_container_width=True)

# ── Insight ───────────────────────────────────────────────────────────────────
if len(selected_models) >= 2:
    model_avgs = filtered.groupby("model")["energy_kwh"].mean()
    cheapest = model_avgs.idxmin()
    costliest = model_avgs.idxmax()
    ratio = model_avgs[costliest] / model_avgs[cheapest]

    st.markdown(
        f"""<div class="insight-box">
            <p>💡 <strong>{costliest}</strong> uses <strong>{ratio:.1f}× more energy</strong>
            per response than <strong>{cheapest}</strong>. That means every time you use
            the larger model for a task the smaller one could handle, you're spending
            {ratio:.1f}× more electricity — and emitting {ratio:.1f}× more CO₂.</p>
        </div>""",
        unsafe_allow_html=True,
    )

# ── Latency chart ─────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">⏱️ Latency per Response</p>', unsafe_allow_html=True)

fig_latency = bar_latency_by_model(filtered)
st.plotly_chart(fig_latency, use_container_width=True)

# ── Scatter: params vs energy ─────────────────────────────────────────────────
st.markdown(
    '<p class="section-header">📈 Model Size vs Energy — The Core Question</p>',
    unsafe_allow_html=True,
)

st.markdown(
    "Does bigger always mean more energy? This scatter plot shows the "
    "relationship between model size (number of parameters) and average "
    "energy consumption per response."
)

fig_scatter = scatter_params_vs_energy(filtered, MODEL_PARAMS)
st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown(
    """<div class="insight-box">
        <p>📐 The relationship between model size and energy is roughly
        <strong>linear to super-linear</strong> — doubling the parameters more
        than doubles the energy cost. This means the efficiency penalty of
        choosing an oversized model compounds at scale.</p>
    </div>""",
    unsafe_allow_html=True,
)

# ── Energy per Token ──────────────────────────────────────────────────────────
st.markdown(
    '<p class="section-header">⚡ Energy per Output Token</p>',
    unsafe_allow_html=True,
)

st.markdown(
    "How much energy does each *token* of output cost? This normalizes "
    "for the fact that larger models tend to generate more tokens."
)

if "tokens_generated" in filtered.columns:
    fig_ept = bar_energy_per_token(filtered)
    st.plotly_chart(fig_ept, use_container_width=True)

# ── Summary table ─────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">📋 Summary Table</p>', unsafe_allow_html=True)

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
