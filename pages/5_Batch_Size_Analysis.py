"""
Page 5 — Batch Size Analysis
"""

import streamlit as st
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.charts import bar_batch_size_energy, bar_batch_size_latency
from utils.constants import BATCH_SIZE_COLORS, BATCH_SIZE_ORDER

st.set_page_config(page_title="Batch Size Analysis", page_icon="📦", layout="wide")

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
        border-left: 4px solid #2ecc71;
    }
    .insight-box p { color: #bdc3c7; margin: 0; }
    .quant-card {
        background: #1a1d23; border-radius: 12px; padding: 1.5rem;
        text-align: center; border: 1px solid #34495e;
    }
    .quant-card .value { font-size: 2rem; font-weight: 700; }
    .quant-card .label { font-size: 0.9rem; color: #95a5a6; margin-top: 0.3rem; }
    .quant-card .saving { font-size: 1.1rem; color: #2ecc71; font-weight: 600; }
    .explainer {
        background: #1a1d23; border-radius: 10px; padding: 1.5rem;
        border: 1px solid #34495e; margin-bottom: 1.5rem;
    }
    .explainer h4 { color: #ecf0f1; margin-top: 0; }
    .explainer p { color: #bdc3c7; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    return pd.read_csv(os.path.join(data_dir, "batch_size_results.csv"))

df = load_data()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 📦 Batch Size Analysis")
st.markdown(
    "How does batching multiple queries affect energy consumption? "
    "Here we test **Llama-2-7B (Real)** at batch sizes 1, 2, and 4."
)

# ── What is batching? ─────────────────────────────────────────────────────────
with st.expander("🧠 What is batching? (click to expand)", expanded=False):
    st.markdown("""
    **Batching** means running multiple queries together instead of one at a time.
    It is like washing clothes in bulk vs one at a time — it's far more efficient to load the washing machine up!
    In AI inference, doing the heavy computation on multiple inputs at once amortizes the overhead of memory transfers.
    """)

# ── Energy savings metric cards ───────────────────────────────────────────────
st.markdown('<p class="section-header">💰 Energy Savings at a Glance</p>', unsafe_allow_html=True)

avg_energy = df.groupby("batch_size")["energy_kwh"].mean()
b1_energy = avg_energy.get(1, 1)

col1, col2, col3 = st.columns(3)

for col, bs in zip([col1, col2, col3], BATCH_SIZE_ORDER):
    energy_wh = avg_energy.get(bs, 0) * 1000
    saving = (1 - avg_energy.get(bs, 0) / b1_energy) * 100
    color = BATCH_SIZE_COLORS.get(bs, "#ffffff")

    with col:
        saving_text = f"↓ {saving:.0f}% vs Batch=1" if bs != 1 else "Baseline"
        st.markdown(
            f"""<div class="quant-card">
                <div class="value" style="color: {color};">{energy_wh:.3f} Wh</div>
                <div class="label">Batch {bs}</div>
                <div class="saving">{saving_text}</div>
            </div>""",
            unsafe_allow_html=True,
        )

# ── Energy comparison chart ───────────────────────────────────────────────────
st.markdown('<p class="section-header">🔋 Energy and Latency by Batch Size</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    fig_energy = bar_batch_size_energy(df)
    st.plotly_chart(fig_energy, use_container_width=True)

with col2:
    fig_latency = bar_batch_size_latency(df)
    st.plotly_chart(fig_latency, use_container_width=True)

# ── Insight ───────────────────────────────────────────────────────────────────
b4_saving = (1 - avg_energy.get(4, 0) / b1_energy) * 100
st.markdown(
    f"""<div class="insight-box">
        <p>🌱 Increasing the batch size from 1 to 4 reduces energy consumption
        by approximately <strong>{b4_saving:.0f}%</strong>. Batching efficiently uses the GPU resources!</p>
    </div>""",
    unsafe_allow_html=True,
)

# ── Detailed comparison table ─────────────────────────────────────────────────
st.markdown('<p class="section-header">📋 Detailed Comparison</p>', unsafe_allow_html=True)

comparison = (
    df.groupby("batch_size")
    .agg(
        Avg_Energy_Wh=("energy_kwh", lambda x: round(x.mean() * 1000, 4)),
        Avg_CO2_g=("co2_grams", lambda x: round(x.mean(), 4)),
        Avg_Latency_s=("duration_s", lambda x: round(x.mean(), 2)),
    )
    .reindex(BATCH_SIZE_ORDER)
    .reset_index()
    .rename(columns={"batch_size": "Batch Size", "Avg_Energy_Wh": "Avg Energy (Wh)", "Avg_CO2_g": "Avg CO₂ (g)", "Avg_Latency_s": "Avg Latency (s)"})
)

# Add savings column
comparison["Energy Saved vs Batch=1"] = comparison.apply(
    lambda row: "—"
    if row["Batch Size"] == 1
    else f"{(1 - row['Avg Energy (Wh)'] / comparison.iloc[0]['Avg Energy (Wh)']) * 100:.0f}%",
    axis=1,
)

st.dataframe(comparison, use_container_width=True, hide_index=True)

# ── Takeaway ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "**Takeaway:** Batching is essentially free efficiency when throughput is high. "
    "Amortizing the costs across multiple queries directly lowers energy per query."
)
