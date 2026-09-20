"""
Page 5 — Batch Size Analysis
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import streamlit as st
import pandas as pd
from utils.theme import EDITORIAL_CSS, apply_editorial_layout, chapter, divider, finding, quote, metric_card, big_number
from utils.charts import bar_batch_size_energy, bar_batch_size_latency
from utils.constants import BATCH_SIZE_COLORS, BATCH_SIZE_ORDER

st.set_page_config(page_title="Batch Size Analysis", page_icon="📦", layout="wide")

st.markdown(EDITORIAL_CSS, unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    return pd.read_csv(os.path.join(data_dir, "batch_size_results.csv"))

df = load_data()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(big_number(5), unsafe_allow_html=True)
st.markdown("# Batch Size Analysis")
st.markdown(
    "How does batching multiple queries affect energy consumption? "
    "Here we test **Llama-2-7B (Real)** at batch sizes 1, 2, and 4."
)

# ── What is batching? ─────────────────────────────────────────────────────────
st.markdown(quote(
    "Batching means running multiple queries together instead of one at a time. "
    "It is like washing clothes in bulk vs one at a time — it's far more efficient to load the washing machine up! "
    "In AI inference, doing the heavy computation on multiple inputs at once amortizes the overhead of memory transfers.",
    "What is Batching?"
), unsafe_allow_html=True)

# ── Energy savings metric cards ───────────────────────────────────────────────
st.markdown(divider(), unsafe_allow_html=True)
st.markdown(chapter(1, "Energy Savings at a Glance"), unsafe_allow_html=True)

avg_energy = df.groupby("batch_size")["energy_kwh"].mean()
b1_energy = avg_energy.get(1, 1)

col1, col2, col3 = st.columns(3)

for col, bs in zip([col1, col2, col3], BATCH_SIZE_ORDER):
    energy_wh = avg_energy.get(bs, 0) * 1000
    saving = (1 - avg_energy.get(bs, 0) / b1_energy) * 100
    color_class = "green" if bs != 1 else ""

    with col:
        saving_text = f"↓ {saving:.0f}% vs Batch=1" if bs != 1 else "Baseline"
        st.markdown(
            metric_card(
                value=f"{energy_wh:.3f}",
                unit="Wh",
                description=f"Batch {bs}<br/>{saving_text}",
                color_class=color_class
            ),
            unsafe_allow_html=True,
        )

# ── Energy comparison chart ───────────────────────────────────────────────────
st.markdown(divider(), unsafe_allow_html=True)
st.markdown(chapter(2, "Energy and Latency by Batch Size"), unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    fig_energy = bar_batch_size_energy(df)
    fig_energy = apply_editorial_layout(fig_energy)
    st.plotly_chart(fig_energy, use_container_width=True)

with col2:
    fig_latency = bar_batch_size_latency(df)
    fig_latency = apply_editorial_layout(fig_latency)
    st.plotly_chart(fig_latency, use_container_width=True)

# ── Insight ───────────────────────────────────────────────────────────────────
b4_saving = (1 - avg_energy.get(4, 0) / b1_energy) * 100
st.markdown(
    finding(
        f"Increasing the batch size from 1 to 4 reduces energy consumption "
        f"by approximately <strong>{b4_saving:.0f}%</strong>. Batching efficiently uses the GPU resources!",
        "KEY INSIGHT"
    ),
    unsafe_allow_html=True,
)

# ── Detailed comparison table ─────────────────────────────────────────────────
st.markdown(divider(), unsafe_allow_html=True)
st.markdown(chapter(3, "Detailed Comparison"), unsafe_allow_html=True)

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
st.markdown(divider(), unsafe_allow_html=True)
st.markdown(
    quote(
        "Batching is essentially free efficiency when throughput is high. "
        "Amortizing the costs across multiple queries directly lowers energy per query.",
        "Takeaway"
    ),
    unsafe_allow_html=True
)
