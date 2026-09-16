"""
Page 2 — Quantization Impact

Shows how quantizing Llama-2-7B (Real) from FP16 → INT8 → INT4 affects energy
consumption, CO₂ emissions, and response quality.
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

st.set_page_config(page_title="Quantization Impact", page_icon="🔧", layout="wide")

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
    return pd.read_csv(os.path.join(data_dir, "quantization_comparison.csv"))

df = load_data()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🔧 Quantization Impact")
st.markdown(
    "How much energy can we save by compressing the same model? "
    "Here we test **Llama-2-7B (Real)** at three precision levels: FP16 (full), "
    "INT8 (medium compression), and INT4 (heavy compression)."
)

# ── What is quantization? ─────────────────────────────────────────────────────
with st.expander("🧠 What is quantization? (click to expand)", expanded=False):
    st.markdown("""
    **Quantization** shrinks a model by storing its weights in fewer bits —
    e.g., going from 16-bit precision (FP16) down to 4-bit (INT4).

    Same model, same architecture, but a much lighter "physical" footprint
    in memory and compute. It's the AI equivalent of **compressing a photo** —
    some quality loss, big size savings.

    | Precision | Bits per Weight | Memory (7B model) | Trade-off |
    |-----------|:-:|:-:|---|
    | **FP16** | 16 | ~14 GB | Full quality, full cost |
    | **INT8** | 8 | ~7 GB | Small quality drop, big savings |
    | **INT4** | 4 | ~3.5 GB | Noticeable quality drop, huge savings |

    This matters because companies quantize models specifically to cut
    inference costs — so we're testing what's actually deployed in production.
    """)

# ── Energy savings metric cards ───────────────────────────────────────────────
st.markdown('<p class="section-header">💰 Energy Savings at a Glance</p>', unsafe_allow_html=True)

avg_energy = df.groupby("precision")["energy_kwh"].mean()
fp16_energy = avg_energy.get("FP16", 1)

col1, col2, col3 = st.columns(3)

for col, precision in zip([col1, col2, col3], QUANT_ORDER):
    energy_wh = avg_energy.get(precision, 0) * 1000
    saving = (1 - avg_energy.get(precision, 0) / fp16_energy) * 100
    color = QUANTIZATION_COLORS[precision]

    with col:
        saving_text = f"↓ {saving:.0f}% vs FP16" if precision != "FP16" else "Baseline"
        st.markdown(
            f"""<div class="quant-card">
                <div class="value" style="color: {color};">{energy_wh:.3f} Wh</div>
                <div class="label">{precision}</div>
                <div class="saving">{saving_text}</div>
            </div>""",
            unsafe_allow_html=True,
        )

# ── Energy comparison chart ───────────────────────────────────────────────────
st.markdown('<p class="section-header">🔋 Energy by Precision Level</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    fig_energy = bar_quantization_energy(df)
    st.plotly_chart(fig_energy, use_container_width=True)

with col2:
    fig_co2 = bar_quantization_co2(df)
    st.plotly_chart(fig_co2, use_container_width=True)

# ── Insight ───────────────────────────────────────────────────────────────────
int4_saving = (1 - avg_energy.get("INT4", 0) / fp16_energy) * 100
st.markdown(
    f"""<div class="insight-box">
        <p>🌱 Quantizing Llama-2-7B (Real) from FP16 to INT4 reduces energy consumption
        by approximately <strong>{int4_saving:.0f}%</strong>. At scale — say 1 million
        queries per day — that's the difference between powering a small office
        and powering a single light bulb.</p>
    </div>""",
    unsafe_allow_html=True,
)

# ── Quality trade-off ─────────────────────────────────────────────────────────
st.markdown(
    '<p class="section-header">⚖️ The Trade-off: Quality vs Efficiency</p>',
    unsafe_allow_html=True,
)

st.markdown(
    "The critical question: **does compressing the model hurt response quality?** "
    "This chart shows the average quality score (0–100) at each precision level."
)

fig_quality = line_quality_vs_precision(df)
st.plotly_chart(fig_quality, use_container_width=True)

# Quality drop stats
avg_quality = df.groupby("precision")["quality_score"].mean()
fp16_quality = avg_quality.get("FP16", 100)
int8_drop = fp16_quality - avg_quality.get("INT8", 0)
int4_drop = fp16_quality - avg_quality.get("INT4", 0)

col1, col2 = st.columns(2)
with col1:
    st.metric(
        "INT8 quality drop vs FP16",
        f"{int8_drop:.1f} points",
        delta=f"-{int8_drop:.1f}",
        delta_color="inverse",
    )
with col2:
    st.metric(
        "INT4 quality drop vs FP16",
        f"{int4_drop:.1f} points",
        delta=f"-{int4_drop:.1f}",
        delta_color="inverse",
    )

st.markdown(
    f"""<div class="insight-box">
        <p>🎯 INT8 quantization drops quality by only <strong>{int8_drop:.1f} points</strong>
        while saving significant energy — making it the "sweet spot" for most tasks.
        INT4 saves more energy but the <strong>{int4_drop:.1f}-point quality drop</strong>
        may matter for tasks requiring high accuracy.</p>
    </div>""",
    unsafe_allow_html=True,
)

# ── Detailed comparison table ─────────────────────────────────────────────────
st.markdown('<p class="section-header">📋 Detailed Comparison</p>', unsafe_allow_html=True)

comparison = (
    df.groupby("precision")
    .agg(
        Avg_Energy_Wh=("energy_kwh", lambda x: round(x.mean() * 1000, 4)),
        Avg_CO2_g=("co2_grams", lambda x: round(x.mean(), 4)),
        Avg_Quality=("quality_score", lambda x: round(x.mean(), 2)),
        Avg_Latency_s=("duration_s", lambda x: round(x.mean(), 2)),
        Queries=("query_id", "count"),
    )
    .reindex(QUANT_ORDER)
    .reset_index()
    .rename(columns={"precision": "Precision"})
)

# Add savings column
comparison["Energy Saved vs FP16"] = comparison.apply(
    lambda row: "—"
    if row["Precision"] == "FP16"
    else f"{(1 - row['Avg_Energy_Wh'] / comparison.iloc[0]['Avg_Energy_Wh']) * 100:.0f}%",
    axis=1,
)

st.dataframe(comparison, use_container_width=True, hide_index=True)

# ── Takeaway ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "**Takeaway:** Quantization is one of the most practical levers for "
    "reducing AI inference emissions. Unlike switching to a smaller model "
    "(which changes capabilities), quantization keeps the same architecture — "
    "you're just running it more efficiently."
)
