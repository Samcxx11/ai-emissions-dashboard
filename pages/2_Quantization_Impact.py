"""
Page 2 — Quantization Impact

Shows how quantizing Llama-2-7B (Real) from FP16 → INT8 → INT4 affects energy
consumption, CO₂ emissions, GPU memory, and response quality.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import streamlit as st
import pandas as pd
from utils.theme import EDITORIAL_CSS, apply_editorial_layout, chapter, divider, finding, quote, metric_card, big_number

from utils.charts import (
    bar_quantization_energy,
    bar_quantization_co2,
    line_quality_vs_precision,
)
from utils.constants import QUANTIZATION_COLORS, QUANT_ORDER

st.set_page_config(page_title="Quantization Impact", page_icon="🔧", layout="wide")

st.markdown(EDITORIAL_CSS, unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    return pd.read_csv(os.path.join(data_dir, "quantization_comparison.csv"))

df = load_data()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(chapter(1, "Quantization Impact"), unsafe_allow_html=True)
st.markdown(
    "What happens when we compress an AI model to use less GPU memory? "
    "Here we test **Llama-2-7B (Real)** at three precision levels: FP16 (full), "
    "INT8 (medium compression), and INT4 (heavy compression)."
)

# ── What is quantization? ─────────────────────────────────────────────────────
with st.expander("🧠 What is quantization? (click to expand)", expanded=False):
    st.markdown("""
    **Quantization** shrinks a model by storing its weights in fewer bits —
    e.g., going from 16-bit precision (FP16) down to 4-bit (INT4).

    Same model, same architecture, but a much lighter "physical" footprint
    in GPU memory.

    | Precision | Bits per Weight | GPU Memory (7B model) | Trade-off |
    |-----------|:-:|:-:|---|
    | **FP16** | 16 | ~14 GB | Full quality, full speed |
    | **INT8** | 8 | ~7 GB | Small quality drop, half memory |
    | **INT4** | 4 | ~3.5 GB | Noticeable quality drop, quarter memory |

    **⚠️ Common Misconception:** Many people assume quantization saves energy.
    Our real benchmark data shows the opposite — quantized models actually use
    **MORE** energy due to dequantization overhead. But they use much **LESS memory**,
    making big models accessible on cheaper hardware.
    """)

st.markdown(divider(), unsafe_allow_html=True)

# ── GPU Memory Savings (The REAL benefit) ─────────────────────────────────────
st.markdown(chapter(2, "GPU Memory Savings (The Real Benefit)"), unsafe_allow_html=True)

st.markdown(
    "Quantization's **primary advantage** is reducing GPU memory usage, "
    "making expensive models runnable on cheaper hardware."
)

# Memory data (based on real Llama-2-7B measurements)
memory_data = {"FP16": 14030, "INT8": 7574, "INT4": 4820}
fp16_mem = memory_data["FP16"]

col1, col2, col3 = st.columns(3)
for col, precision in zip([col1, col2, col3], QUANT_ORDER):
    mem = memory_data[precision]
    mem_saving = ((fp16_mem - mem) / fp16_mem) * 100
    
    with col:
        saving_text = f"↓ {mem_saving:.0f}% memory saved" if precision != "FP16" else "Baseline"
        saving_class = "green" if precision != "FP16" else ""
        st.markdown(
            metric_card(f"{mem/1000:.1f}", "GB", f"{precision} — {saving_text}", saving_class),
            unsafe_allow_html=True,
        )

st.markdown(
    finding(
        "INT4 quantization reduces GPU memory by ~66%. This means a 7B model that normally needs a ₹5 lakh GPU (A100, 80GB) can now run on a ₹50,000 GPU (RTX 3060, 8GB). Quantization is an accessibility tool, not an energy-saving tool.",
        "KEY TAKEAWAY",
    ),
    unsafe_allow_html=True,
)

st.markdown(divider(), unsafe_allow_html=True)

# ── Energy & CO₂ (The Counterintuitive Finding) ──────────────────────────────
st.markdown(chapter(3, "Energy & CO₂ Impact (Counterintuitive Finding!)"), unsafe_allow_html=True)

avg_energy = df.groupby("precision")["energy_kwh"].mean()
fp16_energy = avg_energy.get("FP16", 1)

# Show energy INCREASE cards
col1, col2, col3 = st.columns(3)
for col, precision in zip([col1, col2, col3], QUANT_ORDER):
    energy_wh = avg_energy.get(precision, 0) * 1000
    change = ((avg_energy.get(precision, 0) / fp16_energy) - 1) * 100
    
    with col:
        if precision == "FP16":
            change_text = "Baseline"
            change_class = ""
        else:
            change_text = f"↑ {change:.0f}% MORE energy"
            change_class = "red"
        st.markdown(
            metric_card(f"{energy_wh:.3f}", "Wh", f"{precision} — {change_text}", change_class),
            unsafe_allow_html=True,
        )

# Charts
col1, col2 = st.columns(2)
with col1:
    fig_energy = bar_quantization_energy(df)
    fig_energy = apply_editorial_layout(fig_energy)
    st.plotly_chart(fig_energy, use_container_width=True)
with col2:
    fig_co2 = bar_quantization_co2(df)
    fig_co2 = apply_editorial_layout(fig_co2)
    st.plotly_chart(fig_co2, use_container_width=True)

# Warning insight
int4_increase = ((avg_energy.get("INT4", 0) / fp16_energy) - 1) * 100
st.markdown(
    finding(
        f"Quantizing Llama-2-7B from FP16 to INT4 actually INCREASES energy consumption by ~{int4_increase:.0f}%! This happens because the GPU must perform extra dequantization math at every layer — converting compressed 4-bit weights back to 16-bit for computation. This overhead increases both latency and total energy consumed.",
        "COUNTERINTUITIVE FINDING",
        warning=True,
    ),
    unsafe_allow_html=True,
)

st.markdown(divider(), unsafe_allow_html=True)

# ── Quality trade-off ─────────────────────────────────────────────────────────
st.markdown(
    chapter(4, "The Trade-off: Quality vs Memory Savings"),
    unsafe_allow_html=True,
)

st.markdown(
    "Besides increased energy, quantization also affects response quality. "
    "This chart shows the average quality score (0–100) at each precision level."
)

fig_quality = line_quality_vs_precision(df)
fig_quality = apply_editorial_layout(fig_quality)
st.plotly_chart(fig_quality, use_container_width=True)

# Quality drop stats
avg_quality = df.groupby("precision")["quality_score"].mean()
fp16_quality = avg_quality.get("FP16", 100)
int8_drop = fp16_quality - avg_quality.get("INT8", 0)
int4_drop = fp16_quality - avg_quality.get("INT4", 0)

col1, col2 = st.columns(2)
with col1:
    st.markdown(
        metric_card(f"-{int8_drop:.1f}", "points", "INT8 quality drop vs FP16", "red"),
        unsafe_allow_html=True
    )
with col2:
    st.markdown(
        metric_card(f"-{int4_drop:.1f}", "points", "INT4 quality drop vs FP16", "red"),
        unsafe_allow_html=True
    )

st.markdown(divider(), unsafe_allow_html=True)

# ── Detailed comparison table ─────────────────────────────────────────────────
st.markdown(chapter(5, "Detailed Comparison"), unsafe_allow_html=True)

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

# GPU Memory column
comparison["GPU_Memory_GB"] = comparison["Precision"].map(
    {"FP16": "14.0 GB", "INT8": "7.6 GB", "INT4": "4.8 GB"}
)

# Energy change column (shows increase, not savings)
comparison["Energy Change vs FP16"] = comparison.apply(
    lambda row: "— (Baseline)"
    if row["Precision"] == "FP16"
    else f"+{((row['Avg_Energy_Wh'] / comparison.iloc[0]['Avg_Energy_Wh']) - 1) * 100:.0f}% (more energy)",
    axis=1,
)

# Memory savings column
comparison["Memory Saved vs FP16"] = comparison.apply(
    lambda row: "— (Baseline)"
    if row["Precision"] == "FP16"
    else f"-{((memory_data['FP16'] - memory_data[row['Precision']]) / memory_data['FP16']) * 100:.0f}% ✅",
    axis=1,
)

st.dataframe(comparison, use_container_width=True, hide_index=True)

st.markdown(divider(), unsafe_allow_html=True)

# ── Takeaway ──────────────────────────────────────────────────────────────────
st.markdown(chapter(6, "Takeaway"), unsafe_allow_html=True)

st.markdown(
    """
    Our benchmark reveals a **counterintuitive truth** about quantization:

    | What People Think | What Our Data Shows |
    |---|---|
    | Quantization saves energy ❌ | Quantization **increases** energy by 45-92% |
    | Quantization makes AI faster ❌ | Quantization **increases** latency due to dequantization overhead |
    | Quantization saves memory ✅ | **YES!** INT4 uses 66% less GPU memory |
    | Quantization reduces quality ✅ | Quality drops by ~9 points (FP16 → INT4) |
    """
)

st.markdown(
    quote(
        "Quantization is NOT an energy-saving technique — it's an accessibility technique. It allows large AI models to run on cheaper, smaller GPUs that couldn't otherwise fit the model in memory. Use FP16 when you have the hardware; use INT4 only when you need to fit a model on limited hardware.",
        "Bottom Line"
    ),
    unsafe_allow_html=True
)
