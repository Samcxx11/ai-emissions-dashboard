"""
Chapter 05 — Batch Sizing & GPU Saturation

Evaluating tensor core utilization, static power dissipation, and the
per-query energy amortization curve across batch sizes N in {1, 2, 4}.
"""

import streamlit as st
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.charts import bar_batch_size_energy, bar_batch_size_latency
from utils.constants import BATCH_SIZE_COLORS, BATCH_SIZE_ORDER
from utils.theme import EDITORIAL_CSS, chapter, divider, finding

st.set_page_config(
    page_title="Batch Sizing & Saturation",
    layout="wide",
)

# ── Apply Editorial Theme ─────────────────────────────────────────────────────
st.markdown(EDITORIAL_CSS, unsafe_allow_html=True)


# ── Load Data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    return pd.read_csv(os.path.join(data_dir, "batch_size_results.csv"))


df = load_data()


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    chapter(5, "CONCURRENT SCHEDULING", "Hardware saturation and baseline power amortization"),
    unsafe_allow_html=True,
)
st.markdown("<h1>Batch Sizing and Hardware Saturation</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='hero-subtitle'>"
    "Investigating how dynamic request batching ($N = 1, 2, 4$) amortizes static GPU idle power "
    "and increases tensor execution density on Llama-2-7B."
    "</p>",
    unsafe_allow_html=True,
)


# ── Technical Background ──────────────────────────────────────────────────────
with st.expander("Technical Background: The Static Power Amortization Curve", expanded=False):
    st.markdown(
        """
        Modern accelerator boards (NVIDIA A100 / RTX 4090) consume substantial **static baseline power** 
        (transistor leakage and clock distribution) simply by remaining energized in memory, regardless of computational load:
        
        - **Batch Size = 1 (Single Interactive Query):** Memory bandwidth remains underutilized while static baseline power runs continuously during sequential decoding, driving up the per-query energetic footprint.
        - **Batch Size = 4 (Concurrent Batched Pipeline):** Matrix multiplication kernels fully saturate tensor execution pipelines, amortizing the fixed baseline draw across four concurrent sequences simultaneously.
        """
    )


# ── Batch Sizing Metrics ──────────────────────────────────────────────────────
st.markdown(
    "<div class='chapter-marker'>PERFORMANCE BENCHMARKS</div>"
    "<h3>Per-Query Energy by Concurrency Level</h3>",
    unsafe_allow_html=True,
)

avg_batch = df.groupby("batch_size")["energy_kwh"].mean()
b1_energy = avg_batch.get(1, 1)

col1, col2, col3 = st.columns(3)

for col, b_size in zip([col1, col2, col3], BATCH_SIZE_ORDER):
    e_wh = avg_batch.get(b_size, 0) * 1000
    saving = (1 - avg_batch.get(b_size, 0) / b1_energy) * 100
    color = BATCH_SIZE_COLORS[b_size]

    with col:
        saving_text = f"Amortization: -{saving:.1f}% vs. N=1" if b_size != 1 else "Baseline Single User"
        st.markdown(
            f"""
            <div class="metric-editorial">
                <div style="font-family: var(--mono); font-size: 11px; text-transform: uppercase; color: var(--graphite); margin-bottom: 0.5rem;">
                    Batch Size N = {b_size}
                </div>
                <div class="metric-number" style="color: {color}; font-size: 2.2rem;">
                    {e_wh:.3f} <span style="font-size: 1.1rem; color: var(--graphite);">Wh</span>
                </div>
                <div class="metric-unit">{saving_text}</div>
                <div class="metric-desc">Normalized average energy consumed per individual query response.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ── Visualizations ───────────────────────────────────────────────────────────
st.markdown(divider(), unsafe_allow_html=True)

st.markdown(
    "<div class='chapter-marker'>TELEMETRY CHARTS</div>"
    "<h3>Energetics and Per-Query Processing Latency</h3>",
    unsafe_allow_html=True,
)

c1, c2 = st.columns(2)

with c1:
    fig_b_energy = bar_batch_size_energy(df)
    st.plotly_chart(fig_b_energy, use_container_width=True)

with c2:
    fig_b_lat = bar_batch_size_latency(df)
    st.plotly_chart(fig_b_lat, use_container_width=True)


# ── Analytical Takeaway ───────────────────────────────────────────────────────
st.markdown(divider(), unsafe_allow_html=True)

st.markdown(
    finding(
        "Increasing batch execution concurrency from <strong>N = 1 to N = 4 achieves a 21.4% reduction in per-query energy consumption</strong>. "
        "While aggregate forward-pass latency increases modestly, token generation throughput scales significantly faster than power draw. "
        "Production inference serving should mandate micro-batch scheduling windows (10–50 ms wait thresholds) for non-streaming queries.",
        label="DEPLOYMENT FINDING",
    ),
    unsafe_allow_html=True,
)
