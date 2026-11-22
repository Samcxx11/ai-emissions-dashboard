"""
Chapter 06 — Context Window & Prompt Length Dynamics

Evaluating quadratic attention prefill overhead, token ingestion energetics,
and context scaling across Short, Medium, and Long prompt inputs.
"""

import streamlit as st
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.charts import scatter_input_length_energy, bar_energy_per_input_token
from utils.constants import INPUT_LENGTH_COLORS, INPUT_LENGTH_ORDER
from utils.theme import EDITORIAL_CSS, chapter, divider, finding

st.set_page_config(
    page_title="Context Window Dynamics",
    layout="wide",
)

# ── Apply Editorial Theme ─────────────────────────────────────────────────────
st.markdown(EDITORIAL_CSS, unsafe_allow_html=True)


# ── Load Data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    return pd.read_csv(os.path.join(data_dir, "input_length_results.csv"))


df = load_data()


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    chapter(6, "CONTEXT DYNAMICS", "Quadratic prefill overhead vs. linear autoregressive decoding"),
    unsafe_allow_html=True,
)
st.markdown("<h1>Context Window and Prompt Length Dynamics</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='hero-subtitle'>"
    "Deconstructing inference energy into the parallel prefill phase (prompt ingestion) and the sequential "
    "decode phase (token generation) across Short, Medium, and Long input prompt distributions on Llama-2-7B."
    "</p>",
    unsafe_allow_html=True,
)


# ── Technical Background ──────────────────────────────────────────────────────
with st.expander("Technical Background: Prefill vs. Decode Phase Mechanics", expanded=False):
    st.markdown(
        """
        Transformer inference operates across two mathematically distinct phases:
        
        - **Prefill Phase (Prompt Ingestion):** Processes all input prompt tokens concurrently. Scaled by self-attention matrix multiplication ($O(N^2)$), verbose prompts, system instructions, and bloated RAG contexts consume disproportionate initial GPU memory bandwidth and power.
        - **Decode Phase (Autoregressive Generation):** Produces output response tokens one by one ($O(N)$), bound primarily by memory bandwidth rather than compute density.
        """
    )


# ── Metrics by Input Length Category ─────────────────────────────────────────
st.markdown(
    "<div class='chapter-marker'>CONTEXT REGIMES</div>"
    "<h3>Energetics across Prompt Length Categories</h3>",
    unsafe_allow_html=True,
)

summary_cat = df.groupby("input_length_category").agg(
    avg_energy_kwh=("energy_kwh", "mean"),
    avg_tokens=("input_tokens", "mean"),
    avg_co2=("co2_grams", "mean"),
).reindex(INPUT_LENGTH_ORDER).reset_index()

cols = st.columns(3)

for i, row in summary_cat.iterrows():
    cat = row["input_length_category"]
    e_wh = row["avg_energy_kwh"] * 1000
    color = INPUT_LENGTH_COLORS[cat]
    with cols[i]:
        st.markdown(
            f"""
            <div class="metric-editorial">
                <div style="font-family: var(--mono); font-size: 11px; text-transform: uppercase; color: var(--graphite); margin-bottom: 0.5rem;">
                    {cat} Prompt Context
                </div>
                <div class="metric-number" style="color: {color}; font-size: 2.2rem;">
                    {e_wh:.3f} <span style="font-size: 1.1rem; color: var(--graphite);">Wh</span>
                </div>
                <div class="metric-unit">~{row['avg_tokens']:,.0f} Input Tokens · {row['avg_co2']:.3f} g CO2e</div>
                <div class="metric-desc">Mean electrical energy required to ingest context and generate output.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ── Visualizations ───────────────────────────────────────────────────────────
st.markdown(divider(), unsafe_allow_html=True)

st.markdown(
    "<div class='chapter-marker'>TELEMETRY CHARTS</div>"
    "<h3>Prompt Length vs. Energy & Energy Per Input Token</h3>",
    unsafe_allow_html=True,
)

c1, c2 = st.columns(2)

with c1:
    fig_scatter_in = scatter_input_length_energy(df)
    st.plotly_chart(fig_scatter_in, use_container_width=True)

with c2:
    fig_tok_in = bar_energy_per_input_token(df)
    st.plotly_chart(fig_tok_in, use_container_width=True)


# ── Mitigation Takeaway ───────────────────────────────────────────────────────
st.markdown(divider(), unsafe_allow_html=True)

st.markdown(
    finding(
        "Long prompts with bloated system instructions and excessive RAG passages increase initial prefill energy draw significantly. "
        "Adopting <strong>prompt token compression techniques</strong> (removing conversational boilerplate, using concise system guidelines, and dynamic context trimming) "
        "yields immediate 10% to 25% energy savings prior to token generation.",
        label="ENGINEERING PROTOCOL",
    ),
    unsafe_allow_html=True,
)
