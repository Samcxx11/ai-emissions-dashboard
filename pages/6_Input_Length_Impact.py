"""
Page 6 — Input Length Impact
"""

import streamlit as st
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.charts import scatter_input_length_energy, bar_energy_per_input_token
from utils.constants import INPUT_LENGTH_COLORS, INPUT_LENGTH_ORDER
from utils.theme import EDITORIAL_CSS, apply_editorial_layout, chapter, divider, finding, quote, metric_card, big_number

st.set_page_config(page_title="Input Length Impact", page_icon="📏", layout="wide")
st.markdown(EDITORIAL_CSS, unsafe_allow_html=True)



# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    return pd.read_csv(os.path.join(data_dir, "input_length_results.csv"))

df = load_data()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="editorial-hero">', unsafe_allow_html=True)
st.markdown(big_number(6), unsafe_allow_html=True)
st.markdown('<h1><em>Input Length</em> Impact</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">How does the length of the prompt affect energy consumption? Here we analyze energy per token across different input lengths.</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ── Key finding cards ───────────────────────────────────────────────
st.markdown(chapter(1, "ENERGY BY INPUT LENGTH CATEGORY"), unsafe_allow_html=True)

# Calculate energy per token
if 'energy_wh' not in df.columns:
    df['energy_wh'] = df['energy_kwh'] * 1000
if 'energy_per_token_wh' not in df.columns:
    df['energy_per_token_wh'] = df['energy_wh'] / df['input_tokens']

avg_energy = df.groupby("input_length_category")["energy_wh"].mean()
avg_energy_token = df.groupby("input_length_category")["energy_per_token_wh"].mean()

col1, col2, col3 = st.columns(3)

for col, category in zip([col1, col2, col3], INPUT_LENGTH_ORDER):
    energy = avg_energy.get(category, 0)
    energy_token = avg_energy_token.get(category, 0)
    color = INPUT_LENGTH_COLORS.get(category, "#ffffff")

    with col:
        st.markdown(metric_card(f"{energy:.3f}", "Wh", f"Total Avg Energy ({category})\n{energy_token:.6f} Wh/token"), unsafe_allow_html=True)

# ── Charts ───────────────────────────────────────────────────
st.markdown(divider(), unsafe_allow_html=True)
st.markdown(chapter(2, "INPUT TOKENS VS ENERGY"), unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    fig_scatter = scatter_input_length_energy(df)
    st.plotly_chart(apply_editorial_layout(fig_scatter), use_container_width=True)

with col2:
    fig_bar = bar_energy_per_input_token(df)
    st.plotly_chart(apply_editorial_layout(fig_bar), use_container_width=True)

# ── Insight ───────────────────────────────────────────────────────────────────
st.markdown(finding(r"Energy per input token DECREASES by ~70% from short to medium inputs — longer prompts amortize the fixed costs of model loading and computation setup.", "EFFICIENCY INSIGHT"), unsafe_allow_html=True)

# ── Summary table ─────────────────────────────────────────────────
st.markdown(divider(), unsafe_allow_html=True)
st.markdown(chapter(3, "SUMMARY BY LENGTH CATEGORY"), unsafe_allow_html=True)

comparison = (
    df.groupby("input_length_category")
    .agg(
        Avg_Input_Tokens=("input_tokens", lambda x: round(x.mean(), 1)),
        Avg_Energy_Wh=("energy_wh", lambda x: round(x.mean(), 4)),
        Energy_per_Input_Token_Wh=("energy_per_token_wh", lambda x: round(x.mean(), 6)),
        Avg_Tokens_Generated=("tokens_generated", lambda x: round(x.mean(), 1)),
    )
    .reindex(INPUT_LENGTH_ORDER)
    .reset_index()
    .rename(columns={
        "input_length_category": "Input Length",
        "Avg_Input_Tokens": "Avg Input Tokens",
        "Avg_Energy_Wh": "Avg Energy (Wh)",
        "Energy_per_Input_Token_Wh": "Energy per Input Token (Wh)",
        "Avg_Tokens_Generated": "Avg Tokens Generated"
    })
)

st.dataframe(comparison, use_container_width=True, hide_index=True)

# ── Takeaway ──────────────────────────────────────────────────────────────────
st.markdown(divider(), unsafe_allow_html=True)
st.markdown(chapter(4, "THE BOTTOM LINE"), unsafe_allow_html=True)
st.markdown(quote("Longer inputs are more energy-efficient per token processed — this means batching context together rather than making multiple small queries can save energy.", "Key Takeaway"), unsafe_allow_html=True)
