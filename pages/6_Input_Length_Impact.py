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

st.set_page_config(page_title="Input Length Impact", page_icon="📏", layout="wide")

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
    return pd.read_csv(os.path.join(data_dir, "input_length_results.csv"))

df = load_data()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 📏 Input Length Impact")
st.markdown(
    "How does the length of the prompt affect energy consumption? "
    "Here we analyze energy per token across different input lengths."
)

# ── Key finding cards ───────────────────────────────────────────────
st.markdown('<p class="section-header">📊 Energy by Input Length Category</p>', unsafe_allow_html=True)

# Calculate energy per token
if 'energy_wh' not in df.columns:
    df['energy_wh'] = df['energy_kwh'] * 1000
if 'energy_per_token_wh' not in df.columns:
    df['energy_per_token_wh'] = df['energy_wh'] / df['input_tokens']

avg_energy = df.groupby("input_category")["energy_wh"].mean()
avg_energy_token = df.groupby("input_category")["energy_per_token_wh"].mean()

col1, col2, col3 = st.columns(3)

for col, category in zip([col1, col2, col3], INPUT_LENGTH_ORDER):
    energy = avg_energy.get(category, 0)
    energy_token = avg_energy_token.get(category, 0)
    color = INPUT_LENGTH_COLORS.get(category, "#ffffff")

    with col:
        st.markdown(
            f"""<div class="quant-card">
                <div class="value" style="color: {color};">{energy:.3f} Wh</div>
                <div class="label">Total Avg Energy ({category})</div>
                <div class="saving">{energy_token:.6f} Wh / token</div>
            </div>""",
            unsafe_allow_html=True,
        )

# ── Charts ───────────────────────────────────────────────────
st.markdown('<p class="section-header">📉 Input Tokens vs Energy</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    fig_scatter = scatter_input_length_energy(df)
    st.plotly_chart(fig_scatter, use_container_width=True)

with col2:
    fig_bar = bar_energy_per_input_token(df)
    st.plotly_chart(fig_bar, use_container_width=True)

# ── Insight ───────────────────────────────────────────────────────────────────
st.markdown(
    f"""<div class="insight-box">
        <p>Energy per input token DECREASES by ~70% from short to medium inputs — longer prompts amortize the fixed costs of model loading and computation setup.</p>
    </div>""",
    unsafe_allow_html=True,
)

# ── Summary table ─────────────────────────────────────────────────
st.markdown('<p class="section-header">📋 Summary by Length Category</p>', unsafe_allow_html=True)

comparison = (
    df.groupby("input_category")
    .agg(
        Avg_Input_Tokens=("input_tokens", lambda x: round(x.mean(), 1)),
        Avg_Energy_Wh=("energy_wh", lambda x: round(x.mean(), 4)),
        Energy_per_Input_Token_Wh=("energy_per_token_wh", lambda x: round(x.mean(), 6)),
        Avg_Tokens_Generated=("output_tokens", lambda x: round(x.mean(), 1)),
    )
    .reindex(INPUT_LENGTH_ORDER)
    .reset_index()
    .rename(columns={
        "input_category": "Input Length",
        "Avg_Input_Tokens": "Avg Input Tokens",
        "Avg_Energy_Wh": "Avg Energy (Wh)",
        "Energy_per_Input_Token_Wh": "Energy per Input Token (Wh)",
        "Avg_Tokens_Generated": "Avg Tokens Generated"
    })
)

st.dataframe(comparison, use_container_width=True, hide_index=True)

# ── Takeaway ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "**Takeaway:** Longer inputs are more energy-efficient per token processed — this means batching context together rather than making multiple small queries can save energy."
)
