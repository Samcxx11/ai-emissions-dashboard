"""
Page 4 — Real-World Impact

Shows what these micro-measurements mean at scale (e.g., thousands or millions of
queries per day) and translates them into understandable metrics like car miles
or smartphone charges. Also provides actionable recommendations.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.constants import (
    MODEL_COLORS,
    MODEL_ORDER,
    CO2_KG_PER_CAR_MILE,
    KWH_PER_SMARTPHONE_CHARGE,
    CO2_KG_PER_TREE_PER_YEAR,
    KWH_PER_HOME_PER_DAY,
)
from utils.theme import EDITORIAL_CSS, apply_editorial_layout, chapter, divider, finding, quote, metric_card, big_number

st.set_page_config(page_title="Real-World Impact", page_icon="🌍", layout="wide")
st.markdown(EDITORIAL_CSS, unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    return pd.read_csv(os.path.join(data_dir, "inference_results.csv"))

df = load_data()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="editorial-hero">', unsafe_allow_html=True)
st.markdown(big_number(4), unsafe_allow_html=True)
st.markdown('<h1><em>Real-World</em> Impact</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">What do our measurements mean at real-world scale? This page translates per-query numbers into tangible impacts.</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ── Causal chain ──────────────────────────────────────────────────────────────
st.markdown(chapter(1, "HOW THIS PROJECT LEADS TO REDUCTION"), unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(metric_card("1. Measure", "", "We track actual energy used per query on hardware"), unsafe_allow_html=True)
with c2:
    st.markdown(metric_card("2. Scale", "", "We project those costs to real-world usage volumes"), unsafe_allow_html=True)
with c3:
    st.markdown(metric_card("3. Expose", "", "We show the hidden environmental impact clearly"), unsafe_allow_html=True)
with c4:
    st.markdown(metric_card("4. Optimize", "", "Developers can choose smaller/efficient models"), unsafe_allow_html=True)

st.markdown(divider(), unsafe_allow_html=True)

# ── Scenario Configuration ────────────────────────────────────────────────────
st.markdown(chapter(2, "SCALING CALCULATOR", "Adjust the slider to see the annual impact of scaling."), unsafe_allow_html=True)

col_select, col_slider = st.columns([1, 2])
with col_select:
    selected_model = st.selectbox("Choose a model:", options=MODEL_ORDER, index=2)
with col_slider:
    queries_per_day = st.slider("Queries per day:", min_value=1_000, max_value=10_000_000, value=100_000, step=1_000, format="%d")

model_data = df[df["model"] == selected_model]
avg_energy_kwh = model_data["energy_kwh"].mean()
avg_co2_g = model_data["co2_grams"].mean()

daily_energy_kwh = avg_energy_kwh * queries_per_day
annual_energy_kwh = daily_energy_kwh * 365
annual_co2_kg = (avg_co2_g * queries_per_day * 365) / 1000

car_miles = annual_co2_kg / CO2_KG_PER_CAR_MILE
smartphone_charges = annual_energy_kwh / KWH_PER_SMARTPHONE_CHARGE
trees_needed = annual_co2_kg / CO2_KG_PER_TREE_PER_YEAR
home_days = annual_energy_kwh / KWH_PER_HOME_PER_DAY

st.markdown(f"### {selected_model} @ {queries_per_day:,} queries/day")
met1, met2, met3 = st.columns(3)
met1.metric("Annual Energy", f"{annual_energy_kwh:,.1f} kWh")
met2.metric("Annual CO₂", f"{annual_co2_kg:,.1f} kg")
met3.metric("Daily Energy", f"{daily_energy_kwh:,.2f} kWh")

st.markdown(divider(), unsafe_allow_html=True)
st.markdown(chapter(3, "WHAT DOES THAT LOOK LIKE?"), unsafe_allow_html=True)

eq1, eq2, eq3, eq4 = st.columns(4)
with eq1:
    st.markdown(metric_card(f"{car_miles:,.0f}", "miles", "Driven by a car"), unsafe_allow_html=True)
with eq2:
    st.markdown(metric_card(f"{smartphone_charges:,.0f}", "charges", "Smartphone charges"), unsafe_allow_html=True)
with eq3:
    st.markdown(metric_card(f"{trees_needed:,.1f}", "trees", "Trees needed to offset (1 yr)"), unsafe_allow_html=True)
with eq4:
    st.markdown(metric_card(f"{home_days:,.1f}", "days", "Powering a US home"), unsafe_allow_html=True)

st.markdown(divider(), unsafe_allow_html=True)

# ── Cross-model scaling comparison ───────────────────────────────────────────
st.markdown(chapter(4, "ANNUAL IMPACT: ALL MODELS COMPARED", f"What if each model served {queries_per_day:,} queries/day for a year?"), unsafe_allow_html=True)

comparison_rows = []
for model in MODEL_ORDER:
    m_data = df[df["model"] == model]
    m_energy = m_data["energy_kwh"].mean() * queries_per_day * 365
    m_co2 = (m_data["co2_grams"].mean() * queries_per_day * 365) / 1000
    comparison_rows.append({"Model": model, "Annual Energy (kWh)": m_energy, "Annual CO₂ (kg)": m_co2})

comp_df = pd.DataFrame(comparison_rows)

fig = go.Figure()
fig.add_trace(go.Bar(
    x=comp_df["Model"],
    y=comp_df["Annual CO₂ (kg)"],
    marker_color=[MODEL_COLORS.get(m, "#3498db") for m in comp_df["Model"]],
    text=[f"{v:,.0f} kg" for v in comp_df["Annual CO₂ (kg)"]],
    textposition="outside",
))
fig.update_layout(
    title=dict(text=f"Annual CO₂ Emissions @ {queries_per_day:,} queries/day"),
    xaxis_title="Model",
    yaxis_title="CO₂ (kg/year)",
)
st.plotly_chart(apply_editorial_layout(fig), use_container_width=True)

smallest_co2 = comp_df["Annual CO₂ (kg)"].min()
largest_co2 = comp_df["Annual CO₂ (kg)"].max()
savings_kg = largest_co2 - smallest_co2

st.markdown(finding(f"Switching from the largest to the smallest model would save **{savings_kg:,.0f} kg CO₂/year** at this query volume — equivalent to **{savings_kg / CO2_KG_PER_CAR_MILE:,.0f} fewer car miles driven**.", "POTENTIAL SAVINGS"), unsafe_allow_html=True)

st.markdown(divider(), unsafe_allow_html=True)

# ── Recommendations ───────────────────────────────────────────────────────────
st.markdown(chapter(5, "ACTIONABLE RECOMMENDATIONS", "Practical steps to minimize your AI footprint."), unsafe_allow_html=True)

st.markdown(
    """
    * **1. Right-Size Your Model:** If a 1B model performs almost as well as a 7B model for simple tasks, use the smaller one.
    * **2. Avoid Overkill:** Many people use large general-purpose models for tasks a small specialized model could handle.
    * **3. Quantize When Possible:** Quantization shrinks the memory footprint drastically.
    * **4. Cache Repeated Queries:** Zero energy for a cached response.
    * **5. Batch Requests:** Running multiple queries together is more energy-efficient per-query.
    * **6. Time-of-Use Awareness:** Running heavy AI workloads when the electricity grid has more renewable energy available reduces the carbon intensity per kWh.
    """
)

st.markdown(divider(), unsafe_allow_html=True)
st.markdown(chapter(6, "THE BOTTOM LINE"), unsafe_allow_html=True)

st.markdown(quote("Developers should default to smaller models when the task doesn't need a large one — that's not a compromise, it's an optimization.", "Best Practice"), unsafe_allow_html=True)

st.caption("Equivalency calculations based on EPA Greenhouse Gas Equivalencies Calculator · Grid intensity: India average (0.708 kg CO₂/kWh)")
