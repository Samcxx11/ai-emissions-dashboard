import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.theme import EDITORIAL_CSS, apply_editorial_layout, chapter, divider, finding, quote, metric_card, big_number

st.set_page_config(page_title="Real-World Impact", page_icon="🌍", layout="wide")
st.markdown(EDITORIAL_CSS, unsafe_allow_html=True)

from utils.constants import (
    MODEL_COLORS,
    MODEL_ORDER,
    CO2_KG_PER_CAR_MILE,
    KWH_PER_SMARTPHONE_CHARGE,
    CO2_KG_PER_TREE_PER_YEAR,
    KWH_PER_HOME_PER_DAY,
    GRID_CARBON_INTENSITY_KG_PER_KWH,
)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    return pd.read_csv(os.path.join(data_dir, "inference_results.csv"))

df = load_data()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    chapter(1, "Real-World Impact", "What do our measurements mean at real-world scale? This page translates per-query numbers into tangible, understandable impacts."),
    unsafe_allow_html=True
)
st.markdown(divider(), unsafe_allow_html=True)

# ── Causal chain ──────────────────────────────────────────────────────────────
st.markdown(
    chapter(2, "How This Project Leads to Reduction", ""),
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(big_number(1) + "<br><strong>Measure</strong><br>emissions", unsafe_allow_html=True)
with c2:
    st.markdown(big_number(2) + "<br><strong>Reveal</strong><br>wasteful choices", unsafe_allow_html=True)
with c3:
    st.markdown(big_number(3) + "<br><strong>Recommend</strong><br>better choices", unsafe_allow_html=True)
with c4:
    st.markdown(big_number(4) + "<br><strong>Reduce</strong><br>emissions", unsafe_allow_html=True)

st.write("")
st.markdown(
    finding("We built the first two links. The last two are the 'so what' — the actionable insights we hand to the reader.", "CAUSAL CHAIN"),
    unsafe_allow_html=True
)
st.markdown(divider(), unsafe_allow_html=True)

# ── Scaling Calculator ────────────────────────────────────────────────────────
st.markdown(
    chapter(3, "Scaling Calculator", "See what happens when your per-query measurements scale to real-world usage. Adjust the slider to see the annual impact."),
    unsafe_allow_html=True,
)

col_select, col_slider = st.columns([1, 2])

with col_select:
    selected_model = st.selectbox(
        "Choose a model:",
        options=MODEL_ORDER,
        index=2,  # default to Llama-2-7B (Real)
    )

with col_slider:
    queries_per_day = st.slider(
        "Queries per day:",
        min_value=1_000,
        max_value=10_000_000,
        value=100_000,
        step=1_000,
        format="%d",
    )

# Calculate impact
model_data = df[df["model"] == selected_model]
avg_energy_kwh = model_data["energy_kwh"].mean()
avg_co2_g = model_data["co2_grams"].mean()

daily_energy_kwh = avg_energy_kwh * queries_per_day
annual_energy_kwh = daily_energy_kwh * 365
annual_co2_kg = (avg_co2_g * queries_per_day * 365) / 1000

# Equivalencies
car_miles = annual_co2_kg / CO2_KG_PER_CAR_MILE
smartphone_charges = annual_energy_kwh / KWH_PER_SMARTPHONE_CHARGE
trees_needed = annual_co2_kg / CO2_KG_PER_TREE_PER_YEAR
home_days = annual_energy_kwh / KWH_PER_HOME_PER_DAY

st.markdown(f"**{selected_model} @ {queries_per_day:,} queries/day**")

met1, met2, met3 = st.columns(3)
with met1:
    st.markdown(metric_card(f"{annual_energy_kwh:,.1f}", "kWh", "Annual Energy", "red"), unsafe_allow_html=True)
with met2:
    st.markdown(metric_card(f"{annual_co2_kg:,.1f}", "kg", "Annual CO₂", "red"), unsafe_allow_html=True)
with met3:
    st.markdown(metric_card(f"{daily_energy_kwh:,.2f}", "kWh", "Daily Energy", ""), unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    chapter(4, "What Does That Look Like?", "Translating energy and CO₂ into everyday equivalents."),
    unsafe_allow_html=True,
)

eq1, eq2, eq3, eq4 = st.columns(4)

with eq1:
    st.markdown(metric_card(f"{car_miles:,.0f}", "miles", "Driven by car", "red"), unsafe_allow_html=True)
with eq2:
    st.markdown(metric_card(f"{smartphone_charges:,.0f}", "charges", "Smartphone charges", "red"), unsafe_allow_html=True)
with eq3:
    st.markdown(metric_card(f"{trees_needed:,.1f}", "trees", "Needed to offset (1 yr)", "green"), unsafe_allow_html=True)
with eq4:
    st.markdown(metric_card(f"{home_days:,.1f}", "days", "Powering a US home", "red"), unsafe_allow_html=True)

st.markdown(divider(), unsafe_allow_html=True)

# ── Cross-model scaling comparison ───────────────────────────────────────────
st.markdown(
    chapter(5, "Annual Impact: All Models Compared", f"What if each model served {queries_per_day:,} queries/day for a year?"),
    unsafe_allow_html=True,
)

comparison_rows = []
for model in MODEL_ORDER:
    m_data = df[df["model"] == model]
    m_energy = m_data["energy_kwh"].mean() * queries_per_day * 365
    m_co2 = (m_data["co2_grams"].mean() * queries_per_day * 365) / 1000
    comparison_rows.append({
        "Model": model,
        "Annual Energy (kWh)": m_energy,
        "Annual CO₂ (kg)": m_co2,
    })

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
    title=f"Annual CO₂ Emissions @ {queries_per_day:,} queries/day",
    xaxis=dict(title="Model"),
    yaxis=dict(title="CO₂ (kg/year)"),
    margin=dict(t=60, b=60),
    showlegend=False,
)
apply_editorial_layout(fig)
st.plotly_chart(fig, use_container_width=True)

# Show savings if switching
smallest_co2 = comp_df["Annual CO₂ (kg)"].min()
largest_co2 = comp_df["Annual CO₂ (kg)"].max()
savings_kg = largest_co2 - smallest_co2

st.markdown(
    finding(
        f"Switching from the largest to the smallest model would save {savings_kg:,.0f} kg CO₂/year at this query volume — "
        f"equivalent to {savings_kg / CO2_KG_PER_CAR_MILE:,.0f} fewer car miles driven.",
        "POTENTIAL SAVINGS"
    ),
    unsafe_allow_html=True
)
st.markdown(divider(), unsafe_allow_html=True)

# ── Recommendations ───────────────────────────────────────────────────────────
st.markdown(
    chapter(6, "Actionable Recommendations", "Practical steps to minimize your AI footprint."),
    unsafe_allow_html=True,
)

recommendations = [
    {
        "icon": "📏",
        "title": "1. Right-Size Your Model",
        "text": (
            "If a 1B model performs almost as well as a 7B model for simple tasks "
            "(like basic Q&A), use the smaller one. This is the single biggest lever — "
            "smaller model = less energy every single query, forever."
        ),
    },
    {
        "icon": "🎯",
        "title": "2. Avoid Overkill",
        "text": (
            "Many people use large general-purpose models for tasks a small "
            "specialized model could handle. Our comparison gives evidence for "
            "matching model size to task difficulty."
        ),
    },
    {
        "icon": "🔧",
        "title": "3. Quantize When Possible",
        "text": (
            "INT8 quantization can reduce energy by ~35% with minimal quality loss. "
            "INT4 saves ~58% but check quality for your specific use case."
        ),
    },
    {
        "icon": "💾",
        "title": "4. Cache Repeated Queries",
        "text": (
            "If the same question gets asked often (FAQs, customer support), "
            "cache the answer instead of re-running the model every time. "
            "Zero energy for a cached response."
        ),
    },
    {
        "icon": "📦",
        "title": "5. Batch Requests",
        "text": (
            "Running multiple queries together is often more energy-efficient "
            "per-query than one at a time, due to GPU utilization patterns."
        ),
    },
    {
        "icon": "⏰",
        "title": "6. Time-of-Use Awareness",
        "text": (
            "Running heavy AI workloads when the electricity grid has more "
            "renewable energy available reduces the carbon intensity per kWh. "
            "The same computation can have different CO₂ costs at different times."
        ),
    },
]

for rec in recommendations:
    st.markdown(
        finding(rec['text'], f"{rec['icon']} {rec['title']}"),
        unsafe_allow_html=True,
    )

st.markdown(divider(), unsafe_allow_html=True)

# ── Bottom line ───────────────────────────────────────────────────────────────
st.markdown(
    chapter(7, "The Bottom Line", ""),
    unsafe_allow_html=True,
)

st.markdown(
    quote("People and companies use AI without knowing its environmental cost, so they can't make informed choices. You can't reduce what you can't measure.", "The Core Problem"),
    unsafe_allow_html=True
)

st.markdown(
    quote("Developers should default to smaller models when the task doesn't need a large one — that's not a compromise, it's an optimization.", "The Solution"),
    unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)
st.caption(
    "Equivalency calculations based on EPA Greenhouse Gas Equivalencies Calculator · "
    "Grid intensity: India average (0.708 kg CO₂/kWh)"
)
