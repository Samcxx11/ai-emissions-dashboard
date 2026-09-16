"""
Page 4 — Real-World Impact

Scaling calculator, real-world equivalencies, and actionable
recommendations for reducing AI inference emissions.
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
    GRID_CARBON_INTENSITY_KG_PER_KWH,
)

st.set_page_config(page_title="Real-World Impact", page_icon="🌍", layout="wide")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .section-header {
        font-size: 1.5rem; font-weight: 600; color: #ecf0f1;
        margin: 2rem 0 1rem 0; padding-bottom: 0.5rem;
        border-bottom: 2px solid #3498db;
    }
    .impact-card {
        background: linear-gradient(135deg, #1a1d23 0%, #2c3e50 100%);
        border-radius: 12px; padding: 1.5rem; text-align: center;
        border: 1px solid #34495e;
    }
    .impact-card .icon { font-size: 2.5rem; margin-bottom: 0.5rem; }
    .impact-card .value { font-size: 1.8rem; font-weight: 700; color: #e74c3c; }
    .impact-card .label { font-size: 0.9rem; color: #95a5a6; margin-top: 0.3rem; }
    .rec-card {
        background: #1a1d23; border-radius: 10px; padding: 1.3rem;
        border: 1px solid #34495e; border-left: 4px solid #2ecc71;
        margin-bottom: 1rem;
    }
    .rec-card h4 { color: #ecf0f1; margin: 0 0 0.5rem 0; }
    .rec-card p { color: #bdc3c7; margin: 0; line-height: 1.5; font-size: 0.95rem; }
    .chain-step {
        background: #1a1d23; border-radius: 10px; padding: 1rem 1.5rem;
        border: 1px solid #34495e; text-align: center;
    }
    .chain-step .step-icon { font-size: 2rem; }
    .chain-step .step-text { color: #bdc3c7; font-size: 0.9rem; margin-top: 0.3rem; }
    .chain-arrow {
        display: flex; align-items: center; justify-content: center;
        font-size: 2rem; color: #3498db;
    }
</style>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    return pd.read_csv(os.path.join(data_dir, "inference_results.csv"))

df = load_data()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🌍 Real-World Impact")
st.markdown(
    "What do our measurements mean at real-world scale? "
    "This page translates per-query numbers into tangible, understandable impacts."
)

# ── Causal chain ──────────────────────────────────────────────────────────────
st.markdown(
    '<p class="section-header">🔗 How This Project Leads to Reduction</p>',
    unsafe_allow_html=True,
)

c1, a1, c2, a2, c3, a3, c4 = st.columns([2, 1, 2, 1, 2, 1, 2])

with c1:
    st.markdown(
        """<div class="chain-step">
            <div class="step-icon">📏</div>
            <div class="step-text"><strong>Measure</strong><br>emissions</div>
        </div>""",
        unsafe_allow_html=True,
    )
with a1:
    st.markdown('<div class="chain-arrow">→</div>', unsafe_allow_html=True)
with c2:
    st.markdown(
        """<div class="chain-step">
            <div class="step-icon">👁️</div>
            <div class="step-text"><strong>Reveal</strong><br>wasteful choices</div>
        </div>""",
        unsafe_allow_html=True,
    )
with a2:
    st.markdown('<div class="chain-arrow">→</div>', unsafe_allow_html=True)
with c3:
    st.markdown(
        """<div class="chain-step">
            <div class="step-icon">💡</div>
            <div class="step-text"><strong>Recommend</strong><br>better choices</div>
        </div>""",
        unsafe_allow_html=True,
    )
with a3:
    st.markdown('<div class="chain-arrow">→</div>', unsafe_allow_html=True)
with c4:
    st.markdown(
        """<div class="chain-step">
            <div class="step-icon">🌱</div>
            <div class="step-text"><strong>Reduce</strong><br>emissions</div>
        </div>""",
        unsafe_allow_html=True,
    )

st.markdown("")
st.info(
    "**We built the first two links.** The last two are the 'so what' — "
    "the actionable insights we hand to the reader."
)

# ── Scaling Calculator ────────────────────────────────────────────────────────
st.markdown(
    '<p class="section-header">🧮 Scaling Calculator</p>',
    unsafe_allow_html=True,
)

st.markdown(
    "See what happens when your per-query measurements scale to real-world usage. "
    "Adjust the slider to see the annual impact."
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

# Display metrics
st.markdown(f"### {selected_model} @ {queries_per_day:,} queries/day")

met1, met2, met3 = st.columns(3)
met1.metric("Annual Energy", f"{annual_energy_kwh:,.1f} kWh")
met2.metric("Annual CO₂", f"{annual_co2_kg:,.1f} kg")
met3.metric("Daily Energy", f"{daily_energy_kwh:,.2f} kWh")

# Equivalency cards
st.markdown(
    '<p class="section-header">🔄 What Does That Look Like?</p>',
    unsafe_allow_html=True,
)

eq1, eq2, eq3, eq4 = st.columns(4)

with eq1:
    st.markdown(
        f"""<div class="impact-card">
            <div class="icon">🚗</div>
            <div class="value">{car_miles:,.0f}</div>
            <div class="label">miles driven by car</div>
        </div>""",
        unsafe_allow_html=True,
    )
with eq2:
    st.markdown(
        f"""<div class="impact-card">
            <div class="icon">📱</div>
            <div class="value">{smartphone_charges:,.0f}</div>
            <div class="label">smartphone charges</div>
        </div>""",
        unsafe_allow_html=True,
    )
with eq3:
    st.markdown(
        f"""<div class="impact-card">
            <div class="icon">🌳</div>
            <div class="value">{trees_needed:,.1f}</div>
            <div class="label">trees needed to offset (1 yr)</div>
        </div>""",
        unsafe_allow_html=True,
    )
with eq4:
    st.markdown(
        f"""<div class="impact-card">
            <div class="icon">🏠</div>
            <div class="value">{home_days:,.1f}</div>
            <div class="label">days powering a US home</div>
        </div>""",
        unsafe_allow_html=True,
    )

# ── Cross-model scaling comparison ───────────────────────────────────────────
st.markdown(
    '<p class="section-header">📊 Annual Impact: All Models Compared</p>',
    unsafe_allow_html=True,
)

st.markdown(
    f"What if each model served **{queries_per_day:,} queries/day** for a year?"
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
    paper_bgcolor="#0e1117",
    plot_bgcolor="#1a1d23",
    font=dict(color="#bdc3c7"),
    xaxis=dict(title="Model", gridcolor="#2c3e50"),
    yaxis=dict(title="CO₂ (kg/year)", gridcolor="#2c3e50"),
    margin=dict(t=60, b=60),
    showlegend=False,
)
st.plotly_chart(fig, use_container_width=True)

# Show savings if switching
smallest_co2 = comp_df["Annual CO₂ (kg)"].min()
largest_co2 = comp_df["Annual CO₂ (kg)"].max()
savings_kg = largest_co2 - smallest_co2

st.success(
    f"**Switching from the largest to the smallest model would save "
    f"{savings_kg:,.0f} kg CO₂/year** at this query volume — "
    f"equivalent to {savings_kg / CO2_KG_PER_CAR_MILE:,.0f} fewer car miles driven."
)

# ── Recommendations ───────────────────────────────────────────────────────────
st.markdown(
    '<p class="section-header">💡 Actionable Recommendations</p>',
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
        f"""<div class="rec-card">
            <h4>{rec['icon']} {rec['title']}</h4>
            <p>{rec['text']}</p>
        </div>""",
        unsafe_allow_html=True,
    )

# ── Bottom line ───────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "### 🎯 The Bottom Line"
)
st.markdown(
    "**The real-world problem is:** People and companies use AI without knowing "
    "its environmental cost, so they can't make informed choices. "
    "You can't reduce what you can't measure."
)
st.markdown(
    "**Developers should default to smaller models when the task doesn't need "
    "a large one** — that's not a compromise, it's an optimization."
)

st.markdown("---")
st.caption(
    "Equivalency calculations based on EPA Greenhouse Gas Equivalencies Calculator · "
    "Grid intensity: India average (0.708 kg CO₂/kWh)"
)
