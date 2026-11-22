"""
Chapter 04 — Macro Scaling & Environmental Equivalencies

Extrapolating empirical per-query telemetry to commercial production scales
(1,000 to 10,000,000 daily queries) with EPA carbon equivalencies.
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
from utils.theme import EDITORIAL_CSS, chapter, divider, finding, formula_box

st.set_page_config(
    page_title="Macro Scaling & Equivalencies",
    layout="wide",
)

# ── Apply Editorial Theme ─────────────────────────────────────────────────────
st.markdown(EDITORIAL_CSS, unsafe_allow_html=True)


# ── Load Data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    return pd.read_csv(os.path.join(data_dir, "model_summary.csv"))


summary_df = load_data()


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    chapter(4, "MACRO SCALING & EQUIVALENCIES", "Annual carbon accounting and tangible EPA metrics"),
    unsafe_allow_html=True,
)
st.markdown("<h1>Macro Scaling and Environmental Equivalencies</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='hero-subtitle'>"
    "Translating micro-scale per-query watt-hours into annual macro-scale environmental impacts "
    "across commercial API request volumes using United States EPA carbon equivalency benchmarks."
    "</p>",
    unsafe_allow_html=True,
)


# ── Interactive Scaling Calculator ───────────────────────────────────────────
st.markdown(
    "<div class='chapter-marker'>DYNAMIC SCALE SIMULATOR</div>"
    "<h3>Configure Commercial Deployment Volume</h3>",
    unsafe_allow_html=True,
)

scale_col1, scale_col2 = st.columns([1, 2])

with scale_col1:
    daily_queries = st.select_slider(
        "Daily Query Volume (Requests / Day):",
        options=[
            1_000,
            5_000,
            10_000,
            50_000,
            100_000,
            500_000,
            1_000_000,
            5_000_000,
            10_000_000,
        ],
        value=1_000_000,
        format_func=lambda x: f"{x:,.0f} queries/day",
    )
    st.markdown(
        f"<p style='font-family: var(--mono); font-size: 11px; color: var(--graphite);'>"
        f"Annual cumulative volume: <strong>{daily_queries * 365:,.0f} queries/year</strong>."
        "</p>",
        unsafe_allow_html=True,
    )

with scale_col2:
    st.markdown(
        "<div class='compare-panel' style='padding: 1.2rem;'>"
        "<div class='compare-tag modern'>EPA ACCOUNTING STANDARDS</div>"
        "<p style='font-size: 0.88rem; line-height: 1.5; color: var(--graphite); margin: 0;'>"
        "Calculations utilize US Environmental Protection Agency (EPA) factors: "
        "Passenger Vehicle: 0.404 kg CO2/mile · Mature Tree Sequestration: 21.77 kg CO2/year · "
        "Smartphone Full Charge: 0.012 kWh · Residential Power: 30.0 kWh/day."
        "</p>"
        "</div>",
        unsafe_allow_html=True,
    )


# ── Annual Cumulative Calculations ───────────────────────────────────────────
st.markdown(divider(), unsafe_allow_html=True)

st.markdown(
    "<div class='chapter-marker'>ANNUAL METRICS</div>"
    f"<h3>Cumulative Impact at {daily_queries:,.0f} Daily Inquiries</h3>",
    unsafe_allow_html=True,
)

annual_queries = daily_queries * 365

annual_data = []
for _, row in summary_df.iterrows():
    ann_energy_kwh = row["avg_energy_kwh"] * annual_queries
    ann_co2_kg = (row["avg_co2_grams"] * annual_queries) / 1000.0
    miles = ann_co2_kg / CO2_KG_PER_CAR_MILE
    trees = ann_co2_kg / CO2_KG_PER_TREE_PER_YEAR
    charges = ann_energy_kwh / KWH_PER_SMARTPHONE_CHARGE
    home_days = ann_energy_kwh / KWH_PER_HOME_PER_DAY

    annual_data.append({
        "model": row["model"],
        "parameters_b": row["parameters_b"],
        "annual_energy_kwh": ann_energy_kwh,
        "annual_co2_kg": ann_co2_kg,
        "miles": miles,
        "trees": trees,
        "charges": charges,
        "home_days": home_days,
    })

ann_df = pd.DataFrame(annual_data)

# Contrast smallest vs largest
smallest_row = ann_df.loc[ann_df["parameters_b"].idxmin()]
largest_row = ann_df.loc[ann_df["parameters_b"].idxmax()]
avoidable_co2 = largest_row["annual_co2_kg"] - smallest_row["annual_co2_kg"]
avoidable_miles = largest_row["miles"] - smallest_row["miles"]
avoidable_trees = largest_row["trees"] - smallest_row["trees"]

eq_c1, eq_c2, eq_c3, eq_c4 = st.columns(4)

with eq_c1:
    st.markdown(
        f"""
        <div class="metric-editorial">
            <div class="metric-number red">{avoidable_co2/1000:.1f}</div>
            <div class="metric-unit">Metric Tons CO2e</div>
            <div class="metric-desc">Annual avoidable emissions between unoptimized 13B and right-sized 1.1B.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with eq_c2:
    st.markdown(
        f"""
        <div class="metric-editorial">
            <div class="metric-number">{avoidable_miles:,.0f}</div>
            <div class="metric-unit">Passenger Car Miles</div>
            <div class="metric-desc">Equivalent internal-combustion vehicle miles eliminated annually.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with eq_c3:
    st.markdown(
        f"""
        <div class="metric-editorial">
            <div class="metric-number green">{avoidable_trees:,.0f}</div>
            <div class="metric-unit">Mature Trees Required</div>
            <div class="metric-desc">Annual urban tree sequestration units needed to offset the 13B variance.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with eq_c4:
    st.markdown(
        f"""
        <div class="metric-editorial">
            <div class="metric-number">{largest_row['annual_energy_kwh']/1000:,.1f}</div>
            <div class="metric-unit">Megawatt-Hours (MWh)</div>
            <div class="metric-desc">Total annual electricity consumed by 13B serving this inquiry volume.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Annual Comparison Visualizations ──────────────────────────────────────────
st.markdown(divider(), unsafe_allow_html=True)

v1, v2 = st.columns(2)

with v1:
    fig_ann_co2 = go.Figure(go.Bar(
        x=[m.replace(" (Real)", "") for m in ann_df["model"]],
        y=ann_df["annual_co2_kg"] / 1000.0,
        marker_color=[MODEL_COLORS.get(m, "#111111") for m in ann_df["model"]],
        text=[f"{v:,.1f} t" for v in ann_df["annual_co2_kg"] / 1000.0],
        textposition="outside",
        textfont=dict(color="#000000", size=11, family="JetBrains Mono, monospace"),
    ))
    fig_ann_co2.update_layout(
        title=dict(text="Annual Carbon Emissions (Metric Tons CO2e)", font=dict(family="Fraunces, Georgia, serif", size=18, color="#000000")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(247,245,240,0.5)",
        font=dict(family="Inter, sans-serif", color="#000000", size=12),
        xaxis=dict(title=dict(text="Architecture", font=dict(color="#000000")), gridcolor="rgba(17,17,17,0.08)", tickfont=dict(color="#000000")),
        yaxis=dict(title=dict(text="Metric Tons CO2e", font=dict(color="#000000")), gridcolor="rgba(17,17,17,0.08)", tickfont=dict(color="#000000")),
        margin=dict(l=60, r=30, t=60, b=60),
    )
    st.plotly_chart(fig_ann_co2, use_container_width=True)

with v2:
    fig_trees = go.Figure(go.Bar(
        x=[m.replace(" (Real)", "") for m in ann_df["model"]],
        y=ann_df["trees"],
        marker_color=[MODEL_COLORS.get(m, "#111111") for m in ann_df["model"]],
        text=[f"{v:,.0f}" for v in ann_df["trees"]],
        textposition="outside",
        textfont=dict(color="#000000", size=11, family="JetBrains Mono, monospace"),
    ))
    fig_trees.update_layout(
        title=dict(text="Tree Offset Requirements (Trees Growing for 1 Year)", font=dict(family="Fraunces, Georgia, serif", size=18, color="#000000")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(247,245,240,0.5)",
        font=dict(family="Inter, sans-serif", color="#000000", size=12),
        xaxis=dict(title=dict(text="Architecture", font=dict(color="#000000")), gridcolor="rgba(17,17,17,0.08)", tickfont=dict(color="#000000")),
        yaxis=dict(title=dict(text="Tree Count", font=dict(color="#000000")), gridcolor="rgba(17,17,17,0.08)", tickfont=dict(color="#000000")),
        margin=dict(l=60, r=30, t=60, b=60),
    )
    st.plotly_chart(fig_trees, use_container_width=True)

# ── Macro Scaling Mathematical Specification ──────────────────────────────────
st.markdown(
    formula_box(
        "Annual Macro Extrapolation & EPA Equivalencies",
        r"\text{Annual Carbon (kg)} = \frac{\bar{C} \;(\text{g})}{1000} \times Q_{\text{daily}} \times 365, \quad \text{Vehicle Miles} = \frac{\text{CO}_{2}\text{e (kg)}}{0.404}, \quad \text{Tree-Years} = \frac{\text{CO}_{2}\text{e (kg)}}{21.77}",
        "Converts laboratory single-query telemetry into commercial annual footprints. "
        "EPA equivalency constants: 0.404 kg CO2e per passenger vehicle mile; 21.77 kg CO2 sequestered per mature urban tree per year.",
        tag="MACRO SCALING LAW",
    ),
    unsafe_allow_html=True,
)


# ── Summary Ledger ────────────────────────────────────────────────────────────
st.markdown(divider(), unsafe_allow_html=True)

st.markdown(
    "<div class='chapter-marker'>AUDIT SUMMARY TABLE</div>"
    "<h3>Annualized Footprint and Offsets Ledger</h3>",
    unsafe_allow_html=True,
)

tab_df = pd.DataFrame({
    "Architecture": [m.replace(" (Real)", "") for m in ann_df["model"]],
    "Parameters": [f"{p}B" for p in ann_df["parameters_b"]],
    "Annual Electricity": [f"{k:,.1f} kWh" for k in ann_df["annual_energy_kwh"]],
    "Annual Carbon": [f"{c/1000:,.2f} Metric Tons" for c in ann_df["annual_co2_kg"]],
    "Automobile Distance": [f"{m:,.0f} miles" for m in ann_df["miles"]],
    "Tree Sequestration": [f"{t:,.0f} trees" for t in ann_df["trees"]],
    "Household Power": [f"{h:,.0f} days" for h in ann_df["home_days"]],
})

st.dataframe(tab_df, use_container_width=True, hide_index=True)
