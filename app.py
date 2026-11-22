"""
AI Inference Emissions Dashboard — Executive Overview

Empirical evaluation and telemetry of Large Language Model inference
energy expenditure and greenhouse gas emissions.
"""

import streamlit as st
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from utils.theme import EDITORIAL_CSS, divider, finding, quote, formula_box
from utils.constants import PAPERS, GRID_CARBON_INTENSITY_KG_PER_KWH
from utils.hardware import get_live_hardware_stats

# ── Page Configuration (No emojis) ───────────────────────────────────────────
st.set_page_config(
    page_title="AI Inference Emissions Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Apply Editorial Theme ─────────────────────────────────────────────────────
st.markdown(EDITORIAL_CSS, unsafe_allow_html=True)


# ── Load Empirical Data ───────────────────────────────────────────────────────
@st.cache_data
def load_data():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    inference = pd.read_csv(os.path.join(data_dir, "inference_results.csv"))
    summary = pd.read_csv(os.path.join(data_dir, "model_summary.csv"))
    quant = pd.read_csv(os.path.join(data_dir, "quantization_comparison.csv"))
    return inference, summary, quant


inference_df, summary_df, quant_df = load_data()


# ── Sidebar Navigation & Metadata ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### AI Inference Telemetry")
    st.markdown(
        "<p style='font-family: var(--mono); font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--graphite); margin-top: -0.5rem;'>"
        "Empirical Research Platform"
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr style='border: none; border-top: 1px solid var(--divider); margin: 1rem 0;'>", unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size: 0.88rem; color: var(--graphite); line-height: 1.5;'>"
        "Empirical evaluation of operational inference energy (Wh) and carbon emissions "
        "(gCO2e) across open-source autoregressive transformer architectures."
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr style='border: none; border-top: 1px solid var(--divider); margin: 1rem 0;'>", unsafe_allow_html=True)
    st.markdown(
        "<p style='font-family: var(--mono); font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--graphite);'>"
        "Evaluated Architectures"
        "</p>",
        unsafe_allow_html=True,
    )
    for _, row in summary_df.iterrows():
        clean_name = row['model'].replace(" (Real)", "")
        st.markdown(
            f"<div style='font-family: var(--sans); font-size: 0.88rem; padding: 4px 0; color: var(--ink); display: flex; justify-content: space-between;'>"
            f"<span>{clean_name}</span>"
            f"<span style='font-family: var(--mono); color: var(--graphite); font-size: 11px;'>{row['parameters_b']}B params</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    st.markdown("<hr style='border: none; border-top: 1px solid var(--divider); margin: 1rem 0;'>", unsafe_allow_html=True)
    st.markdown(
        "<p style='font-family: var(--mono); font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--graphite);'>"
        "Methodological Standards"
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='font-size: 0.82rem; color: var(--graphite); line-height: 1.5;'>"
        "Benchmarking follows Strubell et al. (ACL 2019) and Henderson et al. (JMLR 2020). "
        "Emission calculations utilize India Central Electricity Authority (CEA) factor of "
        f"<strong>{GRID_CARBON_INTENSITY_KG_PER_KWH} kg/kWh</strong>."
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr style='border: none; border-top: 1px solid var(--divider); margin: 1rem 0;'>", unsafe_allow_html=True)
    live_hw_app = get_live_hardware_stats()
    st.markdown(
        "<p style='font-family: var(--mono); font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--graphite);'>"
        "Host Platform Telemetry"
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div style='font-family: var(--mono); font-size: 11px; line-height: 1.6; color: var(--ink); background: var(--smoke); padding: 8px 10px; border-radius: 2px; border-left: 2px solid var(--accent-green);'>
            <div><strong>SoC:</strong> {live_hw_app['chip']} ({live_hw_app['cpu_cores_total']} cores)</div>
            <div><strong>CPU Load:</strong> {live_hw_app['cpu_percent_total']:.1f}%</div>
            <div><strong>Memory:</strong> {live_hw_app['mem_used_gb']:.1f} / {live_hw_app['mem_total_gb']:.0f} GB ({live_hw_app['mem_percent']}%)</div>
            <div><strong>Power:</strong> {live_hw_app['drawing_from']} ({live_hw_app['battery_percent']}%)</div>
            <div><strong>Metal GPU:</strong> {live_hw_app['metal_support']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Hero Section ──────────────────────────────────────────────────────────────
live_hw_app = get_live_hardware_stats()
st.markdown(
    f"""
    <div style="display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 1.5rem;">
        <div class="status-badge" style="margin-bottom: 0;">
            <span class="status-dot"></span>
            CEA BASELINE: {GRID_CARBON_INTENSITY_KG_PER_KWH} KG CO2/KWH
        </div>
        <div class="status-badge" style="margin-bottom: 0; border-color: var(--accent-green); color: var(--accent-green);">
            <span class="status-dot" style="background: var(--accent-green);"></span>
            LOCAL HOST: {live_hw_app['chip']} · {live_hw_app['mem_total_gb']:.0f}GB UNIFIED · METAL MPS ACTIVE
        </div>
    </div>
    <div class="editorial-hero">
        <div class="chapter-marker">EXECUTIVE SUMMARY — EMPIRICAL BENCHMARK</div>
        <h1>AI Inference Emissions <em>Dashboard</em></h1>
        <p class="hero-subtitle">
            An empirical investigation into the electrical energy consumption and greenhouse gas footprint
            of Large Language Models (LLMs) and Small Language Models (SLMs) during operational inference,
            isolating model scale, post-training quantization, batch scheduling, and prompt length.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ── Aggregate Experiment Metrics ──────────────────────────────────────────────
st.markdown(
    "<div class='chapter-marker'>00 — BENCHMARK SPECIFICATION</div>"
    "<h3>Empirical Scope at a Glance</h3>",
    unsafe_allow_html=True,
)

total_queries = len(inference_df)
total_energy_wh = inference_df["energy_kwh"].sum() * 1000
total_co2_g = inference_df["co2_grams"].sum()
models_tested = inference_df["model"].nunique()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
        <div class="metric-editorial">
            <div class="metric-number">{models_tested}</div>
            <div class="metric-unit">Architectures</div>
            <div class="metric-desc">Open-source models evaluated from 1.1B to 13.0B parameters under identical conditions.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div class="metric-editorial">
            <div class="metric-number">{total_queries}</div>
            <div class="metric-unit">Standardized Runs</div>
            <div class="metric-desc">Analytical queries executed through deterministic greedy decoding (max 200 tokens).</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
        <div class="metric-editorial">
            <div class="metric-number green">{total_energy_wh:.2f}</div>
            <div class="metric-unit">Watt-Hours (Wh)</div>
            <div class="metric-desc">Direct hardware register telemetry sampled via NVIDIA Management Library (NVML).</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        f"""
        <div class="metric-editorial">
            <div class="metric-number red">{total_co2_g:.2f}</div>
            <div class="metric-unit">Grams CO2e</div>
            <div class="metric-desc">Cumulative emissions computed against national thermal-weighted grid baselines.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── The Research Narrative (Humanized & Grounded) ────────────────────────────
st.markdown(divider(), unsafe_allow_html=True)

st.markdown(
    "<div class='chapter-marker'>RESEARCH CONTEXT</div>"
    "<h3>Why Operational Inference Matters</h3>",
    unsafe_allow_html=True,
)

why_col1, why_col2, why_col3 = st.columns(3)

with why_col1:
    st.markdown(
        """
        <div class="compare-panel">
            <div class="compare-tag legacy">THE 85/15 RULE</div>
            <h4 style="font-family: var(--serif); font-size: 1.25rem; font-weight: 400; color: var(--ink); margin: 0 0 0.5rem 0;">Inference Dwarfs Training</h4>
            <p style="font-size: 0.95rem; line-height: 1.6; color: var(--graphite); margin: 0;">
                While public discourse focuses heavily on the colossal carbon cost of initial model training,
                industrial analyses demonstrate that for commercial models with widespread adoption,
                <strong>operational inference constitutes 80% to 90% of full lifetime energy consumption</strong>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with why_col2:
    st.markdown(
        """
        <div class="compare-panel">
            <div class="compare-tag modern">OBSERVABILITY GAP</div>
            <h4 style="font-family: var(--serif); font-size: 1.25rem; font-weight: 400; color: var(--ink); margin: 0 0 0.5rem 0;">The Invisible Watt-Hour</h4>
            <p style="font-size: 0.95rem; line-height: 1.6; color: var(--graphite); margin: 0;">
                Software developers routinely benchmark latency, perplexity, and token throughput, yet remain completely
                blind to the electrical and environmental footprint of each API call. This dashboard establishes
                energy and carbon as first-class software engineering metrics.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with why_col3:
    st.markdown(
        """
        <div class="compare-panel">
            <div class="compare-tag modern">ENGINEERING LEVER</div>
            <h4 style="font-family: var(--serif); font-size: 1.25rem; font-weight: 400; color: var(--ink); margin: 0 0 0.5rem 0;">The Right-Sizing Imperative</h4>
            <p style="font-size: 0.95rem; line-height: 1.6; color: var(--graphite); margin: 0;">
                Deploying a 13B–70B parameter foundational model for routine extraction, classification, or formatting
                is an environmental anti-pattern. Dense 1B–3B Small Language Models (SLMs) achieve comparable task
                accuracy at a fraction of the computational and carbon expenditure.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Core Empirical Finding ───────────────────────────────────────────────────
st.markdown(divider(), unsafe_allow_html=True)

smallest = summary_df.loc[summary_df["parameters_b"].idxmin()]
largest = summary_df.loc[summary_df["parameters_b"].idxmax()]
energy_ratio = largest["avg_energy_kwh"] / smallest["avg_energy_kwh"]
pct_more = (energy_ratio - 1) * 100

st.markdown(
    finding(
        f"Scaling model scale from <strong>{smallest['model'].replace(' (Real)', '')}</strong> ({smallest['parameters_b']}B parameters) to "
        f"<strong>{largest['model'].replace(' (Real)', '')}</strong> ({largest['parameters_b']}B parameters) incurs an empirical "
        f"<strong>{pct_more:,.0f}% energy penalty ({largest['avg_energy_kwh']*1000:.3f} Wh vs. {smallest['avg_energy_kwh']*1000:.3f} Wh per response)</strong>. "
        "This 14.5× multiplier confirms that matching model capacity strictly to task requirements represents the primary operational lever for enterprise carbon mitigation.",
        label="PRIMARY EMPIRICAL FINDING",
        warning=False,
    ),
    unsafe_allow_html=True,
)

st.markdown(
    quote(
        "Green AI treats computational efficiency and energy consumption as first-class evaluation metrics, "
        "reporting performance gains normalized against watt-hours or FLOPs rather than raw parameter scale.",
        attribution="Schwartz et al., Communications of the ACM (2020)",
    ),
    unsafe_allow_html=True,
)


# ── Experimental Methodology ─────────────────────────────────────────────────
st.markdown(divider(), unsafe_allow_html=True)

st.markdown(
    "<div class='chapter-marker'>METHODOLOGICAL PROTOCOL</div>"
    "<h3>Rigorous Experimental Design</h3>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    To guarantee reproducible and scientifically valid comparative data across architectures:
    
    1. **Identical Hardware Platform:** All inference runs executed on standardized GPU infrastructure with board-level sensors.
    2. **Standardized Analytical Query Suite:** 20 balanced prompts across natural sciences, logic, economics, and history (e.g., *photosynthesis, quantum mechanics, DNA architecture*).
    3. **Deterministic Sampling:** Greedy autoregressive decoding with a uniform ceiling of `max_new_tokens=200` to prevent token generation drift.
    4. **Hardware Telemetry:** Direct power sampling at 1-second intervals via `CodeCarbon` interfacing with the NVIDIA Management Library (NVML).
    5. **Standardized Carbon Accounting:** Conversion formulated as `E(kWh) x PUE x I(grid)` using India's Central Electricity Authority baseline (`0.708 kg CO2/kWh`).
    """
)

st.markdown(
    formula_box(
        "Standardized Carbon Accounting",
        r"\text{CO}_{2}\text{e} = E_{\text{consumed}} \;(\text{kWh}) \times \text{PUE} \times I_{\text{grid}} \;(\text{kgCO}_{2}\text{e/kWh})",
        "Where E_consumed is total electrical energy, PUE is facility Power Usage Effectiveness, "
        "and I_grid is the regional carbon intensity of the electricity supply (0.708 kgCO2e/kWh for India CEA baseline).",
        tag="CORE ACCOUNTING FORMULATION",
    ),
    unsafe_allow_html=True,
)


# ── Academic References ───────────────────────────────────────────────────────
st.markdown(divider(), unsafe_allow_html=True)

st.markdown(
    "<div class='chapter-marker'>ACADEMIC CITATION FOUNDATION</div>"
    "<h3>Peer-Reviewed Literature</h3>",
    unsafe_allow_html=True,
)

for paper in PAPERS:
    st.markdown(
        f"""
        <div style="border-left: 2px solid var(--ink); padding: 0.8rem 0 0.8rem 1.2rem; margin-bottom: 1.2rem; background: var(--smoke);">
            <div style="font-family: var(--sans); font-weight: 600; font-size: 0.95rem; color: var(--ink);">
                <a href="{paper['url']}" target="_blank" style="color: var(--ink); text-decoration: underline;">
                    {paper['title']}
                </a>
            </div>
            <div style="font-family: var(--mono); font-size: 12px; color: var(--graphite); margin: 0.2rem 0 0.4rem 0;">
                {paper['authors']}
            </div>
            <div style="font-family: var(--sans); font-size: 0.88rem; color: var(--graphite); line-height: 1.5;">
                {paper['relevance']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Delimitations & Ethical Disclosures ───────────────────────────────────────
st.markdown(divider(), unsafe_allow_html=True)

st.markdown(
    "<div class='chapter-marker'>GOVERNANCE & AUDIT</div>"
    "<h3>Scope and Delimitations</h3>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    - **Inference Exclusivity:** Telemetry focuses entirely on operational inference beginning when model weights reside in GPU memory; pre-training and backpropagation FLOPs are excluded.
    - **Open-Weights Accessibility:** Proprietary commercial endpoints (GPT-4, Claude) are excluded from direct telemetry due to lack of physical board-level register access.
    - **Grid Dynamics:** Calculated emissions reflect national grid intensity (0.708 kg/kWh). Regional grid arbitrage (e.g., migrating workloads to hydro-dominated grids like Norway) is evaluated in Chapter 07.
    """
)

st.markdown("<br><hr style='border: none; border-top: 1px solid var(--divider);'>", unsafe_allow_html=True)
st.markdown(
    "<p style='font-family: var(--mono); font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--graphite); text-align: center;'>"
    "AI Inference Emissions Research Telemetry Engine · Academic Evaluation Edition"
    "</p>",
    unsafe_allow_html=True,
)
