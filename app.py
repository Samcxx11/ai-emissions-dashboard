"""
AI Inference Emissions Dashboard — Main App (Overview Page)

Measures how much energy and CO₂ different AI language models use when
answering the same questions, to find out whether bigger, more powerful
AI models cost the planet more than smaller ones — and by how much.

Run:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Emissions Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Hero title */
    .hero-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #ecf0f1;
        margin-bottom: 0;
        line-height: 1.2;
    }
    .hero-subtitle {
        font-size: 1.15rem;
        color: #95a5a6;
        margin-top: 0.3rem;
        margin-bottom: 2rem;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1d23 0%, #2c3e50 100%);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid #34495e;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #3498db;
    }
    .metric-label {
        font-size: 0.95rem;
        color: #95a5a6;
        margin-top: 0.3rem;
    }
    .metric-value.green { color: #2ecc71; }
    .metric-value.orange { color: #f39c12; }
    .metric-value.red { color: #e74c3c; }

    /* Paper citation cards */
    .paper-card {
        background: #1a1d23;
        border: 1px solid #34495e;
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        border-left: 4px solid #3498db;
    }
    .paper-card h4 {
        color: #ecf0f1;
        margin: 0 0 0.3rem 0;
        font-size: 1rem;
    }
    .paper-card .authors {
        color: #95a5a6;
        font-size: 0.85rem;
        margin-bottom: 0.5rem;
    }
    .paper-card .relevance {
        color: #bdc3c7;
        font-size: 0.9rem;
    }

    /* Why-it-matters cards */
    .why-card {
        background: linear-gradient(135deg, #1a1d23, #1e272e);
        border-radius: 10px;
        padding: 1.3rem;
        border: 1px solid #34495e;
        height: 100%;
    }
    .why-card .icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    .why-card h4 {
        color: #ecf0f1;
        margin: 0 0 0.5rem 0;
    }
    .why-card p {
        color: #bdc3c7;
        font-size: 0.9rem;
        line-height: 1.5;
    }

    /* Section dividers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #ecf0f1;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #3498db;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: #0e1117;
    }
</style>
""", unsafe_allow_html=True)


# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    inference = pd.read_csv(os.path.join(data_dir, "inference_results.csv"))
    summary = pd.read_csv(os.path.join(data_dir, "model_summary.csv"))
    quant = pd.read_csv(os.path.join(data_dir, "quantization_comparison.csv"))
    return inference, summary, quant


inference_df, summary_df, quant_df = load_data()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/earth-planet.png", width=60)
    st.markdown("### 🌍 AI Emissions Dashboard")
    st.markdown("---")
    st.markdown(
        "Measuring the environmental cost of AI inference — "
        "one query at a time."
    )
    st.markdown("---")
    st.markdown("**Models tested:**")
    for _, row in summary_df.iterrows():
        st.markdown(f"- {row['model']} ({row['parameters_b']}B)")
    st.markdown("---")
    st.markdown(
        "**Methodology:**  \n"
        "Following [Strubell et al. (2019)](https://arxiv.org/abs/1906.02243) "
        "and [Henderson et al. (2020)](https://arxiv.org/abs/2002.05651)"
    )


# ── Hero Section ──────────────────────────────────────────────────────────────
st.markdown(
    '<p class="hero-title">🌍 AI Inference Emissions Dashboard</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="hero-subtitle">'
    "We're measuring how much energy and CO₂ different AI language models use "
    "when answering the same questions, to find out whether bigger, more "
    "powerful AI models cost the planet more than smaller ones — and by how much."
    "</p>",
    unsafe_allow_html=True,
)


# ── Key Metrics ───────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">📊 Experiment at a Glance</p>', unsafe_allow_html=True)

total_queries = len(inference_df)
total_energy_wh = inference_df["energy_kwh"].sum() * 1000
total_co2_g = inference_df["co2_grams"].sum()
models_tested = inference_df["model"].nunique()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""<div class="metric-card">
            <div class="metric-value green">{models_tested}</div>
            <div class="metric-label">Models Tested</div>
        </div>""",
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""<div class="metric-card">
            <div class="metric-value">{total_queries}</div>
            <div class="metric-label">Total Queries Run</div>
        </div>""",
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""<div class="metric-card">
            <div class="metric-value orange">{total_energy_wh:.2f} Wh</div>
            <div class="metric-label">Total Energy Consumed</div>
        </div>""",
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        f"""<div class="metric-card">
            <div class="metric-value red">{total_co2_g:.2f} g</div>
            <div class="metric-label">Total CO₂ Emitted</div>
        </div>""",
        unsafe_allow_html=True,
    )


# ── Why This Matters ──────────────────────────────────────────────────────────
st.markdown('<p class="section-header">❓ Why This Matters</p>', unsafe_allow_html=True)

why_col1, why_col2, why_col3 = st.columns(3)

with why_col1:
    st.markdown(
        """<div class="why-card">
            <div class="icon">📈</div>
            <h4>Scale</h4>
            <p>AI is now used billions of times a day — search, chatbots, coding
            assistants. Even a small per-query energy cost adds up massively at
            that scale.</p>
        </div>""",
        unsafe_allow_html=True,
    )

with why_col2:
    st.markdown(
        """<div class="why-card">
            <div class="icon">👻</div>
            <h4>Invisibility</h4>
            <p>Nobody sees the electricity bill behind an AI answer the way they
            see a car's exhaust. This project makes an invisible cost visible.</p>
        </div>""",
        unsafe_allow_html=True,
    )

with why_col3:
    st.markdown(
        """<div class="why-card">
            <div class="icon">🔬</div>
            <h4>Research Gap</h4>
            <p>Even researchers who study this (Strubell et al., Henderson et al.)
            note that most AI papers don't report energy costs at all — we're
            doing something the field itself struggles with.</p>
        </div>""",
        unsafe_allow_html=True,
    )


# ── Key Finding Preview ──────────────────────────────────────────────────────
st.markdown('<p class="section-header">🔑 Key Finding</p>', unsafe_allow_html=True)

# Calculate the comparison
smallest = summary_df.loc[summary_df["parameters_b"].idxmin()]
largest = summary_df.loc[summary_df["parameters_b"].idxmax()]
energy_ratio = largest["avg_energy_kwh"] / smallest["avg_energy_kwh"]
pct_more = (energy_ratio - 1) * 100

st.info(
    f"**{largest['model']}** ({largest['parameters_b']}B params) used roughly "
    f"**{pct_more:.0f}% more energy** per response than **{smallest['model']}** "
    f"({smallest['parameters_b']}B params), which suggests that choosing the "
    f"right-sized AI model for a task — not just the most powerful one — is a "
    f"real, measurable way to reduce carbon footprint."
)

st.markdown("")
st.markdown(
    "👈 **Use the sidebar pages** to explore detailed comparisons, "
    "quantization impact, per-query analysis, and real-world impact calculations."
)


# ── Methodology ───────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">🔬 Methodology</p>', unsafe_allow_html=True)

st.markdown("""
We followed this process to ensure a fair, reproducible comparison:

1. **Picked 4 real open-source AI models** of different sizes (1.1B → 13B parameters)
2. **Ran the same set of 20 questions** through each model, on the same hardware, so it's a fair comparison
3. **Used CodeCarbon** (used by real researchers, including the BLOOM carbon footprint study by Luccioni et al.) to measure energy per response
4. **Converted energy into CO₂** using India's electricity grid carbon intensity (0.708 kg CO₂/kWh)
5. **Compared results normalized per response**, so bigger models aren't unfairly penalized just for existing
6. **Tested quantization** (FP16 → INT8 → INT4) on Llama-2-7B (Real) to measure precision vs. efficiency trade-offs
""")

st.markdown(
    '> *"We followed the measurement methodology from Strubell et al. (ACL 2019) '
    "and the reporting standard from Henderson et al. (JMLR 2020), and validated "
    "our tool choice against how Luccioni et al. measured BLOOM's carbon "
    'footprint."*'
)


# ── Research Papers ───────────────────────────────────────────────────────────
st.markdown('<p class="section-header">📚 Research Foundation</p>', unsafe_allow_html=True)

import sys
sys.path.insert(0, os.path.dirname(__file__))
from utils.constants import PAPERS

for paper in PAPERS:
    st.markdown(
        f"""<div class="paper-card">
            <h4><a href="{paper['url']}" target="_blank" style="color: #3498db; text-decoration: none;">
                {paper['title']}
            </a></h4>
            <div class="authors">{paper['authors']}</div>
            <div class="relevance">📌 {paper['relevance']}</div>
        </div>""",
        unsafe_allow_html=True,
    )


# ── Limitations ───────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">⚠️ Acknowledged Limitations</p>', unsafe_allow_html=True)

st.markdown("""
- **Inference only**: We measured only inference (answering questions) emissions — not training emissions, which require resources beyond student access.
- **Single hardware setup**: All tests ran on the same machine; results may differ on other hardware configurations.
- **Sample data note**: This dashboard currently uses realistic simulated data. Replace the CSVs in `data/` with real CodeCarbon measurements for production use.
- **Grid intensity**: We used India's average grid carbon intensity (0.708 kg CO₂/kWh). Actual emissions vary by region, time of day, and energy mix.
- **Quantization on CPU**: Quantized models (INT4/INT8) work best on NVIDIA GPUs — CPU-only quantization is possible but slower and less standard.
""")

st.markdown("---")
st.caption(
    "Built as a research project on AI environmental impact · "
    "Data methodology follows Strubell et al. (2019), Henderson et al. (2020), "
    "and Luccioni et al. (2022)"
)
