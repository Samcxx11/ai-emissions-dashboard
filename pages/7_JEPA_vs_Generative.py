"""
Page 7 — JEPA vs Generative AI: The Future of Energy-Efficient AI

Compares energy consumption of Joint Embedding Predictive Architecture (JEPA)
models against traditional Generative models for vision tasks.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd

from utils.charts import bar_jepa_energy_comparison, line_jepa_scaling, bar_jepa_co2_comparison
from utils.constants import ARCHITECTURE_COLORS, JEPA_MODEL_COLORS

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="JEPA vs Generative", page_icon="🧠", layout="wide")

st.markdown(
    """
    <style>
    .section-header { font-size:1.3rem; font-weight:700; margin-top:1.5rem; }
    .insight-box {
        background: linear-gradient(135deg, rgba(46,204,113,0.15), rgba(52,152,219,0.10));
        border-left: 4px solid #2ecc71;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 1rem 0;
    }
    .metric-card {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .metric-value { font-size: 2rem; font-weight: 800; }
    .metric-label { font-size: 0.85rem; opacity: 0.7; }
    .green { color: #2ecc71; }
    .red { color: #e74c3c; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Load data ─────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

@st.cache_data
def load_comparison():
    return pd.read_csv(os.path.join(DATA_DIR, "jepa_comparison.csv"))

@st.cache_data
def load_scaling():
    return pd.read_csv(os.path.join(DATA_DIR, "jepa_scaling.csv"))

df_comp = load_comparison()
df_scale = load_scaling()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🧠 JEPA vs Generative AI")
st.markdown(
    "**Joint Embedding Predictive Architecture (JEPA)** is a new approach by "
    "Meta AI's Yann LeCun that predicts *abstract concepts* instead of every "
    "pixel — resulting in dramatically lower energy consumption."
)

# ── What is JEPA? (Expander) ─────────────────────────────────────────────────
with st.expander("📖 What is JEPA? (Click to learn)", expanded=False):
    st.markdown(
        """
        ### The Core Idea
        
        **Current Generative AI** (like Stable Diffusion, ViT-MAE) tries to predict 
        or reconstruct **every single pixel** in an image. This requires massive 
        computation because the model needs to learn low-level details (textures, 
        colors, edges) that aren't important for understanding.
        
        **JEPA (Joint Embedding Predictive Architecture)** takes a fundamentally 
        different approach:
        
        1. **Divide** the input (image/video) into patches
        2. **Encode** each patch into a compact representation (embedding)  
        3. **Predict** the embedding of the *missing* patches — NOT the actual pixels
        
        ### Why is this more efficient?
        
        | Aspect | Generative AI | JEPA |
        |--------|--------------|------|
        | **Predicts** | Every pixel (e.g., 224×224 = 50,176 values) | Abstract embedding (~768 values) |
        | **GPU Work** | Very Heavy — reconstruct full image | Light — predict compact vectors |
        | **Energy** | High | **60-75% Less** |
        | **Quality** | Excellent at generation | Excellent at **understanding** |
        
        ### Analogy
        Think of it like this: If someone asks you "What's behind that wall?", 
        you don't need to **paint a photorealistic picture** of what's behind it. 
        You just need to **conceptually know** "there's probably a chair and a table." 
        That's what JEPA does — it thinks in concepts, not pixels.
        
        > **Source:** Assran et al., "Self-Supervised Learning from Images with a 
        > Joint-Embedding Predictive Architecture" (Meta AI, CVPR 2023)
        """
    )

# ── Key Metrics ───────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">📊 Key Findings</p>', unsafe_allow_html=True)

gen_avg = df_comp[df_comp["architecture_type"] == "Generative"]["energy_j"].mean()
jepa_avg = df_comp[df_comp["architecture_type"] == "Predictive (JEPA)"]["energy_j"].mean()
savings_pct = ((gen_avg - jepa_avg) / gen_avg) * 100

gen_co2 = df_comp[df_comp["architecture_type"] == "Generative"]["co2_grams"].mean()
jepa_co2 = df_comp[df_comp["architecture_type"] == "Predictive (JEPA)"]["co2_grams"].mean()
co2_savings = ((gen_co2 - jepa_co2) / gen_co2) * 100

gen_mem = df_comp[df_comp["architecture_type"] == "Generative"]["gpu_memory_mb"].mean()
jepa_mem = df_comp[df_comp["architecture_type"] == "Predictive (JEPA)"]["gpu_memory_mb"].mean()
mem_savings = ((gen_mem - jepa_mem) / gen_mem) * 100

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-value green">{savings_pct:.0f}%</div>'
        f'<div class="metric-label">Less Energy (JEPA vs Generative)</div>'
        f'</div>', unsafe_allow_html=True
    )
with c2:
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-value green">{co2_savings:.0f}%</div>'
        f'<div class="metric-label">Less CO₂ Emissions</div>'
        f'</div>', unsafe_allow_html=True
    )
with c3:
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-value green">{mem_savings:.0f}%</div>'
        f'<div class="metric-label">Less GPU Memory</div>'
        f'</div>', unsafe_allow_html=True
    )
with c4:
    gen_lat = df_comp[df_comp["architecture_type"] == "Generative"]["duration_s"].mean()
    jepa_lat = df_comp[df_comp["architecture_type"] == "Predictive (JEPA)"]["duration_s"].mean()
    speed_up = gen_lat / jepa_lat
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-value green">{speed_up:.1f}x</div>'
        f'<div class="metric-label">Faster Inference</div>'
        f'</div>', unsafe_allow_html=True
    )

# ── Energy Comparison Charts ─────────────────────────────────────────────────
st.markdown('<p class="section-header">⚡ Energy Comparison</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    fig_energy = bar_jepa_energy_comparison(df_comp)
    st.plotly_chart(fig_energy, use_container_width=True)
with col2:
    fig_co2 = bar_jepa_co2_comparison(df_comp)
    st.plotly_chart(fig_co2, use_container_width=True)

# ── Insight Box ───────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="insight-box">
        <strong>💡 Key Insight:</strong> JEPA models use <strong>{savings_pct:.0f}% less energy</strong> 
        than Generative models for the same vision tasks. This is because JEPA predicts 
        <em>abstract embeddings</em> (~768 numbers) instead of reconstructing 
        <em>full images</em> (~50,000 pixel values). Less math = less GPU power = less CO₂.
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Scaling Chart ─────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">📈 Scaling Efficiency</p>', unsafe_allow_html=True)
st.markdown(
    "As models get bigger, **Generative models' energy grows quadratically** "
    "(like a rocket 🚀), while **JEPA's energy grows almost linearly** (like a bicycle 🚲). "
    "This means JEPA becomes even MORE efficient at larger scales."
)

fig_scale = line_jepa_scaling(df_scale)
st.plotly_chart(fig_scale, use_container_width=True)

st.markdown(
    """
    <div class="insight-box">
        <strong>🔮 Why This Matters for the Future:</strong> As AI models keep getting bigger 
        (GPT-4 has 1.7 Trillion parameters!), the energy gap between Generative and JEPA 
        architectures will keep widening. At 5B parameters, JEPA already uses <strong>~5x less 
        energy</strong> than Generative. At 100B+ parameters, this difference could be 
        <strong>50-100x</strong> — making JEPA essential for sustainable AI.
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Detailed Table ────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">📋 Model Comparison Table</p>', unsafe_allow_html=True)

summary_table = df_comp.groupby(["model", "architecture_type"]).agg(
    Params_B=("parameters_b", "first"),
    Avg_Energy_J=("energy_j", lambda x: round(x.mean(), 1)),
    Avg_CO2_g=("co2_grams", lambda x: round(x.mean(), 4)),
    Avg_Latency_s=("duration_s", lambda x: round(x.mean(), 2)),
    Avg_GPU_Memory_MB=("gpu_memory_mb", lambda x: round(x.mean(), 0)),
    Tasks_Tested=("task", "nunique"),
).reset_index().rename(columns={
    "model": "Model",
    "architecture_type": "Architecture",
    "Params_B": "Params (B)",
    "Avg_Energy_J": "Avg Energy (J)",
    "Avg_CO2_g": "Avg CO₂ (g)",
    "Avg_Latency_s": "Avg Latency (s)",
    "Avg_GPU_Memory_MB": "GPU Memory (MB)",
    "Tasks_Tested": "Tasks",
})
st.dataframe(summary_table, use_container_width=True, hide_index=True)

# ── Takeaway ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    """
    ### 🌱 Takeaway
    
    JEPA represents a **paradigm shift** in AI architecture design:
    
    - **For Developers:** If your task is *understanding* (classification, detection, search) 
      and not *generation* (creating new images/text), JEPA models will give you the same 
      accuracy at a fraction of the energy cost.
    
    - **For the Planet:** If JEPA-style architectures become mainstream, the carbon 
      footprint of AI inference could drop by **60-75%** — equivalent to taking millions 
      of cars off the road every year.
    
    - **For Research:** JEPA shows that making AI more *human-like* (thinking in concepts, 
      not pixels) also makes it more *energy-efficient*. Intelligence and sustainability 
      can go hand-in-hand.
    
    > *"The key challenge for AI is not to generate but to understand."*  
    > — **Yann LeCun**, Meta AI
    """
)
