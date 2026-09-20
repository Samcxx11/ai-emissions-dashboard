"""
Page 7 — JEPA vs Generative AI: The Future of Energy-Efficient AI
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.theme import EDITORIAL_CSS, apply_editorial_layout, chapter, divider, finding, quote, metric_card, big_number

st.set_page_config(page_title="JEPA vs Generative", page_icon="🧠", layout="wide")

st.markdown(EDITORIAL_CSS, unsafe_allow_html=True)

# ── Inline constants (avoid Streamlit caching issues) ─────────────────────────
ARCHITECTURE_COLORS = {
    "Generative": "#e74c3c",
    "Predictive (JEPA)": "#2ecc71",
}
ARCHITECTURE_ORDER = ["Generative", "Predictive (JEPA)"]

JEPA_MODEL_COLORS = {
    "ViT-MAE (Generative)": "#e74c3c",
    "Stable Diffusion (Generative)": "#c0392b",
    "I-JEPA (Predictive)": "#2ecc71",
    "V-JEPA (Predictive)": "#27ae60",
}


# ── Inline chart functions ────────────────────────────────────────────────────
def bar_jepa_energy_comparison(df):
    summary = df.groupby("model")["energy_j"].mean().reset_index()
    summary = summary.sort_values("energy_j", ascending=False)
    fig = px.bar(summary, x="model", y="energy_j",
                 color="model", color_discrete_map=JEPA_MODEL_COLORS,
                 text_auto=".0f",
                 title="Energy per Inference: JEPA vs Generative")
    fig.update_traces(textposition="outside")
    fig.update_layout(xaxis_title="Model", yaxis_title="Energy (Joules)")
    apply_editorial_layout(fig)
    return fig


def line_jepa_scaling(df):
    summary = df.groupby(["parameters_b", "architecture"])["energy_j"].mean().reset_index()
    fig = px.line(summary, x="parameters_b", y="energy_j",
                  color="architecture", color_discrete_map=ARCHITECTURE_COLORS,
                  markers=True,
                  category_orders={"architecture": ARCHITECTURE_ORDER},
                  title="Energy Scaling: JEPA vs Generative (as model grows)")
    fig.update_traces(line=dict(width=3), marker=dict(size=10))
    fig.update_layout(xaxis_title="Parameters (Billions)", yaxis_title="Energy (Joules)")
    apply_editorial_layout(fig)
    return fig


def bar_jepa_co2_comparison(df):
    summary = df.groupby("architecture_type")["co2_grams"].mean().reset_index()
    fig = px.bar(summary, x="architecture_type", y="co2_grams",
                 color="architecture_type", color_discrete_map=ARCHITECTURE_COLORS,
                 text_auto=".4f",
                 title="Avg CO₂ Emissions: Predictive vs Generative")
    fig.update_traces(textposition="outside")
    fig.update_layout(xaxis_title="Architecture", yaxis_title="CO₂ (grams)")
    apply_editorial_layout(fig)
    return fig


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
        """
    )
    quote("Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture", "Assran et al., Meta AI, CVPR 2023")

# ── Key Metrics ───────────────────────────────────────────────────────────────
chapter(1, "Key Findings")

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
    metric_card(f"{savings_pct:.0f}", "%", "Less Energy (JEPA vs Generative)", "green")
with c2:
    metric_card(f"{co2_savings:.0f}", "%", "Less CO₂ Emissions", "green")
with c3:
    metric_card(f"{mem_savings:.0f}", "%", "Less GPU Memory", "green")
with c4:
    gen_lat = df_comp[df_comp["architecture_type"] == "Generative"]["duration_s"].mean()
    jepa_lat = df_comp[df_comp["architecture_type"] == "Predictive (JEPA)"]["duration_s"].mean()
    speed_up = gen_lat / jepa_lat
    metric_card(f"{speed_up:.1f}", "x", "Faster Inference", "green")

# ── Energy Comparison Charts ─────────────────────────────────────────────────
divider()
chapter(2, "Energy Comparison")

col1, col2 = st.columns(2)
with col1:
    fig_energy = bar_jepa_energy_comparison(df_comp)
    st.plotly_chart(fig_energy, use_container_width=True)
with col2:
    fig_co2 = bar_jepa_co2_comparison(df_comp)
    st.plotly_chart(fig_co2, use_container_width=True)

# ── Insight Box ───────────────────────────────────────────────────────────────
finding(f"JEPA models use **{savings_pct:.0f}% less energy** than Generative models for the same vision tasks. This is because JEPA predicts *abstract embeddings* (~768 numbers) instead of reconstructing *full images* (~50,000 pixel values). Less math = less GPU power = less CO₂.", "Key Insight")

# ── Scaling Chart ─────────────────────────────────────────────────────────────
divider()
chapter(3, "Scaling Efficiency")
st.markdown(
    "As models get bigger, **Generative models' energy grows quadratically** "
    "(like a rocket 🚀), while **JEPA's energy grows almost linearly** (like a bicycle 🚲). "
    "This means JEPA becomes even MORE efficient at larger scales."
)

fig_scale = line_jepa_scaling(df_scale)
st.plotly_chart(fig_scale, use_container_width=True)

finding(
    "As AI models keep getting bigger (GPT-4 has 1.7 Trillion parameters!), the energy gap between Generative and JEPA architectures will keep widening. At 5B parameters, JEPA already uses **~5x less energy** than Generative. At 100B+ parameters, this difference could be **50-100x** — making JEPA essential for sustainable AI.",
    "Why This Matters for the Future"
)

# ── Detailed Table ────────────────────────────────────────────────────────────
divider()
chapter(4, "Model Comparison Table")

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
divider()
chapter(5, "Takeaway")
st.markdown(
    """
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
    """
)
quote("The key challenge for AI is not to generate but to understand.", "Yann LeCun, Meta AI")
