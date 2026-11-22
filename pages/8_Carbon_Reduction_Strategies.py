"""
Page 8 — Carbon Reduction Strategies

A comprehensive, interactive guide showing 22 practical ways to reduce
AI inference carbon emissions — organized into 4 tiers with estimated
savings percentages and an interactive compounded savings simulator.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.constants import (
    MODEL_ORDER,
    CO2_KG_PER_CAR_MILE,
    CO2_KG_PER_TREE_PER_YEAR,
)
from utils.theme import (
    EDITORIAL_CSS,
    apply_editorial_layout,
    chapter,
    divider,
    finding,
    quote,
    metric_card,
    big_number,
)

st.set_page_config(
    page_title="Carbon Reduction Strategies",
    page_icon="♻️",
    layout="wide",
)
st.markdown(EDITORIAL_CSS, unsafe_allow_html=True)

# Extra CSS for strategy cards matching the editorial theme
st.markdown("""
<style>
    .strategy-card {
        border: 1px solid var(--divider);
        border-left: 3px solid var(--ink);
        border-radius: 2px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
        transition: border-color 0.2s;
    }
    .strategy-card:hover { border-left-color: var(--accent-green); }
    .strategy-card .s-title {
        font-family: var(--sans);
        font-weight: 600;
        font-size: 1rem;
        color: var(--ink);
        margin-bottom: 0.4rem;
    }
    .strategy-card .s-body {
        font-family: var(--sans);
        font-size: 0.92rem;
        color: var(--graphite);
        line-height: 1.65;
    }
    .s-badge {
        display: inline-block;
        font-family: var(--mono);
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        padding: 2px 8px;
        border-radius: 2px;
        margin-top: 0.6rem;
        margin-right: 6px;
    }
    .s-badge.high   { background: rgba(5,122,85,0.12); color: var(--accent-green); }
    .s-badge.medium { background: rgba(185,28,28,0.08); color: var(--accent-red); }
    .s-badge.info   { background: rgba(17,17,17,0.07); color: var(--ink); }
    .tier-label {
        font-family: var(--mono);
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--accent-green);
        border-bottom: 1px solid var(--divider);
        padding-bottom: 0.4rem;
        margin: 2rem 0 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    return pd.read_csv(os.path.join(data_dir, "inference_results.csv"))


df = load_data()


# ── Strategy Catalogue ────────────────────────────────────────────────────────
STRATEGIES = {
    "Tier 1 — Model Optimization": [
        {"icon": "📏", "title": "Right-Size Your Model",
         "body": "Match model capacity to task complexity. A 1B-parameter model handles simple Q&A, classification, and extraction just as well as a 7B model — at <strong>14× less energy</strong>. Reserve large models only for tasks that genuinely need them: creative writing, complex reasoning, multi-step coding.",
         "savings": "50–90% savings", "level": "high", "diff": "Easy"},
        {"icon": "🔧", "title": "Post-Training Quantization (FP16 → INT8 → INT4)",
         "body": "Reduce numerical precision of model weights. <strong>INT8</strong> saves ~35% energy with less than 2% quality loss. <strong>INT4</strong> saves ~58% but requires validation for your use-case. Tools: bitsandbytes, GPTQ, AWQ, GGUF (llama.cpp). No retraining needed.",
         "savings": "35–58% savings", "level": "high", "diff": "Medium"},
        {"icon": "✂️", "title": "Knowledge Distillation",
         "body": "Train a smaller 'student' model to mimic a larger 'teacher'. The student runs at a fraction of the energy while retaining 90–95% of teacher quality. Example: Gemma 2B is distilled from larger Gemini models.",
         "savings": "60–80% savings", "level": "high", "diff": "Hard"},
        {"icon": "🎯", "title": "Task-Specific Fine-Tuning",
         "body": "Instead of using a large general-purpose model, fine-tune a small model on your specific domain (customer support, legal docs, medical Q&A). A fine-tuned 1B model often outperforms a general 7B model at the specific task — while using 85% less energy per query.",
         "savings": "70–85% savings", "level": "high", "diff": "Medium"},
        {"icon": "🧹", "title": "Structured Weight Pruning",
         "body": "Remove redundant neurons and attention heads that contribute little to output quality. Structured pruning can remove 20–40% of parameters with minimal accuracy loss. Combined with quantization, savings compound significantly.",
         "savings": "20–40% savings", "level": "medium", "diff": "Hard"},
    ],
    "Tier 2 — Infrastructure & Hardware": [
        {"icon": "🌍", "title": "Choose Low-Carbon Regions",
         "body": "The same query on the same model emits <strong>90× less CO₂</strong> in Norway (hydro: 0.008 kg CO₂/kWh) than India (coal-heavy: 0.708 kg CO₂/kWh). Cloud providers let you pick deployment regions — choose ones with renewable energy.",
         "savings": "Up to 90% CO₂", "level": "high", "diff": "Easy"},
        {"icon": "⚡", "title": "Use Energy-Efficient Hardware",
         "body": "Newer GPUs are dramatically more efficient per token. An NVIDIA H100 delivers ~3× more tokens per watt than an A100, which is ~5× better than a V100. Upgrading hardware is one of the highest-impact, fastest-payback investments.",
         "savings": "40–70% savings", "level": "high", "diff": "Medium"},
        {"icon": "⏰", "title": "Time-Shift Workloads to Green Hours",
         "body": "Electricity grids are cleanest at midday solar hours in many regions. Shift batch/non-urgent workloads (embedding generation, fine-tuning, batch inference) to off-peak green hours. Tools: Google Carbon-Aware Computing API, WattTime.",
         "savings": "10–30% savings", "level": "medium", "diff": "Easy"},
        {"icon": "☁️", "title": "Optimize Data Center PUE",
         "body": "Power Usage Effectiveness (PUE) measures total energy vs. compute energy. A PUE of 1.1 (Google/AWS) wastes only 10% on cooling, while older facilities at PUE 1.6 waste 60%. Self-hosting? Use efficient cooling or choose low-PUE cloud providers.",
         "savings": "10–35% savings", "level": "medium", "diff": "Medium"},
        {"icon": "🔋", "title": "Renewable Energy Certificates / PPAs",
         "body": "Buy RECs or Power Purchase Agreements to offset electricity with clean energy. While this doesn't reduce energy consumption, it makes your grid carbon intensity effectively zero. Major cloud providers offer 100% renewable energy commitments.",
         "savings": "Up to 100% CO₂", "level": "high", "diff": "Easy (cost)"},
    ],
    "Tier 3 — Deployment & Engineering": [
        {"icon": "💾", "title": "Semantic Response Caching",
         "body": "If users frequently ask similar questions (FAQs, common lookups), cache results using vector similarity matching. A cached response uses <strong>zero inference energy</strong>. Even caching 30% of queries saves 30% of total emissions. Tools: Redis, GPTCache, LangChain caching.",
         "savings": "20–50% savings", "level": "high", "diff": "Easy"},
        {"icon": "📦", "title": "Dynamic Request Batching",
         "body": "Process multiple queries in a single forward pass. GPU utilization jumps from ~30% (single query) to ~80%+ (batched), spreading fixed energy overhead across many requests. Our data shows batch size 4 uses ~21% less energy per query than single inference.",
         "savings": "15–30% savings", "level": "medium", "diff": "Easy"},
        {"icon": "✏️", "title": "Prompt Length Compression",
         "body": "Shorter, well-crafted prompts use less energy in the prefill phase. Remove unnecessary context and instructions. Our Input Length analysis shows long prompts consume significantly more total energy due to quadratic self-attention scaling.",
         "savings": "10–25% savings", "level": "medium", "diff": "Easy"},
        {"icon": "🔀", "title": "Intelligent Cascading Model Routing",
         "body": "Use a lightweight classifier to route queries to the appropriate model. Simple questions → small model; complex questions → large model. 70–80% of enterprise queries hit the cheap small model; only the hard ones trigger the expensive large one.",
         "savings": "40–60% savings", "level": "high", "diff": "Medium"},
        {"icon": "🛑", "title": "Strict Max Output Token Limits",
         "body": "Set a tight <code>max_new_tokens</code> ceiling. Generating 2,000 tokens when 200 suffice wastes 10× the decode energy. Analyze your output length distribution and enforce limits. Use stop sequences and early stopping.",
         "savings": "15–50% savings", "level": "medium", "diff": "Easy"},
        {"icon": "🧊", "title": "Speculative Decoding",
         "body": "Use a small 'draft' model to generate candidate tokens, then verify them in parallel with the large model. This achieves 2–3× speed-up with identical output quality, reducing GPU time and energy per query. Supported in vLLM and Hugging Face Transformers.",
         "savings": "30–50% savings", "level": "high", "diff": "Medium"},
        {"icon": "⚙️", "title": "High-Throughput Inference Engines",
         "body": "Replace vanilla Hugging Face inference with optimized runtimes: <strong>vLLM</strong> (PagedAttention, 2–4× throughput), <strong>TensorRT-LLM</strong> (NVIDIA optimization), <strong>llama.cpp</strong> (CPU-efficient GGUF), or <strong>ONNX Runtime</strong>. Same model, same quality, far less energy.",
         "savings": "30–60% savings", "level": "high", "diff": "Medium"},
    ],
    "Tier 4 — Governance & Organization": [
        {"icon": "📊", "title": "Automated Emissions Telemetry",
         "body": "You can't reduce what you don't measure. Use CodeCarbon, ML CO₂ Impact, or cloud-native carbon dashboards to track emissions per model, per team, and per project. Set baselines, track trends, and make carbon a first-class KPI alongside accuracy and latency.",
         "savings": "Enables all other savings", "level": "info", "diff": "Easy"},
        {"icon": "🏷️", "title": "Carbon Budgets per Team / Project",
         "body": "Allocate a monthly CO₂ or kWh budget to each team or application. Teams that stay under budget get credit; teams that exceed it must justify or optimize. This creates natural incentive to pick efficient models and cache effectively.",
         "savings": "15–30% savings", "level": "medium", "diff": "Medium"},
        {"icon": "📋", "title": "Standardized AI Energy Model Cards",
         "body": "Before deploying any model, require documentation of its energy consumption per query, CO₂ per 1K requests, and comparison with smaller alternatives. This creates transparency and encourages teams to benchmark before they ship.",
         "savings": "Culture shift", "level": "info", "diff": "Easy"},
        {"icon": "🌱", "title": "Green AI Procurement Policy",
         "body": "Establish organizational guidelines: prefer smaller models by default, mandate quantization for production, require justification for models above a parameter threshold, and include energy efficiency in model evaluation criteria alongside accuracy.",
         "savings": "20–50% savings", "level": "high", "diff": "Easy"},
        {"icon": "🔄", "title": "Shared Internal Model Registries",
         "body": "Instead of every team fine-tuning or deploying their own model, build a shared internal model registry. One well-optimized, quantized model serving 10 teams is far more efficient than 10 teams each running their own unoptimized copy.",
         "savings": "30–60% savings", "level": "high", "diff": "Medium"},
    ],
}

all_strategies = [s for strats in STRATEGIES.values() for s in strats]


# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown('<div class="editorial-hero">', unsafe_allow_html=True)
st.markdown(big_number(8), unsafe_allow_html=True)
st.markdown('<h1><em>Carbon Reduction</em> Strategies</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">22 actionable ways to cut AI inference emissions — from quick engineering wins to organizational transformation.</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)


# ── Summary KPIs ──────────────────────────────────────────────────────────────
st.markdown(chapter(1, "AT A GLANCE"), unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
n_easy = sum(1 for s in all_strategies if s["diff"] == "Easy")
with k1:
    st.markdown(metric_card("22", "Strategies", "Covering 4 operational tiers"), unsafe_allow_html=True)
with k2:
    st.markdown(metric_card(str(n_easy), "Easy Wins", "Implementable with minimal effort"), unsafe_allow_html=True)
with k3:
    st.markdown(metric_card("4", "Tiers", "Model → Infra → Deploy → Governance"), unsafe_allow_html=True)
with k4:
    st.markdown(metric_card("98%", "Max Savings", "Compounded, stacked optimizations"), unsafe_allow_html=True)

st.markdown(divider(), unsafe_allow_html=True)


# ── Interactive Savings Simulator ─────────────────────────────────────────────
st.markdown(chapter(2, "INTERACTIVE SAVINGS SIMULATOR",
                    "Select strategies to apply and see the compounded CO₂ reduction."), unsafe_allow_html=True)

sim_col, chart_col = st.columns([1, 2])

with sim_col:
    sim_model = st.selectbox("Base model:", options=MODEL_ORDER, index=2)
    sim_queries = st.number_input("Queries per day:", min_value=1_000,
                                  max_value=10_000_000, value=100_000, step=10_000)
    st.markdown("**Apply strategies:**")
    use_right_size  = st.checkbox("Right-size model (use smallest adequate)")
    use_quant       = st.checkbox("INT8 quantization (−35%)")
    use_cache       = st.checkbox("Semantic caching — 30% hit rate (−30%)")
    use_batch       = st.checkbox("Batch scheduling, N=4 (−21%)")
    use_routing     = st.checkbox("Smart cascading routing (−50%)")
    use_region      = st.checkbox("Low-carbon region (−90% CO₂)")
    use_tokens      = st.checkbox("Limit max output tokens (−20%)")
    use_speculative = st.checkbox("Speculative decoding (−35%)")

with chart_col:
    model_data = df[df["model"] == sim_model]
    base_energy_kwh = model_data["energy_kwh"].mean()
    base_co2_g      = model_data["co2_grams"].mean()

    energy_m = 1.0
    co2_m    = 1.0
    steps = []

    if use_right_size:
        smallest_e = df[df["model"] == MODEL_ORDER[0]]["energy_kwh"].mean()
        ratio = smallest_e / base_energy_kwh
        energy_m *= ratio; co2_m *= ratio
        steps.append(("Right-size model", round((1 - ratio) * 100)))

    if use_quant:
        energy_m *= 0.65; co2_m *= 0.65
        steps.append(("INT8 quantization", 35))

    if use_cache:
        energy_m *= 0.70; co2_m *= 0.70
        steps.append(("Response caching", 30))

    if use_batch:
        energy_m *= 0.79; co2_m *= 0.79
        steps.append(("Batch scheduling", 21))

    if use_routing:
        energy_m *= 0.50; co2_m *= 0.50
        steps.append(("Smart routing", 50))

    if use_region:
        co2_m *= 0.10          # only CO₂, not energy
        steps.append(("Low-carbon region", 90))

    if use_tokens:
        energy_m *= 0.80; co2_m *= 0.80
        steps.append(("Limit output tokens", 20))

    if use_speculative:
        energy_m *= 0.65; co2_m *= 0.65
        steps.append(("Speculative decoding", 35))

    base_annual_co2 = (base_co2_g * sim_queries * 365) / 1000
    final_co2       = base_annual_co2 * co2_m
    saved_co2       = base_annual_co2 - final_co2
    pct_saved       = (1 - co2_m) * 100

    base_annual_energy = base_energy_kwh * sim_queries * 365
    saved_energy       = base_annual_energy * (1 - energy_m)

    if steps:
        measures = ["absolute"] + ["relative"] * len(steps) + ["total"]
        x_labels = ["Baseline"] + [s[0] for s in steps] + ["Optimized"]
        y_vals   = [base_annual_co2] + [-(base_annual_co2 * s[1] / 100 * co2_m) for s in steps] + [None]
        texts    = [f"{base_annual_co2:,.0f} kg"] + [f"−{s[1]}%" for s in steps] + [f"{final_co2:,.0f} kg"]

        fig = go.Figure(go.Waterfall(
            orientation="v", measure=measures, x=x_labels, y=y_vals,
            connector={"line": {"color": "rgba(17,17,17,0.15)"}},
            decreasing={"marker": {"color": "#057A55"}},
            totals={"marker": {"color": "#111111"}},
            increasing={"marker": {"color": "#B91C1C"}},
            textposition="outside", text=texts,
        ))
        apply_editorial_layout(fig)
        fig.update_layout(title="Annual CO₂ Reduction Waterfall", showlegend=False,
                          margin=dict(t=60, b=60, l=40, r=40))
        st.plotly_chart(fig, use_container_width=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("CO₂ Saved / Year",  f"{saved_co2:,.0f} kg",    delta=f"−{pct_saved:.0f}%")
        m2.metric("Energy Saved / Year", f"{saved_energy:,.0f} kWh", delta=f"−{(1-energy_m)*100:.0f}%")
        m3.metric("Trees Offset",  f"{saved_co2 / CO2_KG_PER_TREE_PER_YEAR:,.0f}", delta="per year")
    else:
        st.info("👈 Select one or more strategies on the left to see the savings waterfall.")

st.markdown(divider(), unsafe_allow_html=True)


# ── Savings Range Chart ───────────────────────────────────────────────────────
st.markdown(chapter(3, "SAVINGS POTENTIAL BY STRATEGY"), unsafe_allow_html=True)

savings_data = [
    {"Strategy": "Right-Size Model",         "Min": 50, "Max": 90, "Cat": "Model"},
    {"Strategy": "Task Fine-Tuning",          "Min": 70, "Max": 85, "Cat": "Model"},
    {"Strategy": "Knowledge Distillation",    "Min": 60, "Max": 80, "Cat": "Model"},
    {"Strategy": "INT8/INT4 Quantization",    "Min": 35, "Max": 58, "Cat": "Model"},
    {"Strategy": "Smart Routing (Cascade)",   "Min": 40, "Max": 60, "Cat": "Deployment"},
    {"Strategy": "High-Throughput Engine",    "Min": 30, "Max": 60, "Cat": "Deployment"},
    {"Strategy": "Speculative Decoding",      "Min": 30, "Max": 50, "Cat": "Deployment"},
    {"Strategy": "Response Caching",          "Min": 20, "Max": 50, "Cat": "Deployment"},
    {"Strategy": "Limit Output Tokens",       "Min": 15, "Max": 50, "Cat": "Deployment"},
    {"Strategy": "Low-Carbon Region (CO₂)",   "Min": 50, "Max": 90, "Cat": "Infra"},
    {"Strategy": "Efficient GPU (H100)",      "Min": 40, "Max": 70, "Cat": "Infra"},
    {"Strategy": "Structured Pruning",        "Min": 20, "Max": 40, "Cat": "Model"},
    {"Strategy": "Green AI Policy",           "Min": 20, "Max": 50, "Cat": "Org"},
    {"Strategy": "Batch Scheduling",          "Min": 15, "Max": 30, "Cat": "Deployment"},
    {"Strategy": "Time-Shift Workloads",      "Min": 10, "Max": 30, "Cat": "Infra"},
    {"Strategy": "Prompt Compression",        "Min": 10, "Max": 25, "Cat": "Deployment"},
    {"Strategy": "Carbon Budgets",            "Min": 15, "Max": 30, "Cat": "Org"},
]
sdf = pd.DataFrame(savings_data).sort_values("Max", ascending=True)

cat_color = {"Model": "#111111", "Deployment": "#057A55", "Infra": "#B91C1C", "Org": "#666666"}

fig2 = go.Figure()
# invisible base bar
fig2.add_trace(go.Bar(y=sdf["Strategy"], x=sdf["Min"], orientation="h",
                      marker_color="rgba(0,0,0,0)", showlegend=False, hoverinfo="skip"))
# range bar
fig2.add_trace(go.Bar(
    y=sdf["Strategy"],
    x=sdf["Max"] - sdf["Min"],
    orientation="h",
    marker_color=[cat_color[c] for c in sdf["Cat"]],
    text=[f'{r["Min"]}–{r["Max"]}%' for _, r in sdf.iterrows()],
    textposition="outside",
    hovertemplate="%{y}: %{text}<extra></extra>",
    showlegend=False,
))
# legend traces
for cat, color in cat_color.items():
    fig2.add_trace(go.Bar(y=[None], x=[None], orientation="h",
                          marker_color=color, name=cat, showlegend=True))

apply_editorial_layout(fig2)
fig2.update_layout(
    title="Estimated CO₂ Savings Range by Strategy",
    barmode="stack",
    xaxis_title="Potential CO₂ Savings (%)",
    xaxis=dict(range=[0, 105]),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    margin=dict(l=200, r=80, t=80, b=60),
    height=580,
)
st.plotly_chart(fig2, use_container_width=True)
st.markdown(divider(), unsafe_allow_html=True)


# ── Full Strategy Cards ───────────────────────────────────────────────────────
st.markdown(chapter(4, "FULL STRATEGY GUIDE"), unsafe_allow_html=True)

f1, f2 = st.columns(2)
with f1:
    tier_filter = st.selectbox("Filter by tier:", ["All"] + list(STRATEGIES.keys()))
with f2:
    diff_filter = st.selectbox("Filter by difficulty:", ["All", "Easy", "Medium", "Hard"])

for tier, strats in STRATEGIES.items():
    if tier_filter != "All" and tier != tier_filter:
        continue
    filtered = [s for s in strats if diff_filter == "All" or s["diff"] == diff_filter]
    if not filtered:
        continue

    st.markdown(f'<div class="tier-label">{tier}</div>', unsafe_allow_html=True)

    for i in range(0, len(filtered), 2):
        c1, c2 = st.columns(2)
        for col, j in zip([c1, c2], [i, i + 1]):
            if j < len(filtered):
                s = filtered[j]
                with col:
                    st.markdown(f"""
                    <div class="strategy-card">
                        <div class="s-title">{s['icon']} {s['title']}</div>
                        <div class="s-body">{s['body']}</div>
                        <span class="s-badge {s['level']}">{s['savings']}</span>
                        <span class="s-badge info">Difficulty: {s['diff']}</span>
                    </div>""", unsafe_allow_html=True)

st.markdown(divider(), unsafe_allow_html=True)


# ── At-Scale Scenario Comparison ──────────────────────────────────────────────
st.markdown(chapter(5, "COMPOUNDED IMPACT AT 1M QUERIES/DAY"), unsafe_allow_html=True)
st.markdown("If a service runs 1 million queries per day on Llama-2-7B, here is what different optimization stacks achieve annually.")

llama7b_co2_g = df[df["model"] == "Llama-2-7B (Real)"]["co2_grams"].mean() if "Llama-2-7B (Real)" in df["model"].values else df["co2_grams"].mean()
base_1m = (llama7b_co2_g * 1_000_000 * 365) / 1000

scenarios = [
    {"label": "No Optimization",             "mult": 1.000, "color": "#B91C1C"},
    {"label": "+ INT8 Quantization",          "mult": 0.650, "color": "#c0392b"},
    {"label": "+ Caching + Batching",         "mult": 0.650 * 0.70 * 0.79, "color": "#e67e22"},
    {"label": "+ Smart Routing",              "mult": 0.650 * 0.70 * 0.79 * 0.50, "color": "#f39c12"},
    {"label": "+ Renewable Region (Full Stack)", "mult": 0.650 * 0.70 * 0.79 * 0.50 * 0.10, "color": "#057A55"},
]

fig3 = go.Figure()
for s in scenarios:
    co2 = base_1m * s["mult"]
    fig3.add_trace(go.Bar(
        x=[s["label"]], y=[co2],
        marker_color=s["color"],
        text=[f"{co2:,.0f} kg"],
        textposition="outside",
        showlegend=False,
    ))

apply_editorial_layout(fig3)
fig3.update_layout(
    title="Annual CO₂: Llama-2-7B at 1M Queries/Day",
    yaxis_title="CO₂ (kg/year)",
    margin=dict(t=60, b=100),
    height=440,
)
st.plotly_chart(fig3, use_container_width=True)

best_co2  = base_1m * scenarios[-1]["mult"]
saved_all = base_1m - best_co2
pct_all   = saved_all / base_1m * 100
st.markdown(finding(
    f"The full optimization stack reduces annual CO₂ from **{base_1m:,.0f} kg** to just "
    f"**{best_co2:,.0f} kg** — a **{pct_all:.1f}% reduction** equivalent to planting "
    f"**{saved_all / CO2_KG_PER_TREE_PER_YEAR:,.0f} trees** per year or eliminating "
    f"**{saved_all / CO2_KG_PER_CAR_MILE:,.0f} car miles**.",
    "COMPOUNDED IMPACT"
), unsafe_allow_html=True)

st.markdown(divider(), unsafe_allow_html=True)
st.markdown(quote(
    "The greenest computation is the one you don't have to do. "
    "Cache it, shrink it, or skip it.",
    "Green AI Engineering Principle"
), unsafe_allow_html=True)

st.caption("Savings estimates based on published research and industry benchmarks · Actual savings vary by workload and implementation.")
