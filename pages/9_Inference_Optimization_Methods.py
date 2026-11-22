"""
Chapter 09 — Top 3 Inference-Time Carbon Reduction Methods

Practical evaluation framework for Automatic Prefix Caching (APC),
SLO-Aware GPU Dynamic Voltage and Frequency Scaling (DVFS), and
Sliding Window Attention (SWA) on the local Apple M4 platform.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.constants import (
    GRID_CARBON_INTENSITY_KG_PER_KWH,
    CO2_KG_PER_TREE_PER_YEAR,
    INFERENCE_OPT_PAPERS,
    LOCAL_HARDWARE,
)
from utils.hardware import get_live_hardware_stats, run_m4_attention_benchmark
from utils.theme import (
    EDITORIAL_CSS,
    chapter,
    divider,
    finding,
    quote,
    metric_card,
    formula_box,
)

st.set_page_config(
    page_title="Inference Optimization Methods",
    layout="wide",
)

st.markdown(EDITORIAL_CSS, unsafe_allow_html=True)

HW = LOCAL_HARDWARE


# ═══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(
    chapter(
        9,
        "INFERENCE-TIME CARBON REDUCTION",
        "Three complementary methods targeting distinct layers of the inference stack",
    ),
    unsafe_allow_html=True,
)

st.markdown(
    "<h1>Top 3 Methods for Reducing Carbon in LLM Inference</h1>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <p class="hero-subtitle">
    Three complementary optimization methods &mdash; Automatic Prefix Caching, SLO-Aware GPU DVFS,
    and Sliding Window Attention &mdash; target different sources of inference energy at the runtime,
    hardware, and architectural levels respectively. This chapter provides practical implementation
    guidance calibrated to the local experimental platform.
    </p>
    """,
    unsafe_allow_html=True,
)


# ── Live Laptop Hardware Telemetry (Apple M4) ─────────────────────────────────
st.markdown(
    "<div class='chapter-marker'>LIVE ON-DEVICE HARDWARE TELEMETRY</div>",
    unsafe_allow_html=True,
)

live_hw = get_live_hardware_stats()

hw_c1, hw_c2, hw_c3, hw_c4 = st.columns(4)

with hw_c1:
    st.markdown(
        metric_card(
            live_hw["chip"],
            "PROCESSOR",
            f"{live_hw['model']} · 10 Cores (4P + 6E)",
        ),
        unsafe_allow_html=True,
    )

with hw_c2:
    cpu_load = live_hw["cpu_percent_total"]
    cpu_class = "green" if cpu_load < 30 else ("amber" if cpu_load < 70 else "red")
    st.markdown(
        metric_card(
            f"{cpu_load:.1f}%",
            "LIVE CPU LOAD",
            f"10 Cores active · Est. Package: {live_hw['current_estimated_power_w']:.1f}W",
            color_class=cpu_class,
        ),
        unsafe_allow_html=True,
    )

with hw_c3:
    mem_used = live_hw["mem_used_gb"]
    mem_total = live_hw["mem_total_gb"]
    mem_pct = live_hw["mem_percent"]
    st.markdown(
        metric_card(
            f"{mem_used:.1f} GB",
            "UNIFIED MEMORY USED",
            f"{mem_pct:.1f}% of {mem_total:.1f} GB LPDDR5X (120 GB/s)",
        ),
        unsafe_allow_html=True,
    )

with hw_c4:
    batt_pct = f"{live_hw['battery_percent']}%" if live_hw['battery_percent'] is not None else "AC Mode"
    batt_status = f"{live_hw['drawing_from']} ({'Charging' if live_hw['is_charging'] else 'Connected'})"
    st.markdown(
        metric_card(
            batt_pct,
            "BATTERY & POWER",
            batt_status,
            color_class="green" if live_hw["is_charging"] else "",
        ),
        unsafe_allow_html=True,
    )

with st.expander("Inspect Live Apple M4 Core & Memory Telemetry", expanded=False):
    t_c1, t_c2 = st.columns(2)
    with t_c1:
        st.markdown("**Per-Core Processor Utilization (Apple M4):**")
        per_core = live_hw.get("cpu_percent_per_core", [])
        for idx, p in enumerate(per_core):
            core_type = "Performance Core (P)" if idx < 4 else "Efficiency Core (E)"
            st.progress(min(max(p / 100.0, 0.0), 1.0), text=f"Core {idx+1} ({core_type}): {p:.1f}%")
    with t_c2:
        st.markdown("**Memory & Accelerator Subsystem:**")
        st.markdown(
            f"""
            - **SoC Architecture:** Apple M4 (Mac16,13) with unified cache and memory fabric
            - **GPU Accelerator:** {live_hw['gpu_name']} with Metal 4 API
            - **PyTorch MPS Backend:** `{'Active (GPU Accelerated)' if live_hw['mps_available'] else 'CPU Fallback'}`
            - **Neural Engine:** {live_hw['neural_engine']}
            - **Unified Memory Available:** {live_hw['mem_available_gb']:.2f} GB (free for KV cache & weight tensors)
            - **Hardware Profiling Method:** `powermetrics` (Apple Silicon SoC registers) & `psutil`
            """
        )

st.markdown(
    finding(
        f"<strong>Hardware Platform Confirmed:</strong> {live_hw['chip']} unified memory architecture "
        f"({live_hw['mem_total_gb']:.0f} GB LPDDR5X @ {live_hw['memory_bandwidth']}). "
        f"All 3 optimization methods below are evaluated against this specific hardware profile, "
        "leveraging Apple Metal Performance Shaders (MPS) for on-device tensor execution.",
        label="REAL HARDWARE TELEMETRY ACTIVE",
        warning=False,
    ),
    unsafe_allow_html=True,
)


# ── Comparative Summary Table ─────────────────────────────────────────────────
st.markdown(divider(), unsafe_allow_html=True)

st.markdown(
    "<div class='chapter-marker'>METHOD OVERVIEW</div>"
    "<h3>Comparative Architecture of Three Optimization Methods</h3>",
    unsafe_allow_html=True,
)

overview_df = pd.DataFrame(
    {
        "Dimension": [
            "Main Target",
            "Optimization Layer",
            "Best-Suited Workload",
            "Main Energy Mechanism",
            "Main Trade-Off",
            "Implementation Difficulty",
            "Key Metric",
        ],
        "Automatic Prefix Caching": [
            "Repeated prompt prefill",
            "Runtime / software",
            "Many requests share long prefixes",
            "Avoids repeated computation",
            "Cache memory and hit rate",
            "Low to Medium",
            "Cache-hit rate and energy/request",
        ],
        "SLO-Aware GPU DVFS": [
            "GPU operating power",
            "Hardware / power management",
            "Variable load with latency headroom",
            "Reduces power when headroom exists",
            "Latency versus energy",
            "Medium to High",
            "Energy with SLO compliance",
        ],
        "Sliding Window Attention": [
            "Attention computation and KV cache",
            "Model architecture",
            "Long-context inference",
            "Restricts attention and cache work",
            "Long-range context and model compatibility",
            "Medium to High",
            "Energy vs. context length and quality",
        ],
    }
)

st.dataframe(overview_df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  METHOD 1 — AUTOMATIC PREFIX CACHING (APC)
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(divider(), unsafe_allow_html=True)

st.markdown(
    "<div class='chapter-marker'>METHOD 1 — RUNTIME OPTIMIZATION</div>"
    "<h3>Automatic Prefix Caching (APC)</h3>",
    unsafe_allow_html=True,
)

apc_c1, apc_c2 = st.columns([3, 2])

with apc_c1:
    st.markdown(
        """
        Large Language Models repeatedly process input tokens during the **prefill** stage
        before generating an answer. In many deployments, different requests share an
        identical prompt prefix &mdash; a system prompt, a shared document, conversation
        history, or standard templates.

        **Automatic Prefix Caching** retains the KV-cache blocks produced by earlier requests
        and reuses them when a new request matches the same token prefix, skipping redundant
        prefill computation entirely.

        **How the KV Cache works:**
        1. The prompt is tokenized and processed during prefill.
        2. Attention layers produce key (K) and value (V) tensors.
        3. These tensors are stored in the KV cache.
        4. During autoregressive decoding, the model reuses stored K/V tensors.
        5. With APC, cached blocks from prior requests are reused on cache hits.
        """
    )

with apc_c2:
    st.markdown(
        """
        <div class="compare-panel" style="border-left: 3px solid var(--accent-green);">
            <div class="compare-tag modern">CACHE HIT SCENARIO</div>
            <div style="font-size: 0.88rem; color: var(--graphite); line-height: 1.7;">
                <p><strong>Without APC:</strong> Each of 100 requests re-processes the same
                2,000-token document prefix. Total: 200,000 redundant prefill tokens.</p>
                <p><strong>With APC:</strong> First request creates the KV cache.
                Remaining 99 requests reuse it. Only new question tokens are processed.</p>
                <p style="font-family: var(--mono); font-size: 11px; color: var(--accent-green); margin-top: 0.5rem;">
                    PREFILL TOKENS SAVED: ~198,000 (99%)
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── APC vs Semantic Caching ───────────────────────────────────────────────────
st.markdown(
    "<div class='chapter-marker'>DISAMBIGUATION</div>",
    unsafe_allow_html=True,
)

dis_c1, dis_c2 = st.columns(2)

with dis_c1:
    st.markdown(
        """
        <div class="compare-panel" style="border-left: 3px solid var(--ink);">
            <div class="compare-tag modern">AUTOMATIC PREFIX CACHING</div>
            <div style="font-size: 0.88rem; color: var(--graphite); line-height: 1.7;">
                <p><strong>Reuses:</strong> Internal KV-cache state for matching token prefixes</p>
                <p><strong>Matching:</strong> Exact token-prefix / block matching</p>
                <p><strong>Computation avoided:</strong> Repeated prefill for the cached prefix</p>
                <p><strong>Output:</strong> Model continues computation using cached state</p>
                <p><strong>Main concern:</strong> KV-cache capacity and prefix match rate</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with dis_c2:
    st.markdown(
        """
        <div class="compare-panel" style="border-left: 3px solid var(--accent-red);">
            <div class="compare-tag legacy">SEMANTIC CACHING</div>
            <div style="font-size: 0.88rem; color: var(--graphite); line-height: 1.7;">
                <p><strong>Reuses:</strong> A previously generated answer for a similar query</p>
                <p><strong>Matching:</strong> Similarity or equivalence of queries</p>
                <p><strong>Computation avoided:</strong> Potentially the entire model call</p>
                <p><strong>Output:</strong> System returns a stored answer</p>
                <p><strong>Main concern:</strong> Correctness of similarity matching and freshness</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── APC Energy Causal Chain ──────────────────────────────────────────────────
st.markdown(
    finding(
        "APC primarily reduces work in the <strong>prefill stage</strong>. It does not reduce "
        "the work needed to generate each new output token during decoding. Its benefit is greatest "
        "when requests share long prefixes and the system achieves a high cache-hit rate. "
        "APC may have limited impact when prompts are short, requests lack shared prefixes, "
        "or most inference time is spent generating long answers.",
        label="ENERGY CAUSAL CHAIN",
    ),
    unsafe_allow_html=True,
)

# ── APC Implementation on Apple M4 ───────────────────────────────────────────
st.markdown(
    "<div class='chapter-marker'>PRACTICAL IMPLEMENTATION ON APPLE M4</div>",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="compare-panel" style="padding: 1.5rem;">
        <div class="compare-tag modern">LOCAL PLATFORM: {HW['chip']} / {HW['memory_gb']} GB</div>
        <div style="font-size: 0.88rem; color: var(--graphite); line-height: 1.8;">
            <p><strong>Framework:</strong> Use <code>llama.cpp</code> (Metal backend) or <code>mlx</code>
            (Apple MLX framework) for local inference. vLLM's native prefix caching requires CUDA GPUs,
            so the Apple M4 platform uses alternative implementations.</p>
            <p><strong>Model Selection:</strong> Quantized models (Q4_K_M or Q8_0) that fit within
            16 GB unified memory. Recommended: Llama-2-7B-Q4, Mistral-7B-Q4, or TinyLlama-1.1B.</p>
            <p><strong>Energy Measurement:</strong> Use <code>sudo powermetrics --samplers cpu_power,gpu_power
            --sample-interval 1000</code> for hardware-level power sampling, or Apple Instruments with
            the Energy Log template.</p>
            <p><strong>KV Cache on Metal:</strong> With <code>llama.cpp</code>, enable prompt caching via
            <code>--prompt-cache &lt;path&gt;</code> for single-session prefix reuse, or use the server mode
            with slot-based caching for multi-request scenarios.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    **Suggested Experiment Protocol:**

    1. Prepare a workload with one fixed 2,000-token system prompt and 50 distinct questions.
    2. Prepare a control workload with 50 unique prefixes (no sharing).
    3. Run each workload with prefix caching disabled, then enabled.
    4. Keep model, quantization, output-token limit (`max_tokens=200`), and sampling settings identical.
    5. Record total energy (via `powermetrics`), energy per request, prefill latency, TTFT, throughput.
    6. Warm-up runs excluded from measurement.
    """
)

st.markdown(
    formula_box(
        "APC Energy Reduction Hypothesis",
        r"\text{Energy Reduction (\%)} = \frac{E_{\text{no\_cache}} - E_{\text{APC}}}{E_{\text{no\_cache}}} \times 100",
        "Expected to be significant when cache-hit rate is high and shared prefix is long. "
        "Benefit diminishes for unique prompts or decode-heavy workloads.",
        tag="HYPOTHESIS H1",
    ),
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  METHOD 2 — SLO-AWARE GPU DVFS
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(divider(), unsafe_allow_html=True)

st.markdown(
    "<div class='chapter-marker'>METHOD 2 — HARDWARE POWER MANAGEMENT</div>"
    "<h3>SLO-Aware GPU Dynamic Voltage and Frequency Scaling (DVFS)</h3>",
    unsafe_allow_html=True,
)

dvfs_c1, dvfs_c2 = st.columns([3, 2])

with dvfs_c1:
    st.markdown(
        """
        GPUs consume substantial electrical power during LLM inference. GPU power depends on
        workload utilization, voltage, clock frequency, memory activity, and architecture.

        A GPU configured for maximum performance may complete a request faster than required
        by the application's latency target. **Dynamic Voltage and Frequency Scaling (DVFS)**
        adjusts the processor's voltage and clock frequency to match workload demands.

        An **SLO-aware** approach makes these adjustments while respecting a Service-Level
        Objective such as maximum response latency or a required latency percentile (P95/P99).

        **Key terms:**
        - **Voltage:** Electrical potential to GPU circuitry. Lower voltage reduces dynamic power.
        - **Frequency:** Clock rate. Lower frequency reduces operation rate.
        - **SLO:** A measurable performance target (e.g., TTFT < 500ms, P95 latency < 2s).
        - **Power:** Instantaneous energy rate (Watts).
        - **Energy:** Power integrated over time (Joules or Wh).
        """
    )

with dvfs_c2:
    st.markdown(
        """
        <div class="compare-panel" style="border-left: 3px solid var(--accent-green);">
            <div class="compare-tag modern">DVFS OPERATING SCENARIO</div>
            <div style="font-size: 0.88rem; color: var(--graphite); line-height: 1.7;">
                <p><strong>SLO:</strong> 2-second maximum response time</p>
                <p><strong>High clock:</strong> Request completes in 0.8s</p>
                <p><strong>Low clock:</strong> Request completes in 1.2s</p>
                <p>Both satisfy the 2s SLO. If the power reduction exceeds the
                execution time increase, total energy decreases.</p>
                <p style="font-family: var(--mono); font-size: 11px; color: var(--accent-green); margin-top: 0.5rem;">
                    HEADROOM: 0.8s unused latency budget available for DVFS
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── DVFS Power Formula ────────────────────────────────────────────────────────
st.markdown(
    formula_box(
        "Dynamic CMOS Power Approximation",
        r"P_{\text{dynamic}} = \alpha \cdot C \cdot V^{2} \cdot f",
        "Where alpha is the switching activity factor, C is effective capacitance, "
        "V is supply voltage, and f is clock frequency. This illustrates why voltage "
        "and frequency matter but is not a complete GPU power model (omits static/leakage "
        "power, memory, cooling).",
        tag="POWER MODEL",
    ),
    unsafe_allow_html=True,
)

st.markdown(
    formula_box(
        "Total Energy over Workload",
        r"E = \int_{0}^{T} P(t)\,dt \;\approx\; P_{\text{avg}} \times T",
        "Lower frequency may increase execution time T. Therefore, lower power does not "
        "necessarily mean lower total energy. The experiment must measure energy over the "
        "complete workload, not just instantaneous power.",
        tag="ENERGY INTEGRAL",
    ),
    unsafe_allow_html=True,
)

# ── DVFS on Apple M4 ─────────────────────────────────────────────────────────
st.markdown(
    "<div class='chapter-marker'>PRACTICAL IMPLEMENTATION ON APPLE M4</div>",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="compare-panel" style="padding: 1.5rem;">
        <div class="compare-tag modern">LOCAL PLATFORM: {HW['chip']} / {HW['gpu_type']}</div>
        <div style="font-size: 0.88rem; color: var(--graphite); line-height: 1.8;">
            <p><strong>DVFS on Apple Silicon:</strong> Apple M4 manages voltage and frequency
            internally through its power management controller. Unlike NVIDIA GPUs with
            <code>nvidia-smi</code> clock controls, Apple Silicon does not expose direct
            user-controllable DVFS interfaces.</p>
            <p><strong>Practical Approach:</strong> Instead of direct DVFS, use macOS
            <code>powermetrics</code> to observe the chip's autonomous frequency scaling
            under different workload profiles (light vs. heavy batching). Record CPU/GPU
            cluster frequencies alongside power and energy measurements.</p>
            <p><strong>SLO Simulation:</strong> Define application-level SLOs (e.g., TTFT &lt; 500ms)
            and test whether lighter quantization (INT4 vs INT8) or reduced batch sizes create
            measurable power differences while maintaining the SLO. This provides analogous
            insight to explicit DVFS control.</p>
            <p><strong>Alternative:</strong> If a Linux machine with an NVIDIA GPU (e.g., cloud
            instance with A100/T4) is available, use <code>nvidia-smi -lgc &lt;min,max&gt;</code>
            to lock GPU clocks at different frequencies and measure energy with NVML.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    finding(
        "On Apple M4, direct DVFS control is not exposed to the user. The chip manages "
        "voltage/frequency autonomously. For the project, observe and report the chip's "
        "autonomous frequency behaviour under varying inference loads using <code>powermetrics</code>. "
        "If explicit DVFS experimentation is required, consider using an NVIDIA GPU via cloud.",
        label="PLATFORM ADAPTATION",
        warning=True,
    ),
    unsafe_allow_html=True,
)

st.markdown(
    formula_box(
        "SLO-Aware DVFS Energy Reduction Hypothesis",
        r"\text{Energy Reduction (\%)} = \frac{E_{\text{default}} - E_{\text{DVFS}}}{E_{\text{default}}} \times 100",
        "Literature reports 22.8% to 45.5% reduction on NVIDIA hardware under SLO constraints. "
        "On Apple M4, the autonomous power management provides analogous but non-controllable scaling.",
        tag="HYPOTHESIS H2",
    ),
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  METHOD 3 — SLIDING WINDOW ATTENTION (SWA)
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(divider(), unsafe_allow_html=True)

st.markdown(
    "<div class='chapter-marker'>METHOD 3 — MODEL ARCHITECTURE</div>"
    "<h3>Sliding Window Attention (SWA)</h3>",
    unsafe_allow_html=True,
)

swa_c1, swa_c2 = st.columns([3, 2])

with swa_c1:
    st.markdown(
        """
        Self-attention is a core component of Transformer-based LLMs. In standard
        **full causal attention**, each token can attend to all preceding tokens.
        As context length increases, attention computation and KV-cache storage grow
        quadratically.

        **Sliding Window Attention** limits each token's attention to a fixed local
        window of w preceding tokens. Instead of attending to the entire history,
        the model attends to a bounded local region.

        **Example with window size w=4:**

        Sequence: `A B C D E F G H I J`

        Token `J` attends to: `G H I J` (4 tokens)

        Instead of: `A B C D E F G H I J` (10 tokens)

        For a fixed window w, local attention work grows approximately linearly
        with sequence length n (proportional to n*w) rather than the
        quadratic n-squared interaction count of full attention.
        """
    )

with swa_c2:
    st.markdown(
        """
        <div class="compare-panel" style="border-left: 3px solid var(--accent-green);">
            <div class="compare-tag modern">KV-CACHE SAVINGS</div>
            <div style="font-size: 0.88rem; color: var(--graphite); line-height: 1.7;">
                <p><strong>Full Attention:</strong> KV cache retains all n tokens.
                Memory grows linearly with context length.</p>
                <p><strong>Sliding Window:</strong> KV cache for SWA layers is bounded
                by window size w. Memory is capped regardless of total sequence length.</p>
                <p><strong>Benefits:</strong></p>
                <ul style="margin: 0.3rem 0; padding-left: 1.2rem;">
                    <li>Reduced KV-cache memory</li>
                    <li>Lower memory allocation pressure</li>
                    <li>Reduced memory traffic</li>
                    <li>More concurrent requests under same memory budget</li>
                </ul>
                <p style="font-family: var(--mono); font-size: 11px; color: var(--accent-green); margin-top: 0.5rem;">
                    MISTRAL 7B WINDOW SIZE: 4,096 tokens
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Attention Formulas ────────────────────────────────────────────────────────
st.markdown(
    formula_box(
        "Standard Scaled Dot-Product Attention",
        r"\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}} + M\right) V",
        "Where Q, K, V are query, key, and value matrices, d_k is the key dimension, "
        "and M is the causal mask preventing attention to future tokens. For full attention, "
        "the score matrix grows quadratically with sequence length.",
        tag="FULL ATTENTION",
    ),
    unsafe_allow_html=True,
)

st.markdown(
    formula_box(
        "Sliding Window Attention Complexity",
        r"\text{Interactions}_{\text{SWA}} = n \cdot w \quad \text{vs.} \quad \text{Interactions}_{\text{full}} = \frac{n(n+1)}{2}",
        "For long sequences where n >> w, SWA reduces the number of attention interactions "
        "from O(n-squared) to O(n*w). The energy and memory savings become more pronounced "
        "as context length increases beyond the window size.",
        tag="COMPLEXITY COMPARISON",
    ),
    unsafe_allow_html=True,
)

# ── SWA on Apple M4 ──────────────────────────────────────────────────────────
st.markdown(
    "<div class='chapter-marker'>PRACTICAL IMPLEMENTATION ON APPLE M4</div>",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="compare-panel" style="padding: 1.5rem;">
        <div class="compare-tag modern">LOCAL PLATFORM: {HW['chip']} / {HW['memory_gb']} GB UNIFIED MEMORY</div>
        <div style="font-size: 0.88rem; color: var(--graphite); line-height: 1.8;">
            <p><strong>Model Selection:</strong> Use <strong>Mistral 7B</strong> (which natively uses
            SWA with window=4,096) as the treatment model. Use <strong>Llama-2-7B</strong> (full
            causal attention) as the baseline. Both should be quantized to Q4_K_M to fit within
            16 GB unified memory.</p>
            <p><strong>Inference Runtime:</strong> <code>llama.cpp</code> with Metal backend supports
            both architectures. Use identical <code>--ctx-size</code> values and generation parameters.</p>
            <p><strong>Experiment Design:</strong> Test short (256), medium (1,024), and long (4,096+)
            context lengths. Measure energy via <code>powermetrics</code>. Compare prefill energy,
            decode energy, peak memory, and output quality.</p>
            <p><strong>Quality Evaluation:</strong> Include tasks requiring both local context (recent
            sentences) and long-range dependencies (early document sections). SWA may degrade quality
            on tasks requiring information from outside the attention window.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    finding(
        "SWA is <strong>not a toggle</strong> that can be enabled on any model. The model must have "
        "been designed and trained with sliding-window attention. Changing the attention pattern of "
        "a full-attention model at inference time will change its behaviour and output quality. "
        "For a fair comparison, use architecturally compatible models (e.g., Mistral 7B vs. Llama-2-7B) "
        "and acknowledge that differences in training data and architecture are confounding variables.",
        label="ARCHITECTURAL CONSTRAINT",
        warning=True,
    ),
    unsafe_allow_html=True,
)

st.markdown(
    formula_box(
        "SWA Energy Reduction Hypothesis",
        r"\text{Energy Reduction (\%)} = \frac{E_{\text{full\_attn}} - E_{\text{SWA}}}{E_{\text{full\_attn}}} \times 100",
        "Expected to show greater benefit as input length increases beyond the window size. "
        "For short prompts (n < w), the difference may be negligible.",
        tag="HYPOTHESIS H3",
    ),
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  INTERACTIVE ENERGY SIMULATOR
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(divider(), unsafe_allow_html=True)

st.markdown(
    "<div class='chapter-marker'>HYPOTHETICAL ENERGY MODELLING</div>"
    "<h3>Interactive Inference Optimization Simulator</h3>",
    unsafe_allow_html=True,
)

st.markdown(
    finding(
        "The numbers below are <strong>hypothetical illustrations</strong> for exploring the "
        "research hypotheses. They must not be cited as measured results. Each method requires "
        "controlled empirical measurement on the actual hardware platform.",
        label="IMPORTANT DISCLAIMER",
        warning=True,
    ),
    unsafe_allow_html=True,
)

sim_col1, sim_col2 = st.columns([1, 2])

with sim_col1:
    st.markdown("**Baseline Configuration:**")

    baseline_energy_per_req = st.slider(
        "Baseline energy per request (Wh):",
        min_value=0.001,
        max_value=0.100,
        value=0.012,
        step=0.001,
        format="%.3f",
        help="Average energy per inference request on Apple M4 (typical range for 7B Q4 models).",
    )

    daily_requests = st.select_slider(
        "Daily request volume:",
        options=[100, 500, 1_000, 5_000, 10_000, 50_000, 100_000],
        value=10_000,
        format_func=lambda x: f"{x:,.0f}",
    )

    st.markdown("<br>**Method Parameters:**", unsafe_allow_html=True)

    apc_hit_rate = st.slider(
        "APC cache-hit rate (%):",
        min_value=0,
        max_value=95,
        value=60,
        step=5,
    )

    apc_prefill_share = st.slider(
        "Prefill share of total energy (%):",
        min_value=10,
        max_value=70,
        value=40,
        step=5,
    )

    dvfs_reduction_pct = st.slider(
        "DVFS energy reduction (%):",
        min_value=0,
        max_value=50,
        value=25,
        step=5,
        help="Literature reports 22.8-45.5%. Apple M4 autonomous scaling may differ.",
    )

    swa_long_ctx_reduction = st.slider(
        "SWA energy reduction for long contexts (%):",
        min_value=0,
        max_value=40,
        value=20,
        step=5,
    )

    carbon_intensity = st.number_input(
        "Grid carbon intensity (kgCO2e/kWh):",
        min_value=0.01,
        max_value=1.5,
        value=float(GRID_CARBON_INTENSITY_KG_PER_KWH),
        step=0.01,
    )

with sim_col2:
    # ── Calculations ──────────────────────────────────────────────────────────
    annual_requests = daily_requests * 365
    baseline_annual_kwh = baseline_energy_per_req * annual_requests / 1000.0
    baseline_annual_co2 = baseline_annual_kwh * carbon_intensity

    # APC savings: only on prefill portion, only on cache hits
    apc_saving = baseline_energy_per_req * (apc_prefill_share / 100.0) * (apc_hit_rate / 100.0)
    apc_energy = baseline_energy_per_req - apc_saving

    # DVFS savings: on the remaining energy after APC
    dvfs_saving = apc_energy * (dvfs_reduction_pct / 100.0)
    dvfs_energy = apc_energy - dvfs_saving

    # SWA savings: on attention/prefill portion of remaining
    swa_saving = dvfs_energy * (swa_long_ctx_reduction / 100.0)
    combined_energy = dvfs_energy - swa_saving

    total_reduction_pct = ((baseline_energy_per_req - combined_energy) / baseline_energy_per_req) * 100

    combined_annual_kwh = combined_energy * annual_requests / 1000.0
    combined_annual_co2 = combined_annual_kwh * carbon_intensity
    saved_co2_kg = baseline_annual_co2 - combined_annual_co2

    # ── Waterfall Chart ───────────────────────────────────────────────────────
    stages = ["Baseline", "After APC", "After DVFS", "After SWA"]
    energies = [
        baseline_energy_per_req * 1000,  # mWh
        apc_energy * 1000,
        dvfs_energy * 1000,
        combined_energy * 1000,
    ]

    fig_wf = go.Figure()

    fig_wf.add_trace(
        go.Waterfall(
            name="Energy",
            orientation="v",
            measure=["absolute", "relative", "relative", "total"],
            x=stages,
            y=[
                energies[0],
                -(energies[0] - energies[1]),
                -(energies[1] - energies[2]),
                None,
            ],
            connector={"line": {"color": "rgba(17,17,17,0.2)"}},
            decreasing={"marker": {"color": "#057A55"}},
            increasing={"marker": {"color": "#B91C1C"}},
            totals={"marker": {"color": "#2563EB"}},
            textposition="outside",
            text=[
                f"{energies[0]:.1f} mWh",
                f"-{energies[0] - energies[1]:.1f}",
                f"-{energies[1] - energies[2]:.1f}",
                f"{energies[3]:.1f} mWh",
            ],
            textfont=dict(color="#000000", size=11, family="JetBrains Mono, monospace"),
        )
    )

    fig_wf.update_layout(
        title=dict(
            text="Compounded Energy Reduction per Request (mWh) -- Hypothetical",
            font=dict(family="Fraunces, Georgia, serif", size=16, color="#000000"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(247,245,240,0.5)",
        font=dict(family="Inter, sans-serif", color="#000000", size=12),
        xaxis=dict(
            gridcolor="rgba(17,17,17,0.08)",
            tickfont=dict(color="#000000"),
        ),
        yaxis=dict(
            title=dict(text="Energy (mWh per request)", font=dict(color="#000000")),
            gridcolor="rgba(17,17,17,0.08)",
            tickfont=dict(color="#000000"),
        ),
        margin=dict(l=60, r=30, t=50, b=50),
        showlegend=False,
        hoverlabel=dict(
            bgcolor="#000000",
            font_size=12,
            font_family="Inter, sans-serif",
            font_color="#FFFFFF",
        ),
    )

    st.plotly_chart(fig_wf, use_container_width=True)

    # ── Summary Metrics ───────────────────────────────────────────────────────
    met_c1, met_c2, met_c3, met_c4 = st.columns(4)

    with met_c1:
        st.markdown(
            metric_card(
                f"{total_reduction_pct:.1f}%",
                "COMPOUNDED REDUCTION",
                "Hypothetical combined energy saving per request.",
                color_class="green" if total_reduction_pct > 0 else "red",
            ),
            unsafe_allow_html=True,
        )

    with met_c2:
        st.markdown(
            metric_card(
                f"{baseline_annual_kwh - combined_annual_kwh:,.1f}",
                "KWH SAVED / YEAR",
                f"At {daily_requests:,} requests/day over 365 days.",
            ),
            unsafe_allow_html=True,
        )

    with met_c3:
        st.markdown(
            metric_card(
                f"{saved_co2_kg:,.2f}",
                "KG CO2e AVOIDED",
                f"Using grid intensity of {carbon_intensity} kgCO2e/kWh.",
                color_class="green" if saved_co2_kg > 0 else "red",
            ),
            unsafe_allow_html=True,
        )

    with met_c4:
        trees = saved_co2_kg / CO2_KG_PER_TREE_PER_YEAR if saved_co2_kg > 0 else 0
        st.markdown(
            metric_card(
                f"{trees:,.1f}",
                "TREE-YEARS EQUIVALENT",
                "Mature trees for one year to offset equivalent carbon.",
            ),
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  LIVE ON-DEVICE APPLE M4 METAL BENCHMARK
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(divider(), unsafe_allow_html=True)

st.markdown(
    "<div class='chapter-marker'>LIVE EMPIRICAL BENCHMARK</div>"
    "<h3>Execute Real Tensor Operations on Apple M4 GPU</h3>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <p style="color: var(--graphite); font-size: 0.95rem; line-height: 1.6;">
    Unlike static simulations, this interactive module executes real transformer attention tensor operations
    directly on your <strong>Apple M4 GPU via PyTorch Metal Performance Shaders (MPS)</strong>.
    It empirically benchmarks full causal attention vs. Automatic Prefix Caching (reusing prefix KV tensors)
    vs. Sliding Window Attention (bounded local window mask) under your laptop's current thermal and memory state.
    </p>
    """,
    unsafe_allow_html=True,
)

bench_c1, bench_c2 = st.columns([1, 2])

with bench_c1:
    st.markdown("**Benchmark Configuration:**")
    bench_seq = st.select_slider(
        "Total Sequence Length (tokens):",
        options=[256, 512, 1024, 2048],
        value=512,
        help="Full prompt + context length for attention matrix operations.",
    )

    bench_prefix = st.slider(
        "Shared Prefix Length (APC):",
        min_value=64,
        max_value=bench_seq - 64,
        value=min(256, bench_seq // 2),
        step=64,
        help="Number of prompt tokens cached and reused across requests.",
    )

    bench_window = st.select_slider(
        "Sliding Window Size (SWA):",
        options=[64, 128, 256, 512],
        value=min(128, bench_seq // 2),
        help="Local context window for bounded attention.",
    )

    bench_runs = st.slider("Repetitions per condition:", min_value=1, max_value=5, value=2)

    run_btn = st.button("Execute Live Benchmark on Apple M4 GPU", type="primary", use_container_width=True)

with bench_c2:
    if run_btn:
        with st.spinner("Executing tensor operations on Apple M4 Metal accelerator..."):
            bench_res = run_m4_attention_benchmark(
                seq_len=bench_seq,
                window_size=bench_window,
                prefix_len=bench_prefix,
                num_runs=bench_runs,
            )

        if bench_res.get("success"):
            st.markdown(
                f"""
                <div style="display: inline-flex; align-items: center; gap: 8px; padding: 6px 16px;
                            border-radius: 100px; border: 1px solid var(--accent-green);
                            font-family: var(--mono); font-size: 11px; color: var(--accent-green);
                            margin-bottom: 1rem; background: rgba(5,122,85,0.06);">
                    <span style="width: 7px; height: 7px; border-radius: 50%;
                                 background: var(--accent-green); display: inline-block;"></span>
                    HARDWARE ACCELERATION: {bench_res['device']}
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Bar Chart: Latency Comparison
            fig_bench = go.Figure()

            methods = ["Full Causal Attention", "Automatic Prefix Caching", "Sliding Window Attention"]
            latencies = [
                bench_res["full"]["latency_ms"],
                bench_res["apc"]["latency_ms"],
                bench_res["swa"]["latency_ms"],
            ]
            energies_uwh = [
                bench_res["full"]["energy_uwh"],
                bench_res["apc"]["energy_uwh"],
                bench_res["swa"]["energy_uwh"],
            ]

            fig_bench.add_trace(go.Bar(
                x=methods,
                y=latencies,
                marker_color=["#B91C1C", "#057A55", "#2563EB"],
                text=[f"{l:.2f} ms" for l in latencies],
                textposition="outside",
                textfont=dict(color="#000000", size=11, family="JetBrains Mono, monospace"),
            ))

            fig_bench.update_layout(
                title=dict(
                    text=f"Measured Apple M4 Execution Latency (ms) — Sequence Length {bench_seq}",
                    font=dict(family="Fraunces, Georgia, serif", size=16, color="#000000"),
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(247,245,240,0.5)",
                font=dict(family="Inter, sans-serif", color="#000000", size=12),
                xaxis=dict(
                    gridcolor="rgba(17,17,17,0.08)",
                    tickfont=dict(family="Inter, sans-serif", size=11, color="#000000"),
                ),
                yaxis=dict(
                    title=dict(text="Latency (milliseconds)", font=dict(color="#000000")),
                    gridcolor="rgba(17,17,17,0.08)",
                    tickfont=dict(family="JetBrains Mono, monospace", size=11, color="#000000"),
                ),
                margin=dict(l=60, r=30, t=50, b=50),
            )

            st.plotly_chart(fig_bench, use_container_width=True)

            # Live Metrics
            b_c1, b_c2, b_c3 = st.columns(3)
            with b_c1:
                st.markdown(
                    metric_card(
                        f"{bench_res['apc']['latency_speedup']}x",
                        "APC SPEEDUP ON M4",
                        f"Saved {bench_res['apc']['energy_saving_pct']}% tensor execution energy by reusing {bench_prefix} prefix tokens.",
                        color_class="green",
                    ),
                    unsafe_allow_html=True,
                )
            with b_c2:
                st.markdown(
                    metric_card(
                        f"{bench_res['swa']['latency_speedup']}x",
                        "SWA SPEEDUP ON M4",
                        f"Saved {bench_res['swa']['energy_saving_pct']}% energy by restricting attention to window w={bench_window}.",
                        color_class="green",
                    ),
                    unsafe_allow_html=True,
                )
            with b_c3:
                co2_saving = max(bench_res['full']['co2_mg'] - bench_res['apc']['co2_mg'], 0.0)
                st.markdown(
                    metric_card(
                        f"{co2_saving:.4f} mg",
                        "CO2e AVOIDED / PASS",
                        f"Per single forward pass on Apple M4 (at {GRID_CARBON_INTENSITY_KG_PER_KWH} kgCO2e/kWh).",
                        color_class="green",
                    ),
                    unsafe_allow_html=True,
                )

            st.markdown(
                finding(
                    f"Live hardware profiling on your <strong>{live_hw['chip']} GPU</strong> confirms that "
                    f"<strong>Automatic Prefix Caching achieves an empirical {bench_res['apc']['latency_speedup']}x speedup "
                    f"({bench_res['apc']['energy_saving_pct']}% energy abatement)</strong> on a {bench_seq}-token sequence "
                    f"by eliminating prompt prefill matrix multiplications. Sliding Window Attention achieved a "
                    f"<strong>{bench_res['swa']['latency_speedup']}x speedup ({bench_res['swa']['energy_saving_pct']}% abatement)</strong> "
                    f"by bounding attention to {bench_window} tokens.",
                    label="EMPIRICAL HARDWARE CONFIRMATION",
                ),
                unsafe_allow_html=True,
            )
        else:
            st.error(f"Benchmark error: {bench_res.get('error')}")
    else:
        st.markdown(
            """
            <div style="border: 1px dashed var(--divider); border-radius: 4px; padding: 2.5rem; text-align: center;
                        background: rgba(247,245,240,0.5);">
                <div style="font-family: var(--serif); font-size: 1.2rem; color: var(--ink); margin-bottom: 0.5rem;">
                    Ready for Live Hardware Profiling
                </div>
                <div style="font-family: var(--sans); font-size: 0.9rem; color: var(--graphite); max-width: 450px; margin: 0 auto;">
                    Click <strong>'Execute Live Benchmark on Apple M4 GPU'</strong> on the left to run real PyTorch
                    Metal Performance Shaders tensor operations directly on your Apple M4 chip and measure instantaneous latency,
                    energy expenditure, and carbon emissions.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  EXPERIMENTAL METHODOLOGY
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(divider(), unsafe_allow_html=True)

st.markdown(
    "<div class='chapter-marker'>EXPERIMENTAL PROTOCOL</div>"
    "<h3>Unified Measurement Framework</h3>",
    unsafe_allow_html=True,
)

# ── Metrics Table ─────────────────────────────────────────────────────────────
metrics_df = pd.DataFrame(
    {
        "Metric": [
            "Total Energy",
            "Energy per Request",
            "Estimated CO2e",
            "Average GPU/SoC Power",
            "Peak Memory",
            "Time to First Token (TTFT)",
            "Inter-Token Latency",
            "End-to-End Latency",
            "P95 / P99 Latency",
            "Throughput",
            "Output Quality Score",
            "APC Cache-Hit Rate",
        ],
        "Unit": [
            "Wh or J",
            "Wh/request",
            "gCO2e",
            "W",
            "GB",
            "ms",
            "ms/token",
            "ms",
            "ms",
            "tokens/s",
            "Task-specific",
            "%",
        ],
        "Purpose": [
            "Primary measure of electricity use for entire workload",
            "Normalizes energy by completed work",
            "Operational carbon emissions estimate",
            "Explains energy changes (not a substitute for energy)",
            "Shows memory requirements and KV-cache pressure",
            "Measures prefill and initial response delay",
            "Measures decoding responsiveness",
            "Measures full request completion time",
            "Captures tail latency performance for SLO compliance",
            "Measures serving capacity",
            "Checks for quality regression from optimization",
            "Explains APC effectiveness (APC experiments only)",
        ],
    }
)

st.dataframe(metrics_df, use_container_width=True, hide_index=True)

# ── Measurement Formulas ─────────────────────────────────────────────────────
st.markdown(
    formula_box(
        "Energy from Power Sampling",
        r"E_{\text{joules}} = \sum_{i=1}^{N} P_i \cdot \Delta t_i, \quad E_{\text{Wh}} = \frac{E_{\text{joules}}}{3600}",
        "Where P_i is power in watts during sample i and Delta_t_i is the sampling interval "
        "in seconds. Use consistent measurement boundaries; GPU-only energy should be "
        "labelled GPU-only and not presented as total system energy.",
        tag="ENERGY MEASUREMENT",
    ),
    unsafe_allow_html=True,
)

st.markdown(
    formula_box(
        "Carbon Dioxide Equivalent",
        r"\text{CO}_{2}\text{e (g)} = E_{\text{kWh}} \times CI_{\text{gCO}_{2}\text{e/kWh}}",
        "Where CI is the electricity carbon-intensity factor. If held constant between "
        "baseline and treatment, percentage CO2e reduction equals percentage energy reduction. "
        f"This project uses India CEA baseline: {GRID_CARBON_INTENSITY_KG_PER_KWH} kgCO2e/kWh.",
        tag="CARBON ACCOUNTING",
    ),
    unsafe_allow_html=True,
)

st.markdown(
    formula_box(
        "Energy per Output Token",
        r"\text{Energy per token} = \frac{E_{\text{total}}}{N_{\text{generated tokens}}}",
        "Report alongside energy per request. Neither metric replaces the other because "
        "request lengths and output quality can vary across treatment conditions.",
        tag="NORMALIZATION",
    ),
    unsafe_allow_html=True,
)


# ── Baseline vs Treatment Table ───────────────────────────────────────────────
st.markdown(
    "<div class='chapter-marker'>EXPERIMENTAL CONDITIONS</div>",
    unsafe_allow_html=True,
)

conditions_df = pd.DataFrame(
    {
        "Method": ["APC", "DVFS", "SWA"],
        "Baseline": [
            "Prefix caching disabled",
            "Default GPU/SoC power management",
            "Comparable full-attention model (Llama-2-7B-Q4)",
        ],
        "Treatment": [
            "Prefix caching enabled",
            "Selected operating points or observed autonomous scaling",
            "Compatible SWA model (Mistral-7B-Q4, window=4096)",
        ],
        "Local Platform Notes": [
            "Use llama.cpp --prompt-cache or mlx prompt caching",
            "Observe M4 autonomous DVFS via powermetrics; no direct clock control",
            "Both models quantized Q4_K_M; acknowledge architecture differences",
        ],
    }
)

st.dataframe(conditions_df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  RESEARCH LITERATURE
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(divider(), unsafe_allow_html=True)

st.markdown(
    "<div class='chapter-marker'>ACADEMIC FOUNDATIONS</div>"
    "<h3>Research References</h3>",
    unsafe_allow_html=True,
)

for idx, paper in enumerate(INFERENCE_OPT_PAPERS):
    st.markdown(
        f"""
        <div style="border-left: 2px solid var(--ink); padding: 1rem 0 1rem 1.5rem;
                    margin-bottom: 1.2rem; background: var(--smoke); border-radius: 0 2px 2px 0;">
            <div style="font-family: var(--mono); font-size: 11px; text-transform: uppercase;
                        letter-spacing: 0.05em; color: var(--accent-green); margin-bottom: 0.3rem;">
                REFERENCE {idx + 1} &mdash; {paper['key']}
            </div>
            <div style="font-family: var(--sans); font-weight: 600; font-size: 0.95rem; color: var(--ink);">
                <a href="{paper['url']}" target="_blank" style="color: var(--ink); text-decoration: underline;">
                    {paper['title']}
                </a>
            </div>
            <div style="font-family: var(--mono); font-size: 12px; color: var(--graphite);
                        margin: 0.3rem 0 0.5rem 0;">
                {paper['authors']}
            </div>
            <div style="font-family: var(--sans); font-size: 0.88rem; color: var(--graphite); line-height: 1.5;">
                <strong>Relevance:</strong> {paper['relevance']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  THREATS TO VALIDITY
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(divider(), unsafe_allow_html=True)

st.markdown(
    "<div class='chapter-marker'>THREATS TO VALIDITY</div>"
    "<h3>Reporting Precautions</h3>",
    unsafe_allow_html=True,
)

threat_c1, threat_c2 = st.columns(2)

with threat_c1:
    st.markdown(
        """
        1. **Energy boundary:** SoC energy is not the same as total system or data-centre energy.
        2. **Carbon intensity:** Grid intensity varies by location and time. State the source and period.
        3. **Thermal effects:** Chip temperature and thermal throttling affect power and latency.
        4. **Background processes:** Other workloads on the M4 can contaminate energy measurements.
        5. **Model differences:** A comparison between different model families cannot isolate
           SWA as the only cause of observed differences.
        """
    )

with threat_c2:
    st.markdown(
        """
        6. **Output lengths:** Different generated token counts make raw per-request comparisons
           misleading unless token counts are also reported.
        7. **Quality:** Lower energy is not useful if task quality falls unacceptably.
        8. **Repeated trials:** One run is insufficient. Report mean and standard deviation.
        9. **Combined methods:** Do not add individual percentage reductions. Measure the
           combined configuration because interactions may be non-additive.
        10. **Platform specificity:** Results on Apple M4 may not generalize to NVIDIA GPU servers.
        """
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  CONCLUSION
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(divider(), unsafe_allow_html=True)

st.markdown(
    quote(
        "Automatic Prefix Caching, SLO-aware GPU DVFS, and Sliding Window Attention address "
        "three distinct sources of energy use in LLM inference. APC reduces redundant prefill "
        "computation by reusing KV-cache state. DVFS adapts voltage and frequency to workload "
        "demands while meeting service-level objectives. SWA restricts attention to a local "
        "context window, reducing computation and KV-cache requirements for long sequences. "
        "Their environmental benefit must be established empirically by measuring total energy "
        "for a defined workload and converting that energy into CO2e using a documented "
        "carbon-intensity factor. The strongest contribution is a reproducible comparison "
        "reporting energy, carbon, latency, throughput, memory, and output quality.",
        attribution="Project Methodology Statement",
    ),
    unsafe_allow_html=True,
)
