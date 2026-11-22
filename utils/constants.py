"""
Constants for the AI Inference Emissions Dashboard.

Conversion factors, color palettes, and reference values used
throughout the dashboard for consistent styling and calculations.
"""

# ---------------------------------------------------------------------------
# Color palette — Editorial Journal Palette
# ---------------------------------------------------------------------------
MODEL_COLORS = {
    "TinyLlama-1.1B": "#057A55",   # editorial forest green
    "Phi-2-2.7B":     "#2563EB",   # technical cobalt blue
    "Llama-2-7B (Real)": "#D97706", # editorial amber
    "Llama-2-13B":    "#B91C1C",   # editorial crimson
}

QUANTIZATION_COLORS = {
    "FP16":  "#B91C1C",   # crimson — full precision
    "INT8":  "#D97706",   # amber — 8-bit quantized
    "INT4":  "#057A55",   # forest green — 4-bit compressed
}

# General chart palette
PALETTE = ["#057A55", "#2563EB", "#D97706", "#B91C1C", "#4B5563", "#111827"]

# Batch size experiment colors
BATCH_SIZE_COLORS = {
    1: "#B91C1C",   # single query
    2: "#D97706",   # small batch
    4: "#057A55",   # optimized batch
}
BATCH_SIZE_ORDER = [1, 2, 4]

# Input length categories
INPUT_LENGTH_COLORS = {
    "Short": "#057A55",   # forest green
    "Medium": "#D97706",  # amber
    "Long": "#B91C1C",    # crimson
}
INPUT_LENGTH_ORDER = ["Short", "Medium", "Long"]

# ---------------------------------------------------------------------------
# CO₂ conversion factors
# ---------------------------------------------------------------------------
# India average grid carbon intensity (kg CO₂ per kWh)
# Source: IEA 2023, Central Electricity Authority of India
GRID_CARBON_INTENSITY_KG_PER_KWH = 0.708

# ---------------------------------------------------------------------------
# Real-world equivalencies (for the impact calculator)
# Sources: EPA Greenhouse Gas Equivalencies Calculator
# ---------------------------------------------------------------------------
# kg CO₂ per mile driven by an average passenger vehicle
CO2_KG_PER_CAR_MILE = 0.404

# kWh to fully charge a smartphone (~0.012 kWh)
KWH_PER_SMARTPHONE_CHARGE = 0.012

# kg CO₂ absorbed by one tree per year
CO2_KG_PER_TREE_PER_YEAR = 21.77

# kWh consumed by an average US home per day
KWH_PER_HOME_PER_DAY = 30.0

# kg CO₂ from one Google search (estimated)
CO2_GRAMS_PER_GOOGLE_SEARCH = 0.2

# ---------------------------------------------------------------------------
# Model metadata
# ---------------------------------------------------------------------------
MODEL_PARAMS = {
    "TinyLlama-1.1B": 1.1,
    "Phi-2-2.7B":     2.7,
    "Llama-2-7B (Real)":     7.2,
    "Llama-2-13B":    13.0,
}

# Ordered list of models (smallest → largest)
MODEL_ORDER = ["TinyLlama-1.1B", "Phi-2-2.7B", "Llama-2-7B (Real)", "Llama-2-13B"]
QUANT_ORDER = ["FP16", "INT8", "INT4"]

# ---------------------------------------------------------------------------
# Sample prompts used in the experiment
# ---------------------------------------------------------------------------
SAMPLE_PROMPTS = [
    "What is photosynthesis?",
    "Explain Newton's third law of motion.",
    "Summarize the plot of Romeo and Juliet.",
    "What causes earthquakes?",
    "Write a short poem about the ocean.",
    "Explain how vaccines work in simple terms.",
    "What is the difference between weather and climate?",
    "Describe the water cycle.",
    "Who was Ada Lovelace and why is she important?",
    "What are the main causes of air pollution?",
    "Explain the greenhouse effect.",
    "What is machine learning in one paragraph?",
    "Describe how a battery works.",
    "What are renewable energy sources?",
    "Explain why the sky is blue.",
    "What is the theory of evolution?",
    "Describe the structure of DNA.",
    "How does the internet work?",
    "What is inflation in economics?",
    "Explain the concept of supply and demand.",
]

# ---------------------------------------------------------------------------
# Research paper references
# ---------------------------------------------------------------------------
PAPERS = [
    {
        "key": "Strubell et al. (ACL 2019)",
        "title": "Energy and Policy Considerations for Deep Learning in NLP",
        "authors": "E. Strubell, A. Ganesh, A. McCallum",
        "url": "https://arxiv.org/abs/1906.02243",
        "relevance": "Pioneered quantifying energy costs of training large NLP models. "
                     "Our measurement methodology follows their approach.",
    },
    {
        "key": "Lacoste et al. (2019)",
        "title": "Quantifying the Carbon Emissions of Machine Learning",
        "authors": "A. Lacoste, A. Luccioni, V. Schmidt, T. Dandres",
        "url": "https://arxiv.org/abs/1910.09700",
        "relevance": "Created the ML CO₂ Impact calculator; established the formula "
                     "we use: CO₂ = Energy × Grid Carbon Intensity.",
    },
    {
        "key": "Henderson et al. (JMLR 2020)",
        "title": "Towards the Systematic Reporting of the Energy and Carbon "
                 "Footprints of Machine Learning",
        "authors": "P. Henderson, J. Hu, J. Romoff, et al.",
        "url": "https://arxiv.org/abs/2002.05651",
        "relevance": "Proposed a standardized reporting framework for ML energy use. "
                     "We adopt their reporting recommendations.",
    },
    {
        "key": "Luccioni et al. (2022)",
        "title": "Estimating the Carbon Footprint of BLOOM, a 176B Parameter "
                 "Language Model",
        "authors": "A.S. Luccioni, S. Viguier, A.-L. Ligozat",
        "url": "https://arxiv.org/abs/2211.02001",
        "relevance": "Used CodeCarbon to measure BLOOM's full lifecycle emissions — "
                     "validated our tool choice for inference measurement.",
    },
]

# ---------------------------------------------------------------------------
# JEPA research paper references
# ---------------------------------------------------------------------------
JEPA_PAPERS = [
    {
        "key": "Assran et al. (CVPR 2023)",
        "title": "Self-Supervised Learning from Images with a Joint-Embedding "
                 "Predictive Architecture (I-JEPA)",
        "authors": "M. Assran, Q. Duval, I. Misra, P. Bojanowski, P. Vincent, "
                   "M. Rabbat, Y. LeCun, N. Ballas",
        "url": "https://openaccess.thecvf.com/content/CVPR2023/html/Assran_Self-Supervised_Learning_From_Images_With_a_Joint-Embedding_Predictive_Architecture_CVPR_2023_paper.html",
        "relevance": "Introduces I-JEPA for image representation learning via latent "
                     "prediction rather than pixel reconstruction. Reports training on "
                     "16 A100 GPUs with ViT-H/14. Establishes technical feasibility of "
                     "the JEPA learning objective for visual representations.",
    },
    {
        "key": "Patterson et al. (2021)",
        "title": "Carbon Emissions and Large Neural Network Training",
        "authors": "D. Patterson, J. Gonzalez, Q. Le, C. Liang, L.-M. Munguia, "
                   "D. Rothchild, D. So, M. Texier, J. Dean",
        "url": "https://arxiv.org/abs/2104.10350",
        "relevance": "Examines energy use and carbon emissions of training large "
                     "neural networks. Establishes the importance of hardware, "
                     "data-centre efficiency, and electricity sources in determining "
                     "the carbon footprint of AI systems.",
    },
    {
        "key": "LeCun (2022)",
        "title": "A Path Towards Autonomous Machine Intelligence",
        "authors": "Y. LeCun",
        "url": "https://openreview.net/forum?id=BZ5a1r-kVsf",
        "relevance": "Proposes the conceptual foundation for JEPA: hierarchical "
                     "predictive world models that predict abstract representations "
                     "rather than reconstructing all sensory details. Position paper "
                     "providing theoretical motivation for the JEPA direction.",
    },
    {
        "key": "Bardes et al. (2024)",
        "title": "Revisiting Feature Prediction for Learning Visual "
                 "Representations from Video (V-JEPA)",
        "authors": "A. Bardes, Q. Garrido, J. Ponce, X. Chen, M. Rabbat, "
                   "Y. LeCun, M. Assran, N. Ballas",
        "url": "https://arxiv.org/abs/2404.08471",
        "relevance": "Extends JEPA to video representation learning (V-JEPA). "
                     "Video contains large amounts of temporally redundant "
                     "information, making it the strongest candidate for a JEPA "
                     "front-end that compresses multimodal input before an LLM.",
    },
]

# ---------------------------------------------------------------------------
# Inference Optimization Methods — research paper references
# ---------------------------------------------------------------------------
INFERENCE_OPT_PAPERS = [
    {
        "key": "vLLM — Automatic Prefix Caching",
        "title": "Automatic Prefix Caching (vLLM Documentation)",
        "authors": "vLLM Project Contributors",
        "url": "https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/",
        "relevance": "Documents the runtime-level KV-cache reuse mechanism for matching "
                     "token prefixes. Provides implementation details, block-hashing "
                     "strategy, and cache-eviction behaviour.",
    },
    {
        "key": "vLLM — Prefix Caching Design",
        "title": "Automatic Prefix Caching Design (vLLM Documentation)",
        "authors": "vLLM Project Contributors",
        "url": "https://docs.vllm.ai/en/stable/design/prefix_caching/",
        "relevance": "Describes the internal architecture of block-level prefix caching, "
                     "hash-based block identification, and memory management trade-offs.",
    },
    {
        "key": "IEEE CAL 2024 — SLO-Aware GPU DVFS",
        "title": "SLO-Aware GPU DVFS for Energy-Efficient LLM Inference Serving",
        "authors": "IEEE Computer Architecture Letters, Vol. 23, No. 2, pp. 150-153",
        "url": "https://doi.org/10.1109/LCA.2024.3406038",
        "relevance": "Reports 22.8% to 45.5% energy reduction under SLO constraints by "
                     "dynamically adjusting GPU voltage and frequency during inference. "
                     "Results are specific to their evaluated hardware and workloads.",
    },
    {
        "key": "Jiang et al. (2023) — Mistral 7B",
        "title": "Mistral 7B",
        "authors": "A.Q. Jiang, A. Sablayrolles, A. Mensch, C. Bamford, D.S. Chaplot, "
                   "D. de las Casas, et al.",
        "url": "https://arxiv.org/abs/2310.06825",
        "relevance": "Introduces Mistral 7B architecture using Sliding Window Attention "
                     "(window size 4,096) alongside Grouped-Query Attention. Demonstrates "
                     "that bounded local attention can maintain quality while reducing "
                     "memory and computation for long sequences.",
    },
]

# ---------------------------------------------------------------------------
# Hardware specification for local experimental platform
# ---------------------------------------------------------------------------
LOCAL_HARDWARE = {
    "chip": "Apple M4",
    "cores_total": 10,
    "cores_performance": 4,
    "cores_efficiency": 6,
    "gpu_cores": 10,
    "memory_gb": 16,
    "metal_support": "Metal 4",
    "model": "MacBook Air (M4, 2025)",
    "display": "2880 x 1864 Liquid Retina",
    "neural_engine_tops": 38,
    "memory_bandwidth_gbps": 120,
    "gpu_type": "Integrated Apple M4 GPU (10-core)",
    "notes": "Unified memory architecture — CPU, GPU, and Neural Engine share "
             "the same 16 GB LPDDR5X memory pool. No discrete NVIDIA GPU available; "
             "NVML power measurement is not applicable. Use powermetrics or "
             "Apple Instruments for energy profiling.",
}
