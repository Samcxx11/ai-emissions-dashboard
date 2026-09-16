"""
Constants for the AI Inference Emissions Dashboard.

Conversion factors, color palettes, and reference values used
throughout the dashboard for consistent styling and calculations.
"""

# ---------------------------------------------------------------------------
# Color palette — accessible, visually distinct
# ---------------------------------------------------------------------------
MODEL_COLORS = {
    "TinyLlama-1.1B": "#2ecc71",   # green  — smallest, most efficient
    "Phi-2-2.7B":     "#3498db",   # blue
    "Llama-2-7B (Real)":     "#f39c12",   # orange
    "Llama-2-13B":    "#e74c3c",   # red    — largest, most energy
}

QUANTIZATION_COLORS = {
    "FP16":  "#e74c3c",   # red   — full precision
    "INT8":  "#f39c12",   # amber — moderate compression
    "INT4":  "#2ecc71",   # green — heavy compression
}

# General chart palette
PALETTE = ["#2ecc71", "#3498db", "#f39c12", "#e74c3c", "#9b59b6", "#1abc9c"]

# Batch size experiment colors
BATCH_SIZE_COLORS = {
    1: "#e74c3c",   # red — single query
    2: "#f39c12",   # amber — small batch
    4: "#2ecc71",   # green — larger batch
}
BATCH_SIZE_ORDER = [1, 2, 4]

# Input length categories
INPUT_LENGTH_COLORS = {
    "Short": "#2ecc71",   # green — least energy per token
    "Medium": "#f39c12",  # amber
    "Long": "#e74c3c",    # red — most total energy
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
