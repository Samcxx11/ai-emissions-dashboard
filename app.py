import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from utils.theme import EDITORIAL_CSS, chapter, divider, finding, quote, metric_card, big_number

st.set_page_config(page_title="AI Inference Emissions", page_icon="🌍", layout="wide")
st.markdown(EDITORIAL_CSS, unsafe_allow_html=True)

st.markdown('<div class="editorial-hero">', unsafe_allow_html=True)
st.markdown('<div class="status-badge"><div class="status-dot"></div>v1.0.0 — LIVE RESEARCH</div>', unsafe_allow_html=True)
st.markdown('<h1><em>AI Inference</em> Emissions</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Quantifying the invisible carbon footprint of large language models during production inference.</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown(chapter(1, "ABSTRACT"), unsafe_allow_html=True)
st.markdown(
    """
    While the massive energy costs of *training* AI models are well-documented, 
    the environmental impact of *inference* — answering daily queries at scale — 
    remains largely invisible.
    
    This dashboard visualizes real benchmarking data from CodeCarbon, exploring how 
    model size, quantization, and batching affect energy consumption and CO₂ emissions.
    """
)

st.markdown(divider(), unsafe_allow_html=True)

st.markdown(chapter(2, "METHODOLOGY"), unsafe_allow_html=True)
st.markdown(
    """
    Our methodology traces the real hardware energy draw using `CodeCarbon`. We benchmarked 
    models ranging from 1.1B to 13B parameters on standardized input lengths.
    """
)
st.markdown(
    quote("We cannot optimize what we do not measure. This project brings transparency to the inference lifecycle.", "The Project Team"),
    unsafe_allow_html=True
)

st.markdown(divider(), unsafe_allow_html=True)
st.markdown(chapter(3, "QUICK NAVIGATION"), unsafe_allow_html=True)
st.markdown(
    """
    Use the sidebar to explore specific experiments:
    * **Energy & CO₂ Comparison:** The core difference between 1B and 13B models.
    * **Quantization Impact:** How INT4 compression saves memory but increases energy.
    * **Real-World Impact:** Translating kWh to miles driven.
    * **Batch Size Analysis:** How parallel processing saves per-query energy.
    * **JEPA vs Generative:** The future of energy-efficient AI.
    """
)
