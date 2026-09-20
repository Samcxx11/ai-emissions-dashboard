import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.theme import EDITORIAL_CSS, apply_editorial_layout, chapter, divider, finding, quote, metric_card, big_number

st.set_page_config(page_title="Real-World Impact", page_icon="🌍", layout="wide")
st.markdown(EDITORIAL_CSS, unsafe_allow_html=True)

from utils.constants import (
    MODEL_COLORS,
    MODEL_ORDER,
    CO2_KG_PER_CAR_MILE,
    KWH_PER_SMARTPHONE_CHARGE,
    CO2_KG_PER_TREE_PER_YEAR,
    KWH_PER_HOME_PER_DAY,
    GRID_CARBON_INTENSITY_KG_PER_KWH,
)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    return pd.read_csv(os.path.join(data_dir, "inference_results.csv"))

df = load_data()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(chapter(1, "REAL-WORLD IMPACT"), unsafe_allow_html=True)
st.markdown("What do our measurements mean at real-world scale? This page translates per-query numbers into tangible, understandable impacts.")
st.markdown(divider(), unsafe_allow_html=True)
