"""
Reusable Plotly chart functions for the AI Emissions Dashboard.

Every chart function returns a plotly.graph_objects.Figure so the caller
can simply do `st.plotly_chart(fig, use_container_width=True)`.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from utils.constants import MODEL_COLORS, QUANTIZATION_COLORS, PALETTE, MODEL_ORDER, QUANT_ORDER, BATCH_SIZE_COLORS, BATCH_SIZE_ORDER, INPUT_LENGTH_COLORS, INPUT_LENGTH_ORDER, ARCHITECTURE_COLORS, ARCHITECTURE_ORDER, JEPA_MODEL_COLORS


def _apply_layout(fig: go.Figure, title: str, xaxis: str = "", yaxis: str = "") -> go.Figure:
    """Apply consistent dark-themed layout to a figure."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color="#ecf0f1")),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#1a1d23",
        font=dict(color="#bdc3c7", size=13),
        xaxis=dict(
            title=xaxis,
            gridcolor="#2c3e50",
            zerolinecolor="#2c3e50",
        ),
        yaxis=dict(
            title=yaxis,
            gridcolor="#2c3e50",
            zerolinecolor="#2c3e50",
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ecf0f1"),
        ),
        margin=dict(l=60, r=30, t=60, b=60),
        hoverlabel=dict(
            bgcolor="#2c3e50",
            font_size=13,
            font_color="#ecf0f1",
        ),
    )
    return fig


# ── Bar charts ────────────────────────────────────────────────────────────────

def bar_energy_by_model(df: pd.DataFrame) -> go.Figure:
    """Bar chart: average energy (Wh) per response, by model."""
    summary = (
        df.groupby("model")["energy_kwh"]
        .mean()
        .reindex(MODEL_ORDER)
        .reset_index()
    )
    summary["energy_wh"] = summary["energy_kwh"] * 1000  # convert to Wh for readability

    fig = px.bar(
        summary,
        x="model",
        y="energy_wh",
        color="model",
        color_discrete_map=MODEL_COLORS,
        text_auto=".3f",
    )
    fig.update_traces(textposition="outside")
    return _apply_layout(fig, "Average Energy per Response", "Model", "Energy (Wh)")


def bar_co2_by_model(df: pd.DataFrame) -> go.Figure:
    """Bar chart: average CO₂ (grams) per response, by model."""
    summary = (
        df.groupby("model")["co2_grams"]
        .mean()
        .reindex(MODEL_ORDER)
        .reset_index()
    )

    fig = px.bar(
        summary,
        x="model",
        y="co2_grams",
        color="model",
        color_discrete_map=MODEL_COLORS,
        text_auto=".3f",
    )
    fig.update_traces(textposition="outside")
    return _apply_layout(fig, "Average CO₂ per Response", "Model", "CO₂ (grams)")


def bar_latency_by_model(df: pd.DataFrame) -> go.Figure:
    """Bar chart: average latency (seconds) per response, by model."""
    summary = (
        df.groupby("model")["duration_s"]
        .mean()
        .reindex(MODEL_ORDER)
        .reset_index()
    )

    fig = px.bar(
        summary,
        x="model",
        y="duration_s",
        color="model",
        color_discrete_map=MODEL_COLORS,
        text_auto=".2f",
    )
    fig.update_traces(textposition="outside")
    return _apply_layout(fig, "Average Latency per Response", "Model", "Time (seconds)")


# ── Scatter plots ─────────────────────────────────────────────────────────────

def scatter_params_vs_energy(df: pd.DataFrame, params_map: dict) -> go.Figure:
    """Scatter: model parameters (B) vs average energy per response."""
    summary = df.groupby("model")["energy_kwh"].mean().reset_index()
    summary["params_b"] = summary["model"].map(params_map)
    summary["energy_wh"] = summary["energy_kwh"] * 1000

    fig = px.scatter(
        summary,
        x="params_b",
        y="energy_wh",
        color="model",
        color_discrete_map=MODEL_COLORS,
        size="energy_wh",
        size_max=30,
        text="model",
    )
    fig.update_traces(textposition="top center", textfont_size=12)
    return _apply_layout(
        fig,
        "Model Size vs Energy per Response",
        "Parameters (Billions)",
        "Energy (Wh)",
    )


# ── Quantization charts ──────────────────────────────────────────────────────

def bar_quantization_energy(df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart: energy by quantization level."""
    summary = (
        df.groupby("precision")["energy_kwh"]
        .mean()
        .reindex(QUANT_ORDER)
        .reset_index()
    )
    summary["energy_wh"] = summary["energy_kwh"] * 1000

    fig = px.bar(
        summary,
        x="precision",
        y="energy_wh",
        color="precision",
        color_discrete_map=QUANTIZATION_COLORS,
        text_auto=".3f",
    )
    fig.update_traces(textposition="outside")
    return _apply_layout(
        fig,
        "Llama-2-7B (Real): Energy by Quantization Level",
        "Precision",
        "Energy (Wh)",
    )


def bar_quantization_co2(df: pd.DataFrame) -> go.Figure:
    """Bar chart: CO₂ by quantization level."""
    summary = (
        df.groupby("precision")["co2_grams"]
        .mean()
        .reindex(QUANT_ORDER)
        .reset_index()
    )

    fig = px.bar(
        summary,
        x="precision",
        y="co2_grams",
        color="precision",
        color_discrete_map=QUANTIZATION_COLORS,
        text_auto=".3f",
    )
    fig.update_traces(textposition="outside")
    return _apply_layout(
        fig,
        "Llama-2-7B (Real): CO₂ by Quantization Level",
        "Precision",
        "CO₂ (grams)",
    )


def line_quality_vs_precision(df: pd.DataFrame) -> go.Figure:
    """Line chart: quality score vs precision level (the trade-off)."""
    summary = (
        df.groupby("precision")["quality_score"]
        .mean()
        .reindex(QUANT_ORDER)
        .reset_index()
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=summary["precision"],
        y=summary["quality_score"],
        mode="lines+markers+text",
        text=[f"{v:.1f}" for v in summary["quality_score"]],
        textposition="top center",
        marker=dict(size=14, color=[QUANTIZATION_COLORS[p] for p in summary["precision"]]),
        line=dict(color="#3498db", width=3),
        name="Quality Score",
    ))
    fig = _apply_layout(
        fig,
        "Quality vs Precision Trade-off (Llama-2-7B (Real))",
        "Precision",
        "Quality Score (0–100)",
    )
    fig.update_yaxes(range=[70, 102])
    return fig


# ── Distribution / per-query charts ──────────────────────────────────────────

def histogram_energy(df: pd.DataFrame) -> go.Figure:
    """Histogram of energy consumption across all queries."""
    df = df.copy()
    df["energy_wh"] = df["energy_kwh"] * 1000

    fig = px.histogram(
        df,
        x="energy_wh",
        color="model",
        color_discrete_map=MODEL_COLORS,
        nbins=40,
        barmode="overlay",
        opacity=0.7,
    )
    return _apply_layout(fig, "Energy Distribution Across Queries", "Energy (Wh)", "Count")


def box_energy_by_model(df: pd.DataFrame) -> go.Figure:
    """Box plot: energy spread per model."""
    df = df.copy()
    df["energy_wh"] = df["energy_kwh"] * 1000

    fig = px.box(
        df,
        x="model",
        y="energy_wh",
        color="model",
        color_discrete_map=MODEL_COLORS,
        category_orders={"model": MODEL_ORDER},
    )
    return _apply_layout(fig, "Energy Spread per Model", "Model", "Energy (Wh)")


def box_co2_by_model(df: pd.DataFrame) -> go.Figure:
    """Box plot: CO₂ spread per model."""
    fig = px.box(
        df,
        x="model",
        y="co2_grams",
        color="model",
        color_discrete_map=MODEL_COLORS,
        category_orders={"model": MODEL_ORDER},
    )
    return _apply_layout(fig, "CO₂ Spread per Model", "Model", "CO₂ (grams)")


def bar_batch_size_energy(df: pd.DataFrame) -> go.Figure:
    """Bar chart: energy per query at different batch sizes."""
    summary = df.groupby("batch_size")["energy_kwh"].mean().reindex(BATCH_SIZE_ORDER).reset_index()
    summary["energy_wh"] = summary["energy_kwh"] * 1000
    fig = px.bar(summary, x="batch_size", y="energy_wh",
                 color="batch_size", color_discrete_map={str(k): v for k, v in BATCH_SIZE_COLORS.items()},
                 text_auto=".3f")
    fig.update_traces(textposition="outside")
    fig.update_xaxes(type="category")
    return _apply_layout(fig, "Energy per Query by Batch Size (Llama-2-7B (Real))", "Batch Size", "Energy (Wh)")


def bar_batch_size_latency(df: pd.DataFrame) -> go.Figure:
    """Bar chart: latency per query at different batch sizes."""
    summary = df.groupby("batch_size")["duration_s"].mean().reindex(BATCH_SIZE_ORDER).reset_index()
    fig = px.bar(summary, x="batch_size", y="duration_s",
                 color="batch_size", color_discrete_map={str(k): v for k, v in BATCH_SIZE_COLORS.items()},
                 text_auto=".2f")
    fig.update_traces(textposition="outside")
    fig.update_xaxes(type="category")
    return _apply_layout(fig, "Latency per Query by Batch Size (Llama-2-7B (Real))", "Batch Size", "Time (seconds)")


def scatter_input_length_energy(df: pd.DataFrame) -> go.Figure:
    """Scatter: input tokens vs energy consumption."""
    df = df.copy()
    df["energy_wh"] = df["energy_kwh"] * 1000
    fig = px.scatter(df, x="input_tokens", y="energy_wh",
                     color="input_length_category", color_discrete_map=INPUT_LENGTH_COLORS,
                     size="energy_wh", size_max=20,
                     trendline="ols",
                     category_orders={"input_length_category": INPUT_LENGTH_ORDER})
    return _apply_layout(fig, "Input Length vs Energy (Llama-2-7B (Real))", "Input Tokens", "Energy (Wh)")


def bar_energy_per_input_token(df: pd.DataFrame) -> go.Figure:
    """Bar chart: energy per input token by input length category."""
    summary = df.groupby("input_length_category").agg(
        energy_kwh=("energy_kwh", "mean"),
        input_tokens=("input_tokens", "mean")
    ).reindex(INPUT_LENGTH_ORDER).reset_index()
    summary["energy_per_input_token_wh"] = (summary["energy_kwh"] / summary["input_tokens"]) * 1000
    fig = px.bar(summary, x="input_length_category", y="energy_per_input_token_wh",
                 color="input_length_category", color_discrete_map=INPUT_LENGTH_COLORS,
                 text_auto=".4f")
    fig.update_traces(textposition="outside")
    return _apply_layout(fig, "Energy per Input Token by Length Category", "Input Length", "Energy per Token (Wh)")


def bar_energy_per_token(df: pd.DataFrame) -> go.Figure:
    """Bar chart: energy per output token by model."""
    summary = df.groupby("model").apply(
        lambda g: pd.Series({
            "energy_per_token_wh": (g["energy_kwh"].sum() / g["tokens_generated"].sum()) * 1000
        })
    ).reindex(MODEL_ORDER).reset_index()
    fig = px.bar(summary, x="model", y="energy_per_token_wh",
                 color="model", color_discrete_map=MODEL_COLORS,
                 text_auto=".5f")
    fig.update_traces(textposition="outside")
    return _apply_layout(fig, "Energy per Output Token", "Model", "Energy per Token (Wh)")


# ── JEPA Charts ───────────────────────────────────────────────────────────────

def bar_jepa_energy_comparison(df: pd.DataFrame) -> go.Figure:
    """Bar chart: avg energy (Joules) per model — JEPA vs Generative."""
    summary = df.groupby("model")["energy_j"].mean().reset_index()
    summary = summary.sort_values("energy_j", ascending=False)
    fig = px.bar(summary, x="model", y="energy_j",
                 color="model", color_discrete_map=JEPA_MODEL_COLORS,
                 text_auto=".0f")
    fig.update_traces(textposition="outside")
    return _apply_layout(fig, "Energy per Inference: JEPA vs Generative", "Model", "Energy (Joules)")


def line_jepa_scaling(df: pd.DataFrame) -> go.Figure:
    """Line chart: how energy scales with model size for both architectures."""
    summary = df.groupby(["parameters_b", "architecture"])["energy_j"].mean().reset_index()
    fig = px.line(summary, x="parameters_b", y="energy_j",
                  color="architecture", color_discrete_map=ARCHITECTURE_COLORS,
                  markers=True,
                  category_orders={"architecture": ARCHITECTURE_ORDER})
    fig.update_traces(line=dict(width=3), marker=dict(size=10))
    return _apply_layout(fig, "Energy Scaling: JEPA vs Generative (as model grows)", "Parameters (Billions)", "Energy (Joules)")


def bar_jepa_co2_comparison(df: pd.DataFrame) -> go.Figure:
    """Bar chart: CO2 by architecture type."""
    summary = df.groupby("architecture_type")["co2_grams"].mean().reset_index()
    fig = px.bar(summary, x="architecture_type", y="co2_grams",
                 color="architecture_type", color_discrete_map=ARCHITECTURE_COLORS,
                 text_auto=".4f")
    fig.update_traces(textposition="outside")
    return _apply_layout(fig, "Avg CO₂ Emissions: Predictive vs Generative", "Architecture", "CO₂ (grams)")
