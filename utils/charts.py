"""
Reusable Plotly chart functions for the AI Emissions Dashboard.

Every chart function returns a plotly.graph_objects.Figure with pure black
typography and editorial publication aesthetics.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from utils.constants import (
    MODEL_COLORS,
    QUANTIZATION_COLORS,
    PALETTE,
    MODEL_ORDER,
    QUANT_ORDER,
    BATCH_SIZE_COLORS,
    BATCH_SIZE_ORDER,
    INPUT_LENGTH_COLORS,
    INPUT_LENGTH_ORDER,
)


def _apply_layout(fig: go.Figure, title: str, xaxis: str = "", yaxis: str = "") -> go.Figure:
    """Apply consistent editorial publication layout with pure black typography."""
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(family="Fraunces, Georgia, serif", size=18, color="#000000"),
            x=0.01,
            xanchor="left",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(247,245,240,0.5)",
        font=dict(family="Inter, -apple-system, sans-serif", color="#000000", size=12),
        xaxis=dict(
            title=dict(text=xaxis, font=dict(family="Inter, sans-serif", size=12, color="#000000")),
            gridcolor="rgba(17,17,17,0.08)",
            zerolinecolor="rgba(17,17,17,0.25)",
            tickfont=dict(family="JetBrains Mono, monospace", size=11, color="#000000"),
        ),
        yaxis=dict(
            title=dict(text=yaxis, font=dict(family="Inter, sans-serif", size=12, color="#000000")),
            gridcolor="rgba(17,17,17,0.08)",
            zerolinecolor="rgba(17,17,17,0.25)",
            tickfont=dict(family="JetBrains Mono, monospace", size=11, color="#000000"),
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(17,17,17,0.15)",
            borderwidth=1,
            font=dict(family="Inter, sans-serif", size=11, color="#000000"),
        ),
        margin=dict(l=60, r=30, t=60, b=60),
        hoverlabel=dict(
            bgcolor="#000000",
            font_size=12,
            font_family="Inter, sans-serif",
            font_color="#FFFFFF",
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
    summary["energy_wh"] = summary["energy_kwh"] * 1000

    fig = px.bar(
        summary,
        x="model",
        y="energy_wh",
        color="model",
        color_discrete_map=MODEL_COLORS,
        text_auto=".3f",
    )
    fig.update_traces(
        textposition="outside",
        textfont=dict(color="#000000", size=11, family="JetBrains Mono, monospace"),
    )
    return _apply_layout(fig, "Average Energy per Response", "Model Architecture", "Energy (Wh)")


def bar_co2_by_model(df: pd.DataFrame) -> go.Figure:
    """Bar chart: average CO2 (grams) per response, by model."""
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
    fig.update_traces(
        textposition="outside",
        textfont=dict(color="#000000", size=11, family="JetBrains Mono, monospace"),
    )
    return _apply_layout(fig, "Average CO2e Emissions per Response", "Model Architecture", "CO2e (grams)")


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
    fig.update_traces(
        textposition="outside",
        textfont=dict(color="#000000", size=11, family="JetBrains Mono, monospace"),
    )
    return _apply_layout(fig, "Average Latency per Response", "Model Architecture", "Time (seconds)")


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
    fig.update_traces(
        textposition="top center",
        textfont=dict(color="#000000", size=12, family="Inter, sans-serif"),
    )
    return _apply_layout(
        fig,
        "Active Parameters vs Energy per Response",
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
    fig.update_traces(
        textposition="outside",
        textfont=dict(color="#000000", size=11, family="JetBrains Mono, monospace"),
    )
    return _apply_layout(
        fig,
        "Energy Consumption by Precision Regime (Llama-2-7B)",
        "Precision Format",
        "Energy (Wh)",
    )


def bar_quantization_co2(df: pd.DataFrame) -> go.Figure:
    """Bar chart: CO2 by quantization level."""
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
    fig.update_traces(
        textposition="outside",
        textfont=dict(color="#000000", size=11, family="JetBrains Mono, monospace"),
    )
    return _apply_layout(
        fig,
        "CO2e Emissions by Precision Regime (Llama-2-7B)",
        "Precision Format",
        "CO2e (grams)",
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
        textfont=dict(color="#000000", size=12, family="JetBrains Mono, monospace"),
        marker=dict(size=14, color=[QUANTIZATION_COLORS[p] for p in summary["precision"]]),
        line=dict(color="#2563EB", width=3),
        name="Quality Score",
    ))
    fig = _apply_layout(
        fig,
        "Semantic Quality Retention vs Precision (Llama-2-7B)",
        "Precision Format",
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
    return _apply_layout(fig, "Energy Distribution Across Queries", "Energy (Wh)", "Frequency Count")


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
    return _apply_layout(fig, "Energy Spread per Architecture", "Model Architecture", "Energy (Wh)")


def box_co2_by_model(df: pd.DataFrame) -> go.Figure:
    """Box plot: CO2 spread per model."""
    fig = px.box(
        df,
        x="model",
        y="co2_grams",
        color="model",
        color_discrete_map=MODEL_COLORS,
        category_orders={"model": MODEL_ORDER},
    )
    return _apply_layout(fig, "CO2e Spread per Architecture", "Model Architecture", "CO2e (grams)")


def bar_batch_size_energy(df: pd.DataFrame) -> go.Figure:
    """Bar chart: energy per query at different batch sizes."""
    summary = df.groupby("batch_size")["energy_kwh"].mean().reindex(BATCH_SIZE_ORDER).reset_index()
    summary["energy_wh"] = summary["energy_kwh"] * 1000
    fig = px.bar(
        summary,
        x="batch_size",
        y="energy_wh",
        color="batch_size",
        color_discrete_map={str(k): v for k, v in BATCH_SIZE_COLORS.items()},
        text_auto=".3f",
    )
    fig.update_traces(
        textposition="outside",
        textfont=dict(color="#000000", size=11, family="JetBrains Mono, monospace"),
    )
    fig.update_xaxes(type="category")
    return _apply_layout(fig, "Normalized Energy per Query by Batch Size (Llama-2-7B)", "Batch Size (N)", "Energy (Wh)")


def bar_batch_size_latency(df: pd.DataFrame) -> go.Figure:
    """Bar chart: latency per query at different batch sizes."""
    summary = df.groupby("batch_size")["duration_s"].mean().reindex(BATCH_SIZE_ORDER).reset_index()
    fig = px.bar(
        summary,
        x="batch_size",
        y="duration_s",
        color="batch_size",
        color_discrete_map={str(k): v for k, v in BATCH_SIZE_COLORS.items()},
        text_auto=".2f",
    )
    fig.update_traces(
        textposition="outside",
        textfont=dict(color="#000000", size=11, family="JetBrains Mono, monospace"),
    )
    fig.update_xaxes(type="category")
    return _apply_layout(fig, "Latency per Forward Pass by Batch Size (Llama-2-7B)", "Batch Size (N)", "Time (seconds)")


def scatter_input_length_energy(df: pd.DataFrame) -> go.Figure:
    """Scatter: input tokens vs energy consumption."""
    df = df.copy()
    df["energy_wh"] = df["energy_kwh"] * 1000
    fig = px.scatter(
        df,
        x="input_tokens",
        y="energy_wh",
        color="input_length_category",
        color_discrete_map=INPUT_LENGTH_COLORS,
        size="energy_wh",
        size_max=20,
        trendline="ols",
        category_orders={"input_length_category": INPUT_LENGTH_ORDER},
    )
    fig.update_traces(
        textfont=dict(color="#000000", size=11, family="Inter, sans-serif"),
    )
    return _apply_layout(fig, "Input Prompt Length vs Total Energy (Llama-2-7B)", "Input Tokens (Prompt)", "Energy (Wh)")


def bar_energy_per_input_token(df: pd.DataFrame) -> go.Figure:
    """Bar chart: energy per input token by input length category."""
    summary = df.groupby("input_length_category").agg(
        energy_kwh=("energy_kwh", "mean"),
        input_tokens=("input_tokens", "mean"),
    ).reindex(INPUT_LENGTH_ORDER).reset_index()
    summary["energy_per_input_token_wh"] = (summary["energy_kwh"] / summary["input_tokens"]) * 1000
    fig = px.bar(
        summary,
        x="input_length_category",
        y="energy_per_input_token_wh",
        color="input_length_category",
        color_discrete_map=INPUT_LENGTH_COLORS,
        text_auto=".4f",
    )
    fig.update_traces(
        textposition="outside",
        textfont=dict(color="#000000", size=11, family="JetBrains Mono, monospace"),
    )
    return _apply_layout(fig, "Energy per Input Token by Length Category", "Context Window Regime", "Energy per Token (Wh)")


def bar_energy_per_token(df: pd.DataFrame) -> go.Figure:
    """Bar chart: energy per output token by model."""
    summary = df.groupby("model").apply(
        lambda g: pd.Series({
            "energy_per_token_wh": (g["energy_kwh"].sum() / g["tokens_generated"].sum()) * 1000
        })
    ).reindex(MODEL_ORDER).reset_index()
    fig = px.bar(
        summary,
        x="model",
        y="energy_per_token_wh",
        color="model",
        color_discrete_map=MODEL_COLORS,
        text_auto=".5f",
    )
    fig.update_traces(
        textposition="outside",
        textfont=dict(color="#000000", size=11, family="JetBrains Mono, monospace"),
    )
    return _apply_layout(fig, "Energy per Output Token by Architecture", "Model Architecture", "Energy per Token (Wh)")
