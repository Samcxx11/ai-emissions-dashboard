"""
Editorial Theme System — 'Paper and Ink' Design Language
Inspired by high-end technical journals and editorial publications.
"""

EDITORIAL_CSS = """
<style>
    /* ── Google Fonts ──────────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,400;0,9..144,600;1,9..144,300;1,9..144,400&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── Root Variables ────────────────────────────────────────────────── */
    :root {
        --paper: #F7F5F0;
        --ink: #111111;
        --graphite: #444444;
        --divider: rgba(17,17,17,0.10);
        --smoke: rgba(229,226,217,0.30);
        --accent-green: #057A55;
        --accent-red: #B91C1C;
        --sans: 'Inter', -apple-system, sans-serif;
        --serif: 'Fraunces', Georgia, serif;
        --mono: 'JetBrains Mono', 'Consolas', monospace;
    }

    /* ── Global Reset ──────────────────────────────────────────────────── */
    .stApp {
        background-color: var(--paper) !important;
        color: var(--ink) !important;
        font-family: var(--sans) !important;
    }

    /* Override Streamlit's default dark sidebar */
    section[data-testid="stSidebar"] {
        background-color: #EFEDE8 !important;
        border-right: 1px solid var(--divider) !important;
    }
    section[data-testid="stSidebar"] * {
        color: var(--ink) !important;
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stMultiSelect label {
        font-family: var(--mono) !important;
        font-size: 12px !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        color: var(--graphite) !important;
    }

    /* ── Headers ────────────────────────────────────────────────────────── */
    h1 {
        font-family: var(--serif) !important;
        font-size: 3.2rem !important;
        font-weight: 300 !important;
        line-height: 0.95 !important;
        letter-spacing: -0.02em !important;
        color: var(--ink) !important;
        margin-bottom: 1.5rem !important;
    }
    h2 {
        font-family: var(--serif) !important;
        font-size: 2rem !important;
        font-weight: 400 !important;
        line-height: 1.1 !important;
        color: var(--ink) !important;
    }
    h3 {
        font-family: var(--serif) !important;
        font-size: 1.4rem !important;
        font-weight: 400 !important;
        color: var(--ink) !important;
    }

    /* ── Body Text ─────────────────────────────────────────────────────── */
    p, li, .stMarkdown {
        font-family: var(--sans) !important;
        font-size: 1.05rem !important;
        line-height: 1.7 !important;
        color: var(--graphite) !important;
    }

    /* ── Chapter Marker ────────────────────────────────────────────────── */
    .chapter-marker {
        font-family: var(--mono);
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--graphite);
        margin-bottom: 0.5rem;
    }

    /* ── Section Divider ───────────────────────────────────────────────── */
    .section-divider {
        border: none;
        border-top: 1px solid var(--divider);
        margin: 3rem 0 2rem 0;
    }

    /* ── Editorial Hero ────────────────────────────────────────────────── */
    .editorial-hero {
        padding: 4rem 0 3rem 0;
        border-bottom: 1px solid var(--divider);
        margin-bottom: 3rem;
    }
    .editorial-hero h1 {
        font-size: 4rem !important;
        font-weight: 300 !important;
        line-height: 0.9 !important;
        letter-spacing: -0.03em !important;
    }
    .editorial-hero em {
        font-family: var(--serif) !important;
        font-style: italic !important;
        font-weight: 300 !important;
    }
    .hero-subtitle {
        font-family: var(--sans);
        font-size: 1.15rem;
        font-weight: 300;
        color: var(--graphite);
        line-height: 1.6;
        max-width: 640px;
        margin-top: 1.5rem;
    }

    /* ── Status Badge ──────────────────────────────────────────────────── */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 16px;
        border-radius: 100px;
        border: 1px solid var(--divider);
        font-family: var(--mono);
        font-size: 12px;
        color: var(--graphite);
        margin-bottom: 2rem;
    }
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--accent-green);
        display: inline-block;
    }

    /* ── Metric Cards (Editorial) ──────────────────────────────────────── */
    .metric-editorial {
        padding: 2rem 1.5rem;
        border: 1px solid var(--divider);
        border-radius: 2px;
        background: transparent;
        transition: all 0.2s ease-in-out;
    }
    .metric-editorial:hover {
        border-color: var(--ink);
        background: var(--smoke);
    }
    .metric-number {
        font-family: var(--serif);
        font-size: 2.8rem;
        font-weight: 300;
        color: var(--ink);
        line-height: 1;
        margin-bottom: 0.5rem;
    }
    .metric-number.green { color: var(--accent-green); }
    .metric-number.red { color: var(--accent-red); }
    .metric-unit {
        font-family: var(--mono);
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--graphite);
    }
    .metric-desc {
        font-family: var(--sans);
        font-size: 0.85rem;
        color: var(--graphite);
        margin-top: 0.75rem;
        line-height: 1.5;
    }

    /* ── Insight Box (Editorial Blockquote) ─────────────────────────────── */
    .editorial-quote {
        border-left: 2px solid var(--ink);
        padding: 1.5rem 0 1.5rem 2rem;
        margin: 2rem 0;
    }
    .editorial-quote p {
        font-family: var(--serif) !important;
        font-style: italic !important;
        font-size: 1.15rem !important;
        line-height: 1.6 !important;
        color: var(--ink) !important;
    }
    .editorial-quote .attribution {
        font-family: var(--mono);
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--graphite);
        margin-top: 1rem;
        font-style: normal;
    }

    /* ── Warning / Finding Box ─────────────────────────────────────────── */
    .finding-box {
        background: var(--smoke);
        padding: 2rem;
        border-radius: 2px;
        margin: 2rem 0;
        border: 1px solid var(--divider);
    }
    .finding-box .finding-label {
        font-family: var(--mono);
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--accent-green);
        margin-bottom: 0.75rem;
    }
    .finding-box.warning .finding-label {
        color: var(--accent-red);
    }
    .finding-box p {
        font-family: var(--sans) !important;
        font-size: 1rem !important;
        line-height: 1.7 !important;
        color: var(--ink) !important;
    }

    /* ── Comparison Panel ──────────────────────────────────────────────── */
    .compare-panel {
        border: 1px solid var(--divider);
        border-radius: 2px;
        padding: 2rem;
        transition: all 0.2s ease-in-out;
    }
    .compare-panel:hover {
        border-color: var(--ink);
    }
    .compare-tag {
        font-family: var(--mono);
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        padding: 3px 10px;
        border-radius: 2px;
        display: inline-block;
        margin-bottom: 1rem;
    }
    .compare-tag.legacy {
        background: rgba(185,28,28,0.10);
        color: var(--accent-red);
    }
    .compare-tag.modern {
        background: rgba(5,122,85,0.10);
        color: var(--accent-green);
    }

    /* ── Timeline ──────────────────────────────────────────────────────── */
    .timeline-item {
        position: relative;
        padding-left: 2.5rem;
        padding-bottom: 2rem;
        border-left: 1px solid var(--divider);
        margin-left: 0.5rem;
    }
    .timeline-item::before {
        content: '';
        position: absolute;
        left: -8px;
        top: 4px;
        width: 16px;
        height: 16px;
        border-radius: 50%;
        background: var(--paper);
        border: 2px solid var(--ink);
    }
    .timeline-item .tl-title {
        font-family: var(--sans);
        font-weight: 600;
        font-size: 1rem;
        color: var(--ink);
    }
    .timeline-item .tl-desc {
        font-family: var(--sans);
        font-size: 0.9rem;
        color: var(--graphite);
        line-height: 1.6;
        margin-top: 0.25rem;
    }

    /* ── Big Number ─────────────────────────────────────────────────────── */
    .big-number {
        font-family: var(--serif);
        font-size: 6rem;
        font-weight: 300;
        color: rgba(17,17,17,0.08);
        line-height: 1;
        margin-bottom: -1.5rem;
    }

    /* ── Tables ─────────────────────────────────────────────────────────── */
    .stDataFrame {
        border: 1px solid var(--divider) !important;
        border-radius: 2px !important;
    }

    /* ── Expander ───────────────────────────────────────────────────────── */
    .streamlit-expanderHeader {
        font-family: var(--sans) !important;
        font-weight: 500 !important;
        color: var(--ink) !important;
        border: 1px solid var(--divider) !important;
        border-radius: 2px !important;
    }

    /* ── Plotly Charts — light theme ───────────────────────────────────── */
    .js-plotly-plot .plotly .main-svg {
        background: transparent !important;
    }

    /* ── Navigation Bar ────────────────────────────────────────────────── */
    .nav-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1.5rem 0;
        border-bottom: 1px solid var(--divider);
        margin-bottom: 2rem;
    }
    .nav-brand {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .nav-brand .brand-mark {
        width: 12px;
        height: 12px;
        background: var(--ink);
        display: inline-block;
    }
    .nav-brand span {
        font-family: var(--mono);
        font-size: 14px;
        color: var(--ink);
    }

    /* ── Streamlit overrides ───────────────────────────────────────────── */
    .stMetric label { font-family: var(--mono) !important; font-size: 12px !important; text-transform: uppercase !important; letter-spacing: 0.05em !important; }
    .stMetric [data-testid="stMetricValue"] { font-family: var(--serif) !important; font-weight: 300 !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 0 !important; }
    .stTabs [data-baseweb="tab"] { font-family: var(--mono) !important; font-size: 13px !important; text-transform: uppercase !important; letter-spacing: 0.05em !important; }
    [data-testid="stHeader"] { background: var(--paper) !important; }
    .block-container { max-width: 1280px !important; padding-top: 2rem !important; }
</style>
"""


# ── Plotly Light Theme ────────────────────────────────────────────────────────
PLOTLY_LIGHT_TEMPLATE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(247,245,240,0.5)",
    font=dict(family="Inter, sans-serif", size=13, color="#444444"),
    title=dict(font=dict(family="Fraunces, Georgia, serif", size=20, color="#111111")),
    xaxis=dict(gridcolor="rgba(17,17,17,0.08)", linecolor="rgba(17,17,17,0.15)", tickfont=dict(size=11)),
    yaxis=dict(gridcolor="rgba(17,17,17,0.08)", linecolor="rgba(17,17,17,0.15)", tickfont=dict(size=11)),
    colorway=["#111111", "#057A55", "#B91C1C", "#3498db", "#f39c12", "#9b59b6"],
)


def apply_editorial_layout(fig):
    """Apply the editorial light theme to a Plotly figure."""
    fig.update_layout(**PLOTLY_LIGHT_TEMPLATE)
    return fig


def chapter(number, title):
    """Return HTML for a chapter marker."""
    return f'<div class="chapter-marker">{number:02d} — {title}</div>'


def divider():
    """Return HTML for a section divider."""
    return '<hr class="section-divider">'


def finding(text, label="KEY FINDING", warning=False):
    """Return HTML for a finding/insight box."""
    cls = "finding-box warning" if warning else "finding-box"
    return f'<div class="{cls}"><div class="finding-label">{label}</div><p>{text}</p></div>'


def quote(text, attribution=""):
    """Return HTML for an editorial blockquote."""
    attr = f'<div class="attribution">— {attribution}</div>' if attribution else ""
    return f'<div class="editorial-quote"><p>{text}</p>{attr}</div>'


def metric_card(value, unit, description="", color_class="", desc=""): description = description or desc
    """Return HTML for an editorial metric card."""
    return f"""
    <div class="metric-editorial">
        <div class="metric-number {color_class}">{value}</div>
        <div class="metric-unit">{unit}</div>
        <div class="metric-desc">{description}</div>
    </div>
    """


def big_number(num):
    """Return HTML for a decorative big background number."""
    return f'<div class="big-number">{num:02d}</div>'
