import re

with open("utils/charts.py", "r") as f:
    content = f.read()

# Add import for apply_editorial_layout
if "from utils.theme import apply_editorial_layout" not in content:
    content = content.replace(
        "from utils.constants import",
        "from utils.theme import apply_editorial_layout\nfrom utils.constants import"
    )

# Replace _apply_layout with apply_editorial_layout
def replacement(match):
    # Keep the layout overrides (title, etc)
    fig_name = match.group(1)
    title = match.group(2)
    xaxis = match.group(3)
    yaxis = match.group(4)
    return f"""
    fig.update_layout(title=dict(text={title}), xaxis_title={xaxis}, yaxis_title={yaxis})
    return apply_editorial_layout({fig_name})
    """

# We need to change the function `def _apply_layout` and all its calls.
# Let's just redefine _apply_layout to call apply_editorial_layout.

content = re.sub(
    r"def _apply_layout\(fig: go\.Figure, title: str, xaxis: str = \"\", yaxis: str = \"\"\) -> go\.Figure:.*?(?=def |$)",
    r'''def _apply_layout(fig: go.Figure, title: str, xaxis: str = "", yaxis: str = "") -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font_size=18, font_family="Fraunces, serif"),
        xaxis_title=xaxis,
        yaxis_title=yaxis,
    )
    return apply_editorial_layout(fig)
''',
    content,
    flags=re.DOTALL
)

with open("utils/charts.py", "w") as f:
    f.write(content)
print("Charts updated")
