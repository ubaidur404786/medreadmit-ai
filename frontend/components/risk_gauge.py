"""Plotly gauge chart for 30-day readmission risk probability."""

from __future__ import annotations

import plotly.graph_objects as go


def risk_gauge(
    probability: float,
    threshold_low: float = 0.10,
    threshold_high: float = 0.30,
) -> go.Figure:
    """Return a Plotly gauge figure for a readmission risk probability.

    Colour bands:
        green  [0, threshold_low)     — low risk
        yellow [threshold_low, threshold_high) — moderate risk
        red    [threshold_high, 1]    — high risk

    Args:
        probability: Predicted readmission probability in [0, 1].
        threshold_low: Upper bound of the low-risk band (default 0.10).
        threshold_high: Lower bound of the high-risk band (default 0.30).

    Returns:
        A go.Figure containing a single Indicator (gauge) trace.
    """
    if probability < threshold_low:
        risk_label = "LOW RISK"
        bar_color = "#2ecc71"
    elif probability < threshold_high:
        risk_label = "MODERATE RISK"
        bar_color = "#f39c12"
    else:
        risk_label = "HIGH RISK"
        bar_color = "#e74c3c"

    pct = round(probability * 100, 1)

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=pct,
            number={"suffix": "%", "font": {"size": 36}},
            title={"text": f"<b>{risk_label}</b>", "font": {"size": 18}},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%", "tickwidth": 1},
                "bar": {"color": bar_color, "thickness": 0.25},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "#cccccc",
                "steps": [
                    {"range": [0, threshold_low * 100], "color": "#d5f5e3"},
                    {
                        "range": [threshold_low * 100, threshold_high * 100],
                        "color": "#fef9e7",
                    },
                    {"range": [threshold_high * 100, 100], "color": "#fdedec"},
                ],
                "threshold": {
                    "line": {"color": "#555555", "width": 3},
                    "thickness": 0.75,
                    "value": pct,
                },
            },
        )
    )

    fig.update_layout(
        height=280,
        margin={"t": 60, "b": 10, "l": 20, "r": 20},
        paper_bgcolor="white",
        font={"family": "Arial, sans-serif"},
    )
    return fig
