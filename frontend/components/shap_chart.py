"""Plotly horizontal bar chart for SHAP feature contributions."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go


def shap_chart(top_features: list[dict[str, Any]], request_id: str = "") -> go.Figure:
    """Return a Plotly horizontal bar chart of SHAP feature contributions.

    Bars are coloured red for positive contributions (increases risk) and
    blue for negative ones (decreases risk).  Feature labels include the
    raw feature value so clinicians can read the chart without cross-
    referencing the input form.

    Args:
        top_features: List of dicts from the API, each with keys
            ``feature`` (str), ``shap_value`` (float), ``feature_value`` (float).
            Typically 5 items, sorted by descending |shap_value|.
        request_id: Optional request ID shown as a subtitle.

    Returns:
        A go.Figure containing a single Bar trace (horizontal).
    """
    if not top_features:
        fig = go.Figure()
        fig.update_layout(
            title="No feature contributions available",
            height=200,
        )
        return fig

    # Display order: largest |shap| at top → reverse for Plotly (bottom-to-top y axis).
    features = list(reversed(top_features))

    # feature names are pre-humanized and include the value in parentheses.
    labels = [f["feature"] for f in features]
    values = [f["shap_value"] for f in features]
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in values]
    hover = [
        f"<b>{f['feature']}</b><br>"
        f"SHAP: {f['shap_value']:+.4f} log-odds"
        for f in features
    ]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.3f}" for v in values],
            textposition="outside",
            hovertext=hover,
            hoverinfo="text",
        )
    )

    fig.update_layout(
        title={
            "text": "Top contributing factors (impact on log-odds of readmission)",
            "font": {"size": 15},
        },
        xaxis={
            "title": "SHAP value (log-odds)",
            "zeroline": True,
            "zerolinewidth": 2,
            "zerolinecolor": "#555555",
        },
        yaxis={"automargin": True},
        height=max(260, 50 * len(features) + 80),
        margin={"t": 70, "b": 40, "l": 10, "r": 80},
        paper_bgcolor="white",
        plot_bgcolor="#f9f9f9",
        font={"family": "Arial, sans-serif"},
        showlegend=False,
    )

    return fig


def _fmt_value(v: float) -> str:
    """Format a feature value for display (strip trailing .0 for integers)."""
    if v == int(v):
        return str(int(v))
    return f"{v:.3g}"
