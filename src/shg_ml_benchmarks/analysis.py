"""This module defines functions for the analysis of the benchmark results."""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from scipy.stats import spearmanr
from sklearn.metrics import r2_score


def evaluate_predictions(
    predictions: dict[str, float | np.ndarray], holdout_df: pd.DataFrame, target: str
) -> dict[str, float]:
    """Calculate evaluation metrics.

    Args:
        predictions: Dictionary mapping structure IDs to predictions
        holdout_df: DataFrame of the holdout set
        target: name of the column with the target in holdout_df

    Returns:
        Dictionary with evaluation metrics
    """
    true_values = []
    pred_values = []
    for structure_id, pred in predictions.items():
        true_values.append(holdout_df.loc[structure_id][target])
        pred_values.append(pred)

    # Calculate metrics
    mae = np.mean(np.abs(np.array(true_values) - np.array(pred_values)))
    rmse = np.sqrt(np.mean((np.array(true_values) - np.array(pred_values)) ** 2))
    spearmanrho = spearmanr(np.array(true_values), np.array(pred_values)).statistic
    r2score = r2_score(np.array(true_values), np.array(pred_values))

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "spearman": float(spearmanrho),
        "r2_score": float(r2score),
    }


def visualize_predictions(
    predictions: dict[str, float | np.ndarray],
    holdout_df: pd.DataFrame,
    target: str,
    path: Path | None = None,
) -> go.Figure:
    """Save a visualization of the predictions wrt. the true values.

    Args:
        predictions: Dictionary mapping structure IDs to predictions
        holdout_df: DataFrame of the holdout set
        target: name of the column with the target in holdout_df
        path: optional folder in which to save figures.

    Returns:
        go.Figure
    """
    true_values = []
    pred_values = []
    for structure_id, pred in predictions.items():
        true_values.append(holdout_df.loc[structure_id][target])
        pred_values.append(pred)

    # Scatter plot for previous outputs.
    scatter_plot = go.Scatter(
        x=true_values,
        y=pred_values,
        mode="markers",
        name="",
        showlegend=False,
        text=[mpid for mpid in predictions.keys()],
    )

    ideal = go.Scatter(
        x=[-1, 200],
        y=[-1, 200],
        mode="lines",
        line=dict(color="gray", dash="dot"),
        showlegend=False,
    )

    # Layout
    layout = go.Layout(
        # title=dict(text='Scatter Plot'),
        xaxis=dict(title="<i>d</i><sub>KP</sub> (pm/V)", range=[-1, 170]),
        yaxis=dict(title="<i>d&#770;</i><sub>KP</sub> (pm/V)", range=[-1, 170]),
        # legend=dict(font=dict(size=12)),
    )

    # Create figure
    fig = go.Figure(data=[scatter_plot, ideal], layout=layout)

    fig.update_layout(
        autosize=False,
        font_size=20,
        width=600,
        height=600,
        # plot_bgcolor="white",
        template="simple_white",
    )
    fig.update_layout(
        xaxis=dict(
            tickmode="linear",
            tick0=0,
            dtick=20,
            showgrid=False,
        ),
        yaxis=dict(
            tickmode="linear",
            tick0=0,
            dtick=20,
            showgrid=False,
        ),
    )
    if path:
        import plotly.io as pio

        pio.kaleido.scope.mathjax = None  # To remove MathJax box in pdf
        path.parent.mkdir(parents=True, exist_ok=True)
        figs_path = path / "parity_plot_pred_true"

        fig.write_image(f"{str(figs_path)}.pdf")
        fig.write_image(f"{str(figs_path)}.svg")
        fig.write_image(f"{str(figs_path)}.png", scale=10)
        fig.write_html(f"{str(figs_path)}.html")

    return fig
