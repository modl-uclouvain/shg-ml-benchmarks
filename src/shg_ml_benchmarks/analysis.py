"""This module defines functions for the analysis of the benchmark results."""

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

from shg_ml_benchmarks.utils import (
    BENCHMARKS_DIR,
    RESULTS_DIR,
    SHG_BENCHMARK_SPLITS,
    load_holdout,
)


def compute_metrics(true_values, pred_values):
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
    return compute_metrics(true_values, pred_values)


def gather_results() -> dict:
    """Loop over all benchmarks folders, all task folders and all
    hyperparameter sets to compute consistent metrics.

    Write out a JSON summary into the results folder.

    """

    split_results_nested: dict[str, dict] = {
        split: {} for split in SHG_BENCHMARK_SPLITS
    }

    benchmarks = BENCHMARKS_DIR.glob("*")

    for b in benchmarks:
        benchmark_results: dict[str, dict] = {}
        if b.name.startswith("."):
            continue

        benchmark_name = b.name
        print(benchmark_name)

        task_dirs = b.glob("tasks*")

        benchmark_results[b.name] = {}

        for t in task_dirs:
            if t.name == "tasks":
                task_name = "default"
            else:
                task_name = t.name.split("_")[-1]
            print("\t" + task_name)

            benchmark_results[benchmark_name][task_name] = {}

            for split in t.glob("*"):
                if split.name not in SHG_BENCHMARK_SPLITS:
                    warnings.warn(f"Found unknown split: {split.name}, skipping")
                    continue
                print("\t\t" + split.name)

                for results in split.glob("*results.json"):
                    if "_" in results.name:
                        results_label = results.name.split("_")[0]
                    else:
                        results_label = "default"

                    print("\t\t\t" + results_label)

                    benchmark_results[benchmark_name][task_name][results_label] = {}

                    with open(results) as f:
                        results_data = json.load(f)

                    pred_dict = results_data["predictions"]
                    true_df = load_holdout(split.name)

                    metrics = evaluate_predictions(
                        pred_dict, true_df, target="dKP_full_neum"
                    )
                    benchmark_results[benchmark_name][task_name][results_label] = (
                        metrics
                    )
                    split_results_nested[split.name].update(benchmark_results)
                    print(
                        f"\t\t\t\tMAE: {metrics['mae']:.1f} pm/V, RMSE: {metrics['rmse']:.1f} pm/V, Spearman: {metrics['spearman']:.1f}, R^2: {metrics['r2_score']:.1f}"
                    )

    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(split_results_nested, f, indent=4, allow_nan=True)

    return split_results_nested


def load_summary():
    with open(RESULTS_DIR / "summary.json") as f:
        summary = json.load(f)

    return summary


def global_bar_plot():
    summary = load_summary()

    grouping = {
        "Dummy": ["median_value", "mean_value"],
        "LLMs": ["openai-gpt-4o", "claude-3.5-sonnet", "darwin-1.5", "deepseek-chat"],
        "GNNs": ["megnet", "coGN", "coNGN"],
        "Tensor network": ["matten"],
        "Tree-based": ["et", "lgbm"],
        "Neural networks": ["modnet", "modnet_nan"],
    }

    dark2 = plt.get_cmap("Dark2")

    group_colours = {
        "LLMs": dark2(0),
        "Tree-based": dark2(1),
        "Neural networks": dark2(2),
        "GNNs": dark2(3),
        "Tensor network": dark2(4),
        "Dummy": "gray",
    }

    group_patterns = {
        "LLMs": "/",
        "Tree-based": "\\",
        "Neural networks": "o",
        "GNNs": "O",
        "Tensor network": "+",
        "Dummy": "x",
    }

    main_metric = "spearman"

    metrics = {"mae": "min", "spearman": "max"}

    metric_labels = {"mae": "MAE (pm/V)", "spearman": "Spearman correlation"}

    metric_limits = {"mae": [0, 20], "spearman": [-0.3, 1]}

    plt.rcParams["font.family"] = "Liberation Sans"

    for split, split_data in summary.items():
        group_data = {}
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        for group, group_models in grouping.items():
            group_data[group] = {}
            for model, model_data in split_data.items():
                if model not in group_models:
                    continue
                best_run = None
                best_metric = None
                best_tag = None
                for run, runs in model_data.items():
                    for tag, metrics in runs.items():
                        if (
                            best_run is None
                            or (
                                metrics[main_metric] < best_metric
                                and metrics[main_metric] == "min"
                            )
                            or (
                                metrics[main_metric] > best_metric
                                and metrics[main_metric] == "max"
                            )
                        ):
                            best_run = run
                            best_tag = tag
                            best_metric = metrics[main_metric]

                group_data[group][model] = model_data[best_run][best_tag]

        bar_count = 0
        for group in group_data:
            bars = []
            label = group
            for ind, model in enumerate(group_data[group]):
                position = bar_count
                bars.append((model, group_data[group][model]["mae"]))
                ax.bar(
                    position,
                    group_data[group][model][main_metric] or 0,
                    label=label if ind == 0 else None,
                    edgecolor=group_colours[group],
                    lw=2,
                    color="white",
                    hatch=group_patterns[group],
                )
                ax.text(
                    position,
                    min(metric_limits[main_metric]) - 0.01,
                    model,
                    ha="center",
                    va="top",
                    rotation=45,
                    fontsize=8,
                )
                ax.set_xticks([])
                bar_count += 1

        ax.legend()
        ax.set_ylabel(metric_labels[main_metric])
        ax.set_ylim(metric_limits[main_metric])

        plt.savefig(RESULTS_DIR / f"{split}_bar_plot.png")


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
