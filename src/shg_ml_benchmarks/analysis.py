"""This module defines functions for the analysis of the benchmark results."""

import json
import typing
import warnings
from pathlib import Path

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Liberation Sans"
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from scipy.stats import spearmanr
from sklearn.metrics import auc, r2_score

from shg_ml_benchmarks.utils import (
    BENCHMARKS_DIR,
    RESULTS_DIR,
    SHG_BENCHMARK_SPLITS,
    load_holdout,
    load_train,
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
) -> dict[str, typing.Any]:
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

    benchmarks = sorted(BENCHMARKS_DIR.glob("*"))

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
            if not task_name:
                print(f"Ignoring empty task name in {t}")
                continue

            benchmark_results[benchmark_name][task_name] = {}

            for split in sorted(t.glob("*")):
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

                    metrics["source"] = str(
                        results.relative_to(Path(__file__).parent.parent.parent)
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


def pareto_fit(x, a=245.338, b=-0.7378):
    return a * np.exp(b * x)


def compute_enrichment(
    model, split, results_fname, discovery_thresholds=None, top_percent: float = 5
):
    holdout = load_holdout(split)

    if not discovery_thresholds:
        discovery_thresholds = [0.5, 0.7, 0.9, 0.95]

    if not results_fname.exists():
        return None
    with open(results_fname) as f:
        results = json.load(f)

    predictions = pd.DataFrame.from_dict(
        results["predictions"], orient="index", columns=["dKP_pred"]
    )
    if not all(predictions.index == holdout.index):
        raise ValueError("Mismatching indices between predictions and holdout")

    # Compute FOM as y-distance relative to fitted Pareto
    holdout["FOM"] = holdout["dKP_full_neum"] - pareto_fit(holdout["src_bandgap"])
    predictions["FOM"] = predictions["dKP_pred"] - pareto_fit(holdout["src_bandgap"])

    pred_fom = np.array(predictions["FOM"])
    real_fom = np.array(holdout["FOM"])

    # Determine actual top materials
    n_materials = len(real_fom)
    n_top = int(n_materials * top_percent / 100)
    top_indices = np.argsort(-real_fom)[:n_top]
    top_mask = np.zeros(n_materials, dtype=bool)
    top_mask[top_indices] = True

    # Rank materials by predicted values
    rank_indices = np.argsort(-pred_fom)

    # Calculate discovery curve
    discovery_curve_x = []  # % of materials evaluated
    discovery_curve_y = []  # % of top materials discovered

    materials_needed = {}  # % materials needed to discover X% of top materials

    found_top = 0
    for i, idx in enumerate(rank_indices):
        if top_mask[idx]:
            found_top += 1

        percent_evaluated = ((i + 1) / n_materials) * 100
        percent_discovered = (found_top / n_top) * 100

        discovery_curve_x.append(percent_evaluated)
        discovery_curve_y.append(percent_discovered)

        # Check if we've crossed any of our discovery thresholds
        for threshold in discovery_thresholds:
            if percent_discovered >= threshold and threshold not in materials_needed:
                materials_needed[threshold] = percent_evaluated

    # Calculate enrichment factor at specified percentage
    ef_index = int(n_materials * top_percent / 100) - 1
    if ef_index < 0:
        ef_index = 0
    ef_value = discovery_curve_y[ef_index] / top_percent

    # Calculate AUC (normalized so random = 0.5, perfect = 1.0)
    curve_auc = auc(discovery_curve_x, discovery_curve_y) / (100 * 100)
    normalized_auc = 2 * (curve_auc - 0.5)

    return {
        "enrichment_factor": ef_value,
        "auc": curve_auc,
        "normalized_auc": normalized_auc,
        "materials_needed": materials_needed,
        "discovery_curve": {"x": discovery_curve_x, "y": discovery_curve_y},
    }


def plot_discovery_curves(split, top_percent=10.0):
    models = sorted(BENCHMARKS_DIR.glob("*"))
    enrichment_metrics = {}
    for model in models:
        if model.name.startswith("."):
            continue
        results_files = model.glob(f"task*/{split}/*results.json")
        # breakpoint()
        # tags = [f.name.split("_")[0] for f in results_files]
        for results_fname in results_files:
            tag = results_fname.name.split("_")[0]
            if tag == "results.json":
                tag = None
            result = compute_enrichment(
                model.name, split, results_fname, top_percent=top_percent
            )
            if result is not None:
                enrichment_metrics[
                    f"{model.name}-{tag}" if tag is not None else model.name
                ] = result

    fig, axes = plt.subplots(2, 1, figsize=(10, 12))
    for m in sorted(
        enrichment_metrics,
        key=lambda x: enrichment_metrics[x]["enrichment_factor"],
        reverse=True,
    ):
        if enrichment_metrics[m]["enrichment_factor"] < 1.0:
            continue
        label = f"{m} (EF: {enrichment_metrics[m]['enrichment_factor']:.1f})"
        linestyle = "-"
        c = None
        if m == "median_value":
            label = "Median value"
            linestyle = "--"
            c = "k"
        axes[0].plot(
            enrichment_metrics[m]["discovery_curve"]["x"],
            enrichment_metrics[m]["discovery_curve"]["y"],
            # label=f"{m} (EF: {enrichment_metrics[m]['enrichment_factor']:.1f}, AUC: {enrichment_metrics[m]['normalized_auc']:.2f})",
            label=label,
            linestyle=linestyle,
            c=c,
        )
    axes[0].legend()

    # perfect discovery curve
    perfect_curve_x = [0, top_percent, 100]
    perfect_curve_y = [0, 100, 100]

    axes[0].plot(
        perfect_curve_x,
        perfect_curve_y,
        linestyle="--",
        color="black",
        label="Perfect oracle",
    )

    axes[0].set_xlabel("% of materials evaluated")
    axes[0].set_ylabel("% of top materials discovered")
    axes[0].set_xlim(0, 20)
    axes[0].set_ylim(0, 100)

    holdout = load_holdout(split)
    holdout["FOM"] = holdout["dKP_full_neum"] - pareto_fit(holdout["src_bandgap"])
    axes[1].scatter(holdout["src_bandgap"], holdout["dKP_full_neum"], c=holdout["FOM"])

    train = load_train(split)
    axes[1].scatter(
        train["src_bandgap"], train["dKP_full_neum"], c="black", marker="x", zorder=0
    )
    # plot top percent in different colours
    top_materials = holdout.nlargest(int(len(holdout) * top_percent / 100), "FOM")
    axes[1].plot(
        top_materials["src_bandgap"],
        top_materials["dKP_full_neum"],
        c="red",
        marker="*",
    )
    axes[1].set_xlim(0, 10)
    axes[1].set_ylim(-10, 200)
    axes[1].set_ylabel(r"$d_\text{KP}$ (pm/V)")
    axes[1].set_xlabel("Band gap (eV)")
    plt.savefig(RESULTS_DIR / f"discovery_curves-{split}.png", dpi=300)


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

    for split, split_data in summary.items():
        group_data = {}
        fig, ax = plt.subplots(1, 1, figsize=(6.5, 4))
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

        try:
            pio.kaleido.scope.mathjax = None  # To remove MathJax box in pdf
        except Exception:
            pass
        figs_path = path / "parity_plot_pred_true"
        figs_path.parent.mkdir(parents=True, exist_ok=True)

        fig.write_image(f"{str(figs_path)}.pdf")
        fig.write_image(f"{str(figs_path)}.svg")
        fig.write_image(f"{str(figs_path)}.png", scale=10)
        fig.write_html(f"{str(figs_path)}.html")

    return fig


if __name__ == "__main__":
    plot_discovery_curves("distribution_125")
    plot_discovery_curves("distribution_250")
