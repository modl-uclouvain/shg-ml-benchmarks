import importlib.metadata
import json
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from pymatgen.core import Structure
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

from shg_ml_benchmarks.utils import BENCHMARKS_DIR, load_holdout, load_train

__version__ = importlib.metadata.version("shg-ml-benchmarks")


def load_and_split_data(
    data_path: str, holdout_path: str
) -> tuple[dict[str, dict], dict[str, dict]]:
    """Load and split data into train/test sets.

    Args:
        data_path: Path to JSON file with structure data and targets
        holdout_path: Path to JSON file with holdout set IDs

    Returns:
        Tuple of (train_data, test_data) dictionaries
    """
    # Load main data
    with open(data_path) as f:
        data = json.load(f)

    # Load holdout IDs
    with open(holdout_path) as f:
        holdout_ids = set(json.load(f))

    # Split data
    train_data = {}
    test_data = {}

    for structure_id, entry in data.items():
        # Convert structure dict to pymatgen Structure
        structure = Structure.from_dict(entry["structure"])
        target = entry["target"]  # Either coefficient or tensors

        if structure_id in holdout_ids:
            test_data[structure_id] = {"structure": structure, "target": target}
        else:
            train_data[structure_id] = {"structure": structure, "target": target}

    return train_data, test_data

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
    mae         = np.mean(np.abs(np.array(true_values) - np.array(pred_values)))
    rmse        = np.sqrt(np.mean((np.array(true_values) - np.array(pred_values)) ** 2))
    spearmanrho = spearmanr(np.array(true_values), np.array(pred_values)).statistic
    r2score     = r2_score(np.array(true_values), np.array(pred_values))

    return {"mae": float(mae), "rmse": float(rmse), "spearman": float(spearmanrho), "r2_score": float(r2score)}

def visualize_predictions(
    predictions: dict[str, float | np.ndarray], holdout_df: pd.DataFrame, target: str
) -> dict[str, float]:
    """Save a visualization of the predictions wrt. the true values.

    Args:
        predictions: Dictionary mapping structure IDs to predictions
        holdout_df: DataFrame of the holdout set
        target: name of the column with the target in holdout_df

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
        mode='markers',
        name=f'',
        showlegend=False,
        text=[mpid for mpid in predictions.keys()]
    )

    ideal = go.Scatter(
        x=[-1,200],
        y=[-1,200],
        mode="lines",
        line=dict(color='gray', dash='dot'),
        showlegend=False
    )

    # Layout
    layout = go.Layout(
        # title=dict(text='Scatter Plot'),
        xaxis=dict(title='<i>d</i><sub>KP</sub> (pm/V)',  range=[-1,170]),
        yaxis=dict(title='<i>d&#770;</i><sub>KP</sub> (pm/V)', range=[-1,170]),
        # legend=dict(font=dict(size=12)),
    )

    # Create figure
    fig = go.Figure(data=[scatter_plot,ideal], layout=layout)

    fig.update_layout(
        autosize=False,
        font_size=20,
        width=600,
        height=600,
        # plot_bgcolor="white",
        template='simple_white',
    )
    fig.update_layout(
        xaxis = dict(
            tickmode = 'linear',
            tick0 = 0,
            dtick = 20,
            showgrid=False,
        ),
        yaxis = dict(
            tickmode = 'linear',
            tick0 = 0,
            dtick = 20,
            showgrid=False,
        ),
    )

    return fig



def run_benchmark(
    model: Any,
    predict_fn: Callable[[Any, Structure], float | np.ndarray],
    train_fn: Callable | None = None,
    task: str = "random_250",
    target: str = "dKP_full_neum",
    write_results: bool = True,
    model_label=None,
    model_tags=None,
    predict_individually=True,
) -> dict | None:
    """Run benchmark using provided training and prediction functions.

    Args:
        model: Model object
        predict_fn: Function that takes (model, structure) and returns prediction
        train_fn: Function that takes (structures, target) and returns model (optional)
        task: the task to run; corresponds to the filenames of pre-defined holdout sets found in `./data`.
        target: the target property to predict
        write_results: whether to write the results to disk

    Returns:
        Dictionary with benchmark results and metrics
    """
    if write_results:
        # Check if the benchmark has already been run
        if getattr(model, "tags", None) is not None:
            results_fname = f"{model.tags}_results.json"
        elif model_tags is not None:
            results_fname = f"{model_tags}_results.json"
        else:
            results_fname = "results.json"
        if getattr(model, "label", None) is not None:
            results_path = BENCHMARKS_DIR / model.label / "tasks" / task / results_fname
        else:
            results_path = BENCHMARKS_DIR / model_label / "tasks" / task / results_fname

        if results_path.exists():
            print(
                "Benchmark has already been run. Use `write_results=False` to skip writing results."
            )
            return None

    # Load data
    holdout_df = load_holdout(task)
    train_df = load_train(task)

    if train_fn:
        model = train_fn(train_df, target=target)

    # Get predictions
    predictions = {}
    uncertainties = {}
    if predict_individually:
        for structure_id, entry in holdout_df.iterrows():
            try:
                pred, unc = predict_fn(model, Structure.from_dict(entry["structure"]))
            except ValueError:
                pred = predict_fn(model, Structure.from_dict(entry["structure"]))
                unc = None
            # Convert numpy types to Python native types for JSON serialization
            if isinstance(pred, np.ndarray):
                pred = pred.tolist()
            elif isinstance(pred, np.generic):
                pred = pred.item()
            predictions[structure_id] = pred
            if unc:
                uncertainties[structure_id] = unc
    else:
        try:
            df_pred, df_unc = predict_fn(model, [Structure.from_dict(s) for s in holdout_df['structure']], holdout_df.index.tolist())
        except ValueError:
            df_pred = predict_fn(model, [Structure.from_dict(s) for s in holdout_df['structure']], holdout_df.index.tolist())
            df_unc = None
        predictions = df_pred[df_pred.columns[0]].to_dict()
        uncertainties = df_unc[df_unc.columns[0]].to_dict()

    # Calculate metrics
    metrics = evaluate_predictions(predictions, holdout_df, target)
    fig_parity_plot = visualize_predictions(predictions, holdout_df, target)

    # Compile results
    if uncertainties!={}:
        results = {"predictions": predictions, "uncertainties": uncertainties, "metrics": metrics}
    else:
        results = {"predictions": predictions, "metrics": metrics}

    if write_results:
        results_path.parent.mkdir(parents=True, exist_ok=True)

        # Save results
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        # Save figure
        fig_fname = "parity_plot_pred_true"
        figs_path = results_path.parent / f"{model_tags}_figures" / f"{fig_fname}"
        figs_path.parent.mkdir(parents=True, exist_ok=True)
        import plotly.io as pio   
        pio.kaleido.scope.mathjax = None # To remove MathJax box in pdf
        fig_parity_plot.write_image(f'{str(figs_path)}.pdf')
        fig_parity_plot.write_image(f'{str(figs_path)}.svg')
        fig_parity_plot.write_image(f'{str(figs_path)}.png', scale=10)
        fig_parity_plot.write_html(f'{str(figs_path)}.html')

    return results
