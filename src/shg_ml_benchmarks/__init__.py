import importlib.metadata
import json
import logging
from collections.abc import Callable
from typing import Any

import numpy as np
from pymatgen.core import Structure

from shg_ml_benchmarks.analysis import evaluate_predictions, visualize_predictions
from shg_ml_benchmarks.utils import BENCHMARKS_DIR, load_holdout, load_train

logging.basicConfig(level=logging.INFO)

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


def run_benchmark(
    model: Any,
    predict_fn: Callable,
    train_fn: Callable | None = None,
    task: str = "random_250",
    target: str = "dKP_full_neum",
    write_results: bool = True,
    model_label: str | None = None,
    model_tags: str | None = None,
    predict_individually: bool = True,
    tasks_tag: str = "",
) -> dict | None:
    """Run benchmark using provided training and prediction functions.

    Args:
        model: Model object
        predict_fn: Function that takes (model, structure) and returns prediction
        train_fn: Function that takes (structures, target) and returns model (optional)
        task: the task to run; corresponds to the filenames of pre-defined holdout sets found in `./data`.
        target: the target property to predict
        write_results: whether to write the results to disk
        predict_individually: whether to predict each structure individually or batch predict,

    Returns:
        Dictionary with benchmark results and metrics
    """
    if write_results:
        # Check if the benchmark has already been run
        if getattr(model, "tags", None) is not None:
            results_fname = f"{model.tags}_results.json"
            model_tags = model.tags
        elif model_tags is not None:
            results_fname = f"{model_tags}_results.json"
        else:
            results_fname = "results.json"
        if getattr(model, "label", None) is not None:
            results_path = (
                BENCHMARKS_DIR
                / model.label
                / f"tasks{tasks_tag}"
                / task
                / results_fname
            )
        elif model_label:
            results_path = (
                BENCHMARKS_DIR
                / model_label
                / f"tasks{tasks_tag}"
                / task
                / results_fname
            )

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
    predictions: dict[str, float] = {}
    uncertainties: dict[str, float] = {}
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
            df_pred, df_unc = predict_fn(
                model,
                structures=[Structure.from_dict(s) for s in holdout_df["structure"]],
                ids=holdout_df.index.tolist(),
            )
            uncertainties = df_unc[df_unc.columns[0]].to_dict()
        except ValueError:
            df_pred = predict_fn(
                model,
                structures=[Structure.from_dict(s) for s in holdout_df["structure"]],
                ids=holdout_df.index.tolist(),
            )
            df_unc = None
        predictions = df_pred[df_pred.columns[0]].to_dict()

    # Calculate metrics
    metrics = evaluate_predictions(predictions, holdout_df, target)

    # Compile results
    if not uncertainties:
        uncertainties = None  # type: ignore

    results = {
        "predictions": predictions,
        "uncertainties": uncertainties,
        "metrics": metrics,
    }
    if getattr(model, "meta", None) is not None:
        results["meta"] = model.meta

    if write_results:
        results_path.parent.mkdir(parents=True, exist_ok=True)

        # Save results
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        # Save figure
        figs_path = results_path.parent / f"{model_tags}_figures"
        visualize_predictions(predictions, holdout_df, target, path=figs_path)

        logging.info(f"The results have been saved at {results_path}.")

    return results
