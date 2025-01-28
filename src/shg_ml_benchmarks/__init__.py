from typing import Any
from collections.abc import Callable
import json
import numpy as np
from pymatgen.core import Structure
import importlib.metadata

__version__ = importlib.metadata.version("shg-ml-benchmarks")


class DummyModel:
    """Simple baseline model that predicts the mean of training data."""

    def __init__(self):
        self.mean_value: float

    def train(
        self, structures: list[Structure], targets: dict[str, float | np.ndarray]
    ) -> None:
        """Compute mean of training targets."""
        values = list(targets.values())
        self.mean_value = np.mean(values)  # type: ignore

    def predict(self, structure: Structure) -> float | np.ndarray:
        """Return mean value for all predictions."""
        if self.mean_value is None:
            raise RuntimeError("Model must be trained before prediction")
        return self.mean_value


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
    predictions: dict[str, float | np.ndarray], test_data: dict[str, dict]
) -> dict[str, float]:
    """Calculate evaluation metrics.

    Args:
        predictions: Dictionary mapping structure IDs to predictions
        test_data: Test dataset dictionary

    Returns:
        Dictionary with evaluation metrics
    """
    true_values = [test_data[id]["target"] for id in predictions.keys()]
    pred_values = list(predictions.values())

    # Calculate metrics
    mae = np.mean(np.abs(np.array(true_values) - np.array(pred_values)))
    rmse = np.sqrt(np.mean((np.array(true_values) - np.array(pred_values)) ** 2))

    return {"mae": float(mae), "rmse": float(rmse)}


def run_benchmark(
    model: Any,
    train_fn: Callable[[list[Structure], dict[str, Any]], Any],
    predict_fn: Callable[[Any, Structure], float | np.ndarray],
    data_path: str,
    holdout_path: str,
    output_path: str = "benchmark_results.json",
) -> dict:
    """Run benchmark using provided training and prediction functions.

    Args:
        train_fn: Function that takes (structures, targets) and returns model
        predict_fn: Function that takes (model, structure) and returns prediction
        data_path: Path to data JSON
        holdout_path: Path to holdout IDs JSON
        output_path: Where to save results

    Returns:
        Dictionary with benchmark results and metrics
    """
    # Load data
    train_data, test_data = load_and_split_data(data_path, holdout_path)

    # Prepare training data
    train_structures = [entry["structure"] for entry in train_data.values()]
    train_targets = {id: entry["target"] for id, entry in train_data.items()}

    # Train model
    model = train_fn(train_structures, train_targets)

    # Get predictions
    predictions = {}
    for structure_id, entry in test_data.items():
        pred = predict_fn(model, entry["structure"])
        # Convert numpy types to Python native types for JSON serialization
        if isinstance(pred, np.ndarray):
            pred = pred.tolist()
        elif isinstance(pred, np.generic):
            pred = pred.item()
        predictions[structure_id] = pred

    # Calculate metrics
    metrics = evaluate_predictions(predictions, test_data)

    # Compile results
    results = {"predictions": predictions, "metrics": metrics}

    # Save results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    # Example usage
    DATA_PATH = "path/to/data.json"
    HOLDOUT_PATH = "path/to/holdout.json"

    dummy = DummyModel()

    def train_fn(structures, targets):
        dummy.train(structures, targets)
        return dummy

    def predict_fn(model, structure):
        return model.predict(structure)

    results = run_benchmark(
        model=dummy,
        train_fn=train_fn,
        predict_fn=predict_fn,
        data_path=DATA_PATH,
        holdout_path=HOLDOUT_PATH,
    )

    print(f"Model metrics: {results['metrics']}")
