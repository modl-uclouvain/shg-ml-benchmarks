import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

OUTLIERS = ["mp-622018", "mp-13032", "mp-28264", "mp-13150", "mp-604884", "mp-1227604"]

_DATA_PATH_DFLT = str(
    Path(os.path.dirname(os.path.realpath(__file__))).parent.parent
    / "data"
    / "df_rot_ieee_pmg.pkl.gz"
)

BENCHMARKS_DIR = Path(__file__).parent.parent.parent / "benchmarks"
RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
DATA_DIR = Path(__file__).parent.parent.parent / "data"
SHG_BENCHMARK_SPLITS = (
    "random_125",
    "random_250",
    "distribution_125",
    "distribution_250",
)


# from https://github.com/scikit-learn/scikit-learn/issues/4757#issuecomment-791644181
# Define the cross-validator object for regression, which inherits from
# StratifiedKFold, overwritting the split method
class StratifiedKFoldReg(StratifiedKFold):
    """

    This class generate cross-validation partitions
    for regression setups, such that these partitions
    resemble the original sample distribution of the
    target variable.

    """

    def split(
        self,
        y: pd.DataFrame | np.ndarray | list,
        target_to_bin=None,
        n_bins=None,
    ):
        n_samples = len(y)

        # Number of labels to discretize our target variable,
        # into bins of quasi equal size
        if not n_bins:
            n_bins = int(np.round(n_samples / self.n_splits))

        if target_to_bin:
            lim_bins = np.linspace(
                np.min(y[target_to_bin]), np.max(y[target_to_bin]), n_bins
            )
            y_labels = np.digitize(y[target_to_bin], lim_bins)
        else:
            lim_bins = np.linspace(np.min(y), np.max(y), n_bins)
            y_labels = np.digitize(y, lim_bins)

        return super().split(X=np.zeros(n_samples), y=y_labels)


def get_holdout_validation_set(
    data_path: str | Path = _DATA_PATH_DFLT,
    data: pd.DataFrame | list | np.ndarray = None,
    target_name: str = "dKP_full_neum",
    n_holdout: int = 100,
    include_validation: bool = False,
    strategy_holdout: str = "distribution",
    n_bins_distribution=None,
    random_seed: int = 42,
    shuffle: bool = True,
) -> tuple[list, list]:
    if "pkl" not in str(data_path):
        raise NameError(
            "data_path should be a pd.Dataframe in a pickle file (compressed or not)."
        )

    if not data:
        df = pd.read_pickle(data_path)
        df = df.query("is_unique_here == True")
        targets = df[target_name].tolist()
    elif isinstance(data, pd.DataFrame):
        df = data
        targets = df[target_name].tolist()
    elif isinstance(data, list) or isinstance(data, np.ndarray):
        df = None
        targets = data
    else:
        raise TypeError(
            f"data is of type {type(data)} instead of pd.Dataframe | list | np.ndarray"
        )

    print(f"{len(targets) = }")

    n_splits = len(df) // n_holdout

    if strategy_holdout == "distribution":
        skf = StratifiedKFoldReg(
            n_splits=n_splits, shuffle=shuffle, random_state=random_seed
        )
        split_gen = skf.split(y=targets, n_bins=n_bins_distribution)
    elif strategy_holdout == "random":
        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_seed)
        split_gen = kf.split(X=np.zeros(len(targets)), y=targets)
    else:
        raise ValueError(
            "strategy_holdout should either be 'distribution' or 'random'."
        )

    _, ind_holdout = next(split_gen)
    ind_holdout = ind_holdout[:n_holdout]
    if isinstance(df, pd.DataFrame):
        idx_holdout = df.iloc[ind_holdout].index.tolist()
    else:
        idx_holdout = list(ind_holdout)

    idx_validation = []
    if include_validation:
        _, ind_validation = next(split_gen)
        ind_validation = ind_validation[:n_holdout]
        if isinstance(df, pd.DataFrame):
            idx_validation = df.iloc[ind_validation].index.tolist()
        else:
            idx_validation = list(ind_validation)

    return idx_holdout, idx_validation


def load_holdout(task: str = "distribution_250") -> "pd.DataFrame":
    """Load holdout dataframe for a given task.

    Args:
        task (str): Task name

    Returns:
        The holdout dataframe mapping structure to target.
    """
    # Load holdout and validation indices
    holdout_path = DATA_DIR / f"holdout_id_{task}.json"
    with open(holdout_path) as f:
        holdout_ids = json.load(f)

    df = pd.read_pickle(_DATA_PATH_DFLT).loc[holdout_ids]
    # Remove outliers by ID
    return df.loc[~df.index.isin(OUTLIERS)]


def load_train(task: str = "distribution_250") -> "pd.DataFrame":
    """Load training set dataframe for a given task."""
    holdout_path = DATA_DIR / f"holdout_id_{task}.json"
    with open(holdout_path) as f:
        holdout_ids = json.load(f)

    val_path = DATA_DIR / f"validation_id_{task}.json"
    with open(val_path) as f:
        val_ids = json.load(f)

    full_df = pd.read_pickle(_DATA_PATH_DFLT)
    full_df = full_df.query("is_unique_here == True")
    return full_df.loc[~full_df.index.isin(holdout_ids + val_ids + OUTLIERS)]


def load_full(only_unique=True):
    full_df = pd.read_pickle(_DATA_PATH_DFLT)
    if not only_unique:
        return full_df
    return full_df.query("is_unique_here == True")


class DummyModel:
    """Simple baseline model that predicts averages of the training data."""

    from pymatgen.core import Structure

    label: str
    mode: str = "mean"
    meta: None = None

    def __init__(self, mode="mean"):
        self.mode = mode
        self.mean_value: float | None = None
        self.median_value: float | None = None
        self.label = "mean_value"
        if self.mode == "median":
            self.label = "median_value"

    def train(self, train_df: pd.DataFrame, target: str) -> "DummyModel":
        """Compute mean of training targets."""
        targets = train_df[target].values
        self.mean_value = np.mean(targets)  # type: ignore
        self.median_value = np.median(targets)  # type: ignore
        return self

    def predict(self, structure: Structure) -> float | np.ndarray:
        """Return mean value for all predictions."""
        if self.mode == "mean":
            value = self.mean_value
        elif self.mode == "median":
            value = self.median_value
        else:
            raise NotImplementedError(f"Mode {self.mode} not implemented.")

        if value is None:
            raise RuntimeError("Model must be trained before prediction")

        return value
