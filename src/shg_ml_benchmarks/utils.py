import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

_DATA_PATH_DFLT = str(
    Path(os.path.dirname(os.path.realpath(__file__))).parent.parent
    / "data"
    / "df_rot_ieee_pmg.pkl.gz"
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


def get_holdout_set(
    data_path: str | Path = _DATA_PATH_DFLT,
    data: pd.DataFrame | list | np.ndarray = None,
    target_name: str = "dKP_full_neum",
    n_holdout: int = 100,
    strategy_holdout: str = "distribution",
    n_bins_distribution=None,
    random_seed: int = 42,
    shuffle: bool = True,
) -> list:
    if "pkl" not in str(data_path):
        raise NameError(
            "data_path should be a pd.Dataframe in a pickle file (compressed or not)."
        )

    if not data:
        df = pd.read_pickle(data_path)
        targets = df[target_name].tolist()
    elif isinstance(data, pd.DataFrame):
        df = data
        targets = df[target_name].tolist()
    elif isinstance(data, list) or isinstance(data, np.ndarray):
        df = None
        targets = data
    else:
        raise TypeError(
            f"data is of type {type(data)} instead pf pd.Dataframe | list | np.ndarray"
        )

    n_splits = len(df) // n_holdout

    if strategy_holdout == "distribution":
        skf = StratifiedKFoldReg(
            n_splits=n_splits, shuffle=shuffle, random_state=random_seed
        )
        _, ind_holdout = next(skf.split(y=targets, n_bins=n_bins_distribution))
    elif strategy_holdout == "random":
        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_seed)
        _, ind_holdout = next(kf.split(X=np.zeros(len(targets)), y=targets))
    else:
        raise ValueError(
            "strategy_holdout should either be 'distribution' or 'random'."
        )

    ind_holdout = ind_holdout[:n_holdout]

    if isinstance(df, pd.DataFrame):
        return df.iloc[ind_holdout].index.tolist()
    else:
        return list(ind_holdout)
