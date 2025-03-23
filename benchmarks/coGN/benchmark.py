import logging
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd

logging.basicConfig(level=logging.INFO)

from shg_ml_benchmarks import run_benchmark
from shg_ml_benchmarks.utils import SHG_BENCHMARK_SPLITS


def predict_fn(
    model: Any,
    ids: list,
    path_pred: str | Path,
    structures: list | None = None,  # Needed for compatibility with the other models
):
    # Load the predictions
    df_pred = pd.read_json(path_pred)
    check_len = len(df_pred)
    df_pred = df_pred.filter(ids, axis=0)
    assert len(df_pred) == check_len, (
        "The size of the predictions list was modified by the filtering on the ids given."
    )
    df_pred = df_pred.rename(columns={"predictions": "dKP_full_neum"})

    return df_pred


# Just to have the model.label accessible
class Object:
    pass


model = Object()
model.label = "coGN"

for split in SHG_BENCHMARK_SPLITS:
    logging.info("Running benchmark for split %s", split)

    path_pred = "./training/" + split + f"/hparams_matbench/results_{split}.json.gz"

    run_benchmark(
        model=model,
        predict_fn=partial(predict_fn, path_pred=path_pred),
        task=split,
        train_fn=None,
        predict_individually=False,
    )
