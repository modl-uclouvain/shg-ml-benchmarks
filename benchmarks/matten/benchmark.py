import logging
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd

logging.basicConfig(level=logging.INFO)

import shg_ml_benchmarks.utils_shg as shg
from shg_ml_benchmarks import run_benchmark
from shg_ml_benchmarks.utils import SHG_BENCHMARK_SPLITS


def predict_fn(
    model: Any,
    ids: list,
    path_pred: str | Path,
    structures: list | None = None,  # Needed for compatibility with the other models
):
    # Load the predictions
    df_pred_matten = pd.read_json(path_pred)
    df_pred_matten = df_pred_matten.filter(ids, axis=0)

    # Get the dKP from the tensor predictions into a single df
    df_pred = pd.DataFrame(
        index=ids,
        data=[
            shg.get_dKP(dijk_matten) for dijk_matten in df_pred_matten["dijk_matten"]
        ],
        columns=["dKP_full_neum"],
    )

    return df_pred


# Just to have the model.label accessible
class Object:
    pass


model = Object()
model.label = "matten"

for split in SHG_BENCHMARK_SPLITS:
    logging.info("Running benchmark for split %s", split)

    path_pred = "./training/predict_" + split + "/df_pred_matten_holdout.json.gz"

    run_benchmark(
        model=model,
        predict_fn=partial(predict_fn, path_pred=path_pred),
        task=split,
        train_fn=None,
        predict_individually=False,
    )
