import logging
from functools import partial
from modnet.preprocessing import MODData
from pathlib import Path
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)

from shg_ml_benchmarks import run_benchmark
from shg_ml_benchmarks.utils import SHG_BENCHMARK_SPLITS

from train import train_fn, get_features


def predict_fn(
        model,
        structures,
        ids,
        n_jobs=2,
):

    # . Load the predictions
    df_pred_matten = pd.read_json("df_pred_matten_holdout.json.gz")


    # 3. Use the predict function of the EnsembleMODNet to predict the target
    return model.predict(md, return_unc=True)


for split in SHG_BENCHMARK_SPLITS:
    for incl_feat in ['mmf', 'pgnn', 'mmf_pgnn']:
        logging.info("Running benchmark %s for split %s", incl_feat, split)

        # 1. Load the model corresponding to this task (split)
            # - use train_fn?
        model = train_fn(
            ids = ['whatever'],
            structures = [None], # Not okay wrt. doc of the function
            targets = [0.0],
            name_target = ['whatever'],
            path_model = f"training/{split}/{incl_feat}/models/model_ensemble_modnet.pkl",
        )

        # TODO: maybe need to adapt run_benchmark for incl_feat tag somewhere?
        type_features = []
        if "mmf" in incl_feat:
            type_features.append("mm_fast")
        if "pgnn" in incl_feat:
            type_features.extend(["pgnn_mm", "pgnn_ofm", "pgnn_mvl32"])
        run_benchmark(
            model=model,
            predict_fn=partial(predict_fn, type_features=type_features),
            task=split,
            train_fn=None,
            model_label="modnet",
            model_tags=incl_feat,
            predict_individually=False,
        )
