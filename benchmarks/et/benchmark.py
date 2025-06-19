"Contains functions to predict a given holdout set."

import logging
from functools import partial

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)

from train import get_features, train_fn

from shg_ml_benchmarks import run_benchmark
from shg_ml_benchmarks.utils import SHG_BENCHMARK_SPLITS, load_full, load_holdout


def predict_fn(
    model,
    structures,
    ids,
    task,
    task_feat,
    type_features=["mm_fast", "pgnn_mm", "pgnn_ofm", "pgnn_mvl32"],
    features=None,
    n_jobs=2,
):
    # 1. Featurize the structure with the same features as the one the model was trained on (consider that an input of the function?)
    # - Use get_features()?

    df_featurized = get_features(
        ids=ids,
        structures=structures,
        type_features=type_features,
        features=features,
        n_jobs=n_jobs,
        path_saved=None,
        remove_dir_indiv_feat=True,
        drop_nans=False,
    )
    # 1.b Padd the missing features by zero
    # TODO ?

    # 1.c Replace the NaN values by the mean of the corresponding feature in the training dataset
    # Load training + validation data
    path_training_features = f"{task_feat}/features/df_featurized_final.csv.gz"
    logging.info(
        f"Loading the training features at {path_training_features} to replace NaNs by the mean of the training set for relevant features."
    )
    df_train = load_full()
    df_train = df_train.drop(load_holdout(task).index, axis=0)
    df_train_featurized = get_features(
        ids=df_train.index.tolist(),
        structures=[None],
        path_saved=path_training_features,
    )
    cols_with_nan = df_featurized.columns[df_featurized.isnull().any()].tolist()
    for c in cols_with_nan:
        if c in df_train_featurized.columns.tolist():
            for v in df_train_featurized[c]:
                if np.isnan(v):
                    continue
                if int(v) != v:
                    df_featurized[c].fillna(df_train_featurized[c].mean(), inplace=True)
                    break
            else:
                df_featurized[c].fillna(
                    round(df_train_featurized[c].mean()), inplace=True
                )
        else:
            df_featurized = df_featurized.drop([c], axis=1)
    logging.info("The NaNs have been replaced.")

    # 2. Use the predict function to predict the target
    logging.info("Predicting the target.")
    preds = model.predict(df_featurized.to_numpy())
    logging.info("The predictions have been computed.")
    return pd.DataFrame(index=ids, data=preds)


# Needed to distinguish between two different hparams optimization for example, just "" if not needed
# Also corresponds to the end of the model file name
# tasks_tag = "_An"
tasks_tag = "_opti"

for split in SHG_BENCHMARK_SPLITS:
    for incl_feat in ["mmf", "pgnn", "mmf_pgnn"]:
        logging.info("Running benchmark %s for split %s", incl_feat, split)

        # 1. Load the model corresponding to this task (split)
        # - use train_fn?
        task_feat = "training/" + split + "/" + incl_feat
        model = train_fn(
            targets=[0.0],
            path_model=f"{task_feat}/models/model_et{tasks_tag}.pkl",
        )

        # TODO: maybe need to adapt run_benchmark for incl_feat tag somewhere?
        type_features = []
        if "mmf" in incl_feat:
            type_features.append("mm_fast")
        if "pgnn" in incl_feat:
            type_features.extend(["pgnn_mm", "pgnn_ofm", "pgnn_mvl32"])

        run_benchmark(
            model=model,
            predict_fn=partial(
                predict_fn, type_features=type_features, task=split, task_feat=task_feat
            ),
            task=split,
            train_fn=None,
            model_label="et",
            model_tags=incl_feat,
            predict_individually=False,
            tasks_tag=tasks_tag,
        )
