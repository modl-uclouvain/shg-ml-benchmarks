import logging
from functools import partial

import numpy as np
from modnet.preprocessing import MODData

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

    # 2. Create the MODData to predict from df_featurized
    md = MODData(
        materials=structures,
        targets=[0.0] * len(structures),
        target_names=["dKP_full_neum"],
        structure_ids=ids,
    )
    md.df_featurized = df_featurized.filter(ids, axis=0)
    # 3. Use the predict function of the EnsembleMODNet to predict the target
    return model.predict(md, return_unc=True)


for split in SHG_BENCHMARK_SPLITS:
    for incl_feat in ["mmf", "pgnn", "mmf_pgnn"]:
        logging.info("Running benchmark %s for split %s", incl_feat, split)

        task_feat = "training/" + split + "/" + incl_feat

        # 1. Load the model corresponding to this task (split)
        # - use train_fn?
        model = train_fn(
            ids=["whatever"],
            structures=[None],  # Not okay wrt. doc of the function
            targets=[0.0],
            name_target=["whatever"],
            path_model=f"training/{split}/{incl_feat}/models/model_ensemble_modnet.pkl",
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
            model_label="modnet_nan",
            model_tags=incl_feat,
            predict_individually=False,
        )
