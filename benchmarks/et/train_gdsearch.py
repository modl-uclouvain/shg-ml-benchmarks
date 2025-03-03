"Contains functions to featurize given structures and perform a Gridsearch on the hparams of ExtraTreesRegressor on a given training and validation set."

import json
import logging
import os
import shutil
from os import walk
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from modnet.featurizers.presets.matminer_2024_fast import Matminer2024FastFeaturizer
from modnet.preprocessing import MODData
from pgnn.featurizers.structure import (
    l_MM_v1,
    l_OFM_v1,
    mvl32,
)
from pymatgen.core.structure import Structure
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import ParameterGrid

from shg_ml_benchmarks.analysis import evaluate_predictions
from shg_ml_benchmarks.utils import (
    SHG_BENCHMARK_SPLITS,
    load_full,
    load_holdout,
    load_train,
)

logging.basicConfig(level=logging.INFO)


def check_need_to_featurize(
    path_df: str | Path,
    ids: list[str],
) -> bool:
    if Path(path_df).exists():
        logging.info(f"Loading '{path_df}'...")
        index_feat = pd.read_csv(path_df, index_col=[0]).index.tolist()
        if len(index_feat) == len(ids) and all([i in index_feat for i in ids]):
            logging.info("Refeaturizing is not necessary.")
            return False
        logging.info("Refeaturizing is necessary.")
        print(f"{[i in index_feat for i in ids] = }")
        print(f"{ids = }")
        print(f"{index_feat = }")
    return True


# Copied from MODNet benchmark folder
def get_features(
    ids: list[str],
    structures: list[Structure] | None = None,
    type_features: list[str] | None = ["mm_fast", "pgnn_mm", "pgnn_ofm", "pgnn_mvl32"],
    path_features: str | Path | None = None,
    features: pd.DataFrame | None = None,
    n_jobs: int = 2,
    path_saved: None | str | Path = "features/df_featurized_final.csv.gz",
    path_dir_tmp="tmp_df_feat",
    remove_dir_indiv_feat=True,
    drop_nans=True,
) -> pd.DataFrame:
    if path_saved and Path(path_saved).exists():
        logging.info(f"Loading the final features at '{path_saved}'...")
        df_featurized = pd.read_csv(path_saved, index_col=[0])
        index_feat = df_featurized.index.tolist()
        if len(index_feat) == len(ids) and all(
            [True if i in index_feat else False for i in ids]
        ):
            logging.info("Refeaturizing is not necessary.")
            return df_featurized
        else:
            raise Exception(
                "The provided ids and the index of the loaded final features do not match... Refeaturizing might be necessary."
            )

    # Check arguments
    if not type_features and not features:
        raise ValueError(
            "At least one of 'type_features' or 'features' must be different than 'None'."
        )

    # Featurization needed
    if type_features:
        # Create a tmp dir to save the sets of features individually in case of restart
        os.makedirs(name=path_dir_tmp, exist_ok=True)

        # Check the different types of features required

        # Fast Matminer
        if "mm_fast" in type_features:
            logging.info("Considering the 'mm_fast' features...")
            path_df = f"{path_dir_tmp}/df_mm_fast.csv.gz"
            need_to_featurize = check_need_to_featurize(path_df, ids)

            if need_to_featurize:
                logging.info("Featurizing with Matminer2024FastFeaturizer...")
                md = MODData(
                    materials=structures,
                    targets=[None] * len(structures),
                    target_names=["dKP"],
                    structure_ids=ids,
                )
                md.featurizer = Matminer2024FastFeaturizer()
                md.featurize(n_jobs=n_jobs)
                logging.info(
                    f"Matminer2024FastFeaturizer features shape: {md.df_featurized.shape}"
                )
                md.df_featurized.to_csv(path_df)
                logging.info(
                    f"Matminer2024FastFeaturizer features have been saved at {path_df}"
                )

        if "pgnn_mm" in type_features:
            logging.info("Considering the 'pgnn_mm' features...")
            path_df = f"{path_dir_tmp}/df_pgnn_mm.csv.gz"
            need_to_featurize = check_need_to_featurize(path_df, ids)

            if need_to_featurize:
                logging.info("Featurizing with pGNN l_MM_v1...")
                df_feat = l_MM_v1.get_features(
                    pd.DataFrame(index=ids, data={"structure": structures})["structure"]
                )
                df_feat = df_feat.drop(["structure"], axis=1, errors="ignore")
                logging.info(f"l_MM_v1 features shape: {df_feat.shape}")
                df_feat.to_csv(path_df)
                logging.info(f"pGNN l_MM_v1 features have been saved at {path_df}")

        if "pgnn_ofm" in type_features:
            logging.info("Considering the 'pgnn_ofm' features...")
            path_df = f"{path_dir_tmp}/df_pgnn_ofm.csv.gz"
            need_to_featurize = check_need_to_featurize(path_df, ids)

            if need_to_featurize:
                logging.info("Featurizing with pGNN l_OFM_v1...")
                df_feat = l_OFM_v1.get_features(
                    pd.DataFrame(index=ids, data={"structure": structures})["structure"]
                )
                df_feat = df_feat.drop(["structure"], axis=1, errors="ignore")
                logging.info(f"l_OFM_v1 features shape: {df_feat.shape}")
                df_feat.to_csv(path_df)
                logging.info(f"pGNN l_OFM_v1 features have been saved at {path_df}")

        if "pgnn_mvl32" in type_features:
            logging.info("Considering the 'pgnn_mvl32' features...")
            path_df = f"{path_dir_tmp}/df_pgnn_mvl32.csv.gz"
            need_to_featurize = check_need_to_featurize(path_df, ids)

            if need_to_featurize:
                logging.info("Featurizing with pGNN mvl32...")
                df_feat = mvl32.get_features(
                    pd.DataFrame(index=ids, data={"structure": structures})["structure"]
                )
                df_feat = df_feat.drop(["structure"], axis=1, errors="ignore")
                logging.info(f"mvl32 features shape: {df_feat.shape}")
                df_feat.to_csv(path_df)
                logging.info(f"pGNN mvl32 features have been saved at {path_df}")

    # Consider the custom features
    df_featurized = pd.DataFrame()
    if path_features:
        df_featurized = pd.read_csv(path_features, index_col=[0])
        if not all(df_featurized.index.tolist() == ids):
            raise ValueError(
                f"The index of the features at {path_features} does not match the materials ids provided. It might be worth verifying your data."
            )
    if features:
        if not all(features.index.tolist() == ids):
            raise ValueError(
                "The index of 'features' does not match the materials ids provided. It might be worth verifying your data."
            )
        df_featurized = pd.concat([df_featurized, features], axis=1)

    # Concatenate all the features
    if Path(path_dir_tmp).exists():
        filenames = next(walk(path_dir_tmp), (None, None, []))[2]
        for f in filenames:
            df_featurized = pd.concat(
                [df_featurized, pd.read_csv(f"{path_dir_tmp}/{f}", index_col=[0])],
                axis=1,
            )

    # Remove columns with more than 50% NaN values and replace the NaN values by the mean of the rest
    if drop_nans:
        logging.info(
            f"WARNING: Before dropping columns with >=50% NaNs, {df_featurized.shape = }"
        )
        cols_with_nan = df_featurized.columns[df_featurized.isnull().any()].tolist()
        cols_to_remove = []
        for c in cols_with_nan:
            if df_featurized[c].isnull().sum() / len(df_featurized) >= 0.5:
                cols_to_remove.append(c)
                continue
            for v in df_featurized[c]:
                if np.isnan(v):
                    continue
                if int(v) != v:
                    df_featurized[c].fillna(df_featurized[c].mean(), inplace=True)
                    break
            else:
                df_featurized[c].fillna(round(df_featurized[c].mean()), inplace=True)
        df_featurized = df_featurized.drop(cols_to_remove, axis=1)
        logging.info(f"WARNING: In the end, {df_featurized.shape = }")
        logging.info(
            "WARNING: The other NaN values have been replaced by the mean of the values accross the corresponding column."
        )

    if path_saved:
        os.makedirs(Path(path_saved).parents[0], exist_ok=True)
        df_featurized.to_csv(path_saved)

    if remove_dir_indiv_feat:
        shutil.rmtree(path_dir_tmp)

    return df_featurized


def train_fn(
    targets: list[float],
    df_featurized: pd.DataFrame | None = None,
    n_jobs: int = 2,
    path_model: str | Path = "models/model_et.pkl",
    save_model: bool = True,
    do_training: bool = False,
    kwargs_model: dict = {},
) -> ExtraTreesRegressor | None:
    if Path(path_model).exists():
        logging.info(f"The model already exists at {path_model}.")
        logging.info("Loading the model...")
        model = joblib.load(path_model)
        logging.info("ExtraTreesRegressor model loaded.")
    else:
        if do_training:
            logging.info("Training the model...")
            model = ExtraTreesRegressor(**kwargs_model).fit(
                df_featurized.to_numpy(), targets
            )
            logging.info("Training done.")
            os.makedirs(Path(path_model).parents[0], exist_ok=True)

            # save
            if save_model:
                joblib.dump(model, path_model)
                logging.info(f"The model has been saved at {path_model}.")
        else:
            logging.info("The model has not been trained.")
            model = None

    return model


def replace_nans(df_tbr, df_tr):
    df_tbr_tmp = df_tbr.copy()
    cols_with_nan = df_tbr_tmp.columns[df_tbr_tmp.isnull().any()].tolist()
    for c in cols_with_nan:
        if c in df_tr.columns.tolist():
            for v in df_tr[c]:
                if np.isnan(v):
                    continue
                if int(v) != v:
                    df_tbr_tmp[c].fillna(df_tr[c].mean(), inplace=True)
                    break
            else:
                df_tbr_tmp[c].fillna(round(df_tr[c].mean()), inplace=True)
        else:
            df_tbr_tmp = df_tbr_tmp.drop([c], axis=1)
    logging.info("The NaNs have been replaced.")
    return df_tbr_tmp


def main():
    n_jobs = 4

    end_model_name = ""

    # ====================================================================
    for task in SHG_BENCHMARK_SPLITS:
        # if task != "distribution_125": # TODO: TO REMOVE
        #     continue
        print(f"Gridsearching for task {task}")

        # --------------------------------------------------------------------
        # To specify which features are considered
        for type_features, name_feat in zip(
            [
                ["mm_fast"],
                ["mm_fast", "pgnn_mm", "pgnn_ofm", "pgnn_mvl32"],
                ["pgnn_mm", "pgnn_ofm", "pgnn_mvl32"],
            ],
            ["mmf", "mmf_pgnn", "pgnn"],
        ):
            # if name_feat!="mmf": # TODO: TO REMOVE
            #     continue

            print(f"Gridsearching for features {name_feat}")

            task_feat = "training/" + task + f"/{name_feat}"

            # Loading the features for the current task

            # Load training data
            df_train = load_train(task)
            ids = df_train.index.tolist()
            structures = [Structure.from_dict(s) for s in df_train["structure_rot"]]
            name_target = "dKP_full_neum"
            targets = df_train[name_target]

            df_featurized = get_features(
                ids=ids,
                structures=structures,
                type_features=type_features,
                path_features=None,
                features=None,
                n_jobs=n_jobs,
                path_saved=f"{task_feat}/features/df_featurized_train.csv.gz",
                path_dir_tmp=f"{task_feat}/tmp_df_feat_train",
                remove_dir_indiv_feat=True,
                drop_nans=True,
            )
            df_featurized = df_featurized.filter(ids, axis=0)  # just to be sure

            # Load validation data
            df_val = load_full()
            df_val = df_val.drop(df_train.index, axis=0)
            df_val = df_val.drop(load_holdout(task).index, axis=0)
            ids_val = df_val.index.tolist()
            structures_val = [Structure.from_dict(s) for s in df_val["structure_rot"]]

            df_featurized_val = get_features(
                ids=ids_val,
                structures=structures_val,
                type_features=type_features,
                path_features=None,
                features=None,
                n_jobs=n_jobs,
                path_saved=f"{task_feat}/features/df_featurized_val.csv.gz",
                path_dir_tmp=f"{task_feat}/tmp_df_feat_val",
                remove_dir_indiv_feat=True,
                drop_nans=False,
            )
            df_featurized_val = replace_nans(df_featurized_val, df_featurized)

            continue  # TODO: TO REMOVE

            # Defining the hparams of the model and where to store the results
            param_grid = {
                "max_depth": list(range(4, 11, 2)),
                "min_samples_split": list(
                    np.linspace(0.0001, 0.5, 100, endpoint=False)
                ),
                "min_samples_leaf": list(np.linspace(0.0001, 0.5, 100, endpoint=False)),
                "n_estimators": list(range(100, 501, 50)),
                "random_state": [42],
            }

            path_results_gdsearch = f"{task_feat}/gridsearch/results_gdsearch.json"
            os.makedirs(name=Path(path_results_gdsearch).parent, exist_ok=True)
            with open(path_results_gdsearch, "w") as file:
                file.write("{")

            # Iterating over each combination of hparams
            for i, kwargs_model in enumerate(ParameterGrid(param_grid)):
                igdsearch = i
                with open(path_results_gdsearch, "a") as file:
                    file.write(f'"{igdsearch}": ')

                print(f"Gridsearching at combination {igdsearch}")

                results_gdsearch_unit = {}
                results_gdsearch_unit["hparams"] = kwargs_model

                model = train_fn(
                    targets=targets,
                    df_featurized=df_featurized,
                    n_jobs=n_jobs,
                    path_model=f"{task_feat}/models/model_et{end_model_name}.pkl",
                    do_training=True,
                    save_model=False,
                    kwargs_model=kwargs_model,
                )

                logging.info("Predicting the validation targets.")
                preds = model.predict(df_featurized_val.to_numpy())
                logging.info("The predictions have been computed.")
                predictions = {k: v for k, v in zip(ids_val, preds)}
                metric = evaluate_predictions(predictions, df_val, name_target)

                results_gdsearch_unit.update(metric)

                with open(path_results_gdsearch, "a") as file:
                    file.write(json.dumps(results_gdsearch_unit, indent=4))
                    file.write(",\n")

            with open(path_results_gdsearch, "a") as file:
                file.write("}")


if __name__ == "__main__":
    main()
