# TODO:
# V take a list of Structures and their ids as input
# V featurize them
#       V with the fast Matminer (including polarization?)
#       V pgnn
#       V accept additional custom features
#       - include possible prediction of other properties such as n and treat it as a feature?
# V train and save the model in a given location

import os
import shutil
from copy import deepcopy
from os import walk
from pathlib import Path

import pandas as pd
from modnet.featurizers.presets.matminer_2024_fast import Matminer2024FastFeaturizer
from modnet.hyper_opt import FitGenetic
from modnet.models import EnsembleMODNetModel
from modnet.preprocessing import MODData
from pgnn.featurizers.structure import (
    l_MM_v1,
    l_OFM_v1,
    mvl32,
)
from pymatgen.core.structure import Structure

from shg_ml_benchmarks.utils import load_full, load_holdout


def check_need_to_featurize(
    path_df: str | Path,
    ids: list[str],
) -> bool:
    if Path(path_df).exists():
        print(f"Loading '{path_df}'...")
        index_feat = pd.read_csv(path_df, index_col=[0]).index.tolist()
        if len(index_feat) == len(ids) and all([i in index_feat for i in ids]):
            print("Refeaturizing is not necessary.")
            return False
        print("Refeaturizing is necessary.")
        print(f"{[i in index_feat for i in ids] = }")
        print(f"{ids = }")
        print(f"{index_feat = }")
    return True


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
) -> pd.DataFrame:
    if path_saved and Path(path_saved).exists():
        print(f"Loading the final features at '{path_saved}'...")
        df_featurized = pd.read_csv(path_saved, index_col=[0])
        index_feat = df_featurized.index.tolist()
        if len(index_feat) == len(ids) and all(
            [True if i in index_feat else False for i in ids]
        ):
            print("Refeaturizing is not necessary.")
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
            print("Considering the 'mm_fast' features...")
            path_df = f"{path_dir_tmp}/df_mm_fast.csv.gz"
            need_to_featurize = check_need_to_featurize(path_df, ids)

            if need_to_featurize:
                print("Featurizing with Matminer2024FastFeaturizer...")
                md = MODData(
                    materials=structures,
                    targets=[None] * len(structures),
                    target_names=["dKP"],
                    structure_ids=ids,
                )
                md.featurizer = Matminer2024FastFeaturizer()
                md.featurize(n_jobs=n_jobs)
                print(
                    "Matminer2024FastFeaturizer features shape:", md.df_featurized.shape
                )
                md.df_featurized.to_csv(path_df)
                print(
                    f"Matminer2024FastFeaturizer features have been saved at {path_df}"
                )

        if "pgnn_mm" in type_features:
            print("Considering the 'pgnn_mm' features...")
            path_df = f"{path_dir_tmp}/df_pgnn_mm.csv.gz"
            need_to_featurize = check_need_to_featurize(path_df, ids)

            if need_to_featurize:
                print("Featurizing with pGNN l_MM_v1...")
                df_feat = l_MM_v1.get_features(
                    pd.DataFrame(index=ids, data={"structure": structures})["structure"]
                )
                df_feat = df_feat.drop(["structure"], axis=1, errors="ignore")
                print("l_MM_v1 features shape:", df_feat.shape)
                df_feat.to_csv(path_df)
                print(f"pGNN l_MM_v1 features have been saved at {path_df}")

        if "pgnn_ofm" in type_features:
            print("Considering the 'pgnn_ofm' features...")
            path_df = f"{path_dir_tmp}/df_pgnn_ofm.csv.gz"
            need_to_featurize = check_need_to_featurize(path_df, ids)

            if need_to_featurize:
                print("Featurizing with pGNN l_OFM_v1...")
                df_feat = l_OFM_v1.get_features(
                    pd.DataFrame(index=ids, data={"structure": structures})["structure"]
                )
                df_feat = df_feat.drop(["structure"], axis=1, errors="ignore")
                print("l_OFM_v1 features shape:", df_feat.shape)
                df_feat.to_csv(path_df)
                print(f"pGNN l_OFM_v1 features have been saved at {path_df}")

        if "pgnn_mvl32" in type_features:
            print("Considering the 'pgnn_mvl32' features...")
            path_df = f"{path_dir_tmp}/df_pgnn_mvl32.csv.gz"
            need_to_featurize = check_need_to_featurize(path_df, ids)

            if need_to_featurize:
                print("Featurizing with pGNN mvl32...")
                df_feat = mvl32.get_features(
                    pd.DataFrame(index=ids, data={"structure": structures})["structure"]
                )
                df_feat = df_feat.drop(["structure"], axis=1, errors="ignore")
                print("mvl32 features shape:", df_feat.shape)
                df_feat.to_csv(path_df)
                print(f"pGNN mvl32 features have been saved at {path_df}")

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

    if path_saved:
        os.makedirs(Path(path_saved).parents[0], exist_ok=True)
        df_featurized.to_csv(path_saved)

    if remove_dir_indiv_feat:
        shutil.rmtree(path_dir_tmp)

    return df_featurized


def train_fn(
    ids: list[str],
    structures: list[Structure],
    targets: list[float],
    name_target: str,
    path_moddata: str | Path | None = "moddata/mod.data_training",
    df_featurized: pd.DataFrame | None = None,
    moddata: MODData | None = None,
    do_feature_selection: bool = True,
    save_moddata: bool = True,
    n_jobs: int = 2,
    path_model: str | Path = "models/model_ensemble_modnet.pkl",
    do_training: bool = False,
) -> EnsembleMODNetModel | None:
    if Path(path_model).exists():
        print(f"The model already exists at {path_model}.")
        print("Loading the model...")
        model = EnsembleMODNetModel.load(path_model)
        print("EnsembleMODNet model loaded.")
    else:
        # Build the MODData that will be used for training
        print("Creating the MODData...")
        if path_moddata is not None and Path(path_moddata).exists():
            print(f"The MODData has been loaded from {path_moddata}.")
            md = MODData.load(path_moddata)
        elif df_featurized is None and moddata is not None:
            md = deepcopy(moddata)
        elif df_featurized is not None and moddata is None:
            md = MODData(
                materials=structures,
                targets=targets,
                target_names=[name_target],
                structure_ids=ids,
            )
            md.df_featurized = df_featurized
        else:
            raise ValueError(
                "If 'path_moddata' does not exist or is None, then either 'df_featurized' or 'moddata' must be provided."
            )
        print("The MODData has been created.")

        if do_feature_selection:
            print("Beginning feature selection...")
            md.cross_nmi = None
            md.feature_selection(
                n=-1,
                n_samples=len(md.df_featurized) + 1,
                n_jobs=n_jobs,
            )
            print("Feature selection is now done.")
            path_moddata_hasfeatselect = str(path_moddata) + "_featselect"
        else:
            path_moddata_hasfeatselect = str(path_moddata) + "_notfeatselect"

        if save_moddata and not Path(path_moddata_hasfeatselect).exists():
            os.makedirs(Path(path_moddata_hasfeatselect).parents[0], exist_ok=True)
            md.save(path_moddata_hasfeatselect)
            print(f"The MODData has been saved at {path_moddata_hasfeatselect}.")

        if do_training:
            # Use FitGenetic to optimize the hyperparameters and train the model
            print("Training the model...")
            ga = FitGenetic(md, sample_threshold=len(md.df_featurized) + 1)
            model = ga.run(
                size_pop=20,  # dflt = 20
                num_generations=10,  # dflt = 10
                nested=5,  # dflt = 5
                n_jobs=n_jobs,
                early_stopping=6,  # dflt = 4
                refit=0,  # dflt = 5
            )
            print("Training done.")
            os.makedirs(Path(path_model).parents[0], exist_ok=True)
            model.save(path_model)
            print(f"The model has been saved at {path_model}.")
        else:
            print("The model has not been trained.")
            model = None

    return model


def main():
    n_jobs = 4
    do_feature_selection = False

    # ====================================================================
    task = "distribution_125"

    # Load training + validation data
    df_train = load_full()
    df_train = df_train.drop(load_holdout(task).index, axis=0)

    ids = df_train.index.tolist()
    structures = [Structure.from_dict(s) for s in df_train["structure_rot"]]
    name_target = "dKP_full_neum"
    targets = df_train[name_target]

    # --------------------------------------------------------------------
    # To specify which features are considered
    task_feat = "training/" + task + "/mmf_pgnn"

    df_featurized = get_features(
        ids=ids,
        structures=structures,
        type_features=["mm_fast", "pgnn_mm", "pgnn_ofm", "pgnn_mvl32"],
        path_features=None,
        features=None,
        n_jobs=n_jobs,
        path_saved=f"{task_feat}/features/df_featurized_final.csv.gz",
        path_dir_tmp=f"{task_feat}/tmp_df_feat",
        remove_dir_indiv_feat=False,
    )

    _ = train_fn(
        ids=ids,
        structures=structures,
        targets=targets,
        name_target=name_target,
        path_moddata=f"{task_feat}/moddata/mod.data_training",
        df_featurized=df_featurized,
        moddata=None,
        do_feature_selection=do_feature_selection,
        save_moddata=True,
        n_jobs=n_jobs,
        path_model=f"{task_feat}/models/model_ensemble_modnet.pkl",
        do_training=False,
    )

    # --------------------------------------------------------------------
    # To specify which features are considered
    task_feat = "training/" + task + "/mmf"

    df_featurized = get_features(
        ids=ids,
        structures=structures,
        type_features=["mm_fast"],
        path_features=None,
        features=None,
        n_jobs=n_jobs,
        path_saved=f"{task_feat}/features/df_featurized_final.csv.gz",
        path_dir_tmp=f"{task_feat}/tmp_df_feat",
        remove_dir_indiv_feat=False,
    )

    _ = train_fn(
        ids=ids,
        structures=structures,
        targets=targets,
        name_target=name_target,
        path_moddata=f"{task_feat}/moddata/mod.data_training",
        df_featurized=df_featurized,
        moddata=None,
        do_feature_selection=do_feature_selection,
        save_moddata=True,
        n_jobs=n_jobs,
        path_model=f"{task_feat}/models/model_ensemble_modnet.pkl",
        do_training=False,
    )

    # --------------------------------------------------------------------
    # To specify which features are considered
    task_feat = "training/" + task + "/pgnn"

    df_featurized = get_features(
        ids=ids,
        structures=structures,
        type_features=["pgnn_mm", "pgnn_ofm", "pgnn_mvl32"],
        path_features=None,
        features=None,
        n_jobs=n_jobs,
        path_saved=f"{task_feat}/features/df_featurized_final.csv.gz",
        path_dir_tmp=f"{task_feat}/tmp_df_feat",
        remove_dir_indiv_feat=False,
    )

    _ = train_fn(
        ids=ids,
        structures=structures,
        targets=targets,
        name_target=name_target,
        path_moddata=f"{task_feat}/moddata/mod.data_training",
        df_featurized=df_featurized,
        moddata=None,
        do_feature_selection=do_feature_selection,
        save_moddata=True,
        n_jobs=n_jobs,
        path_model=f"{task_feat}/models/model_ensemble_modnet.pkl",
        do_training=False,
    )

    # ====================================================================
    task = "distribution_250"

    # Load training + validation data
    df_train = load_full()
    df_train = df_train.drop(load_holdout(task).index, axis=0)

    ids = df_train.index.tolist()
    structures = [Structure.from_dict(s) for s in df_train["structure_rot"]]
    name_target = "dKP_full_neum"
    targets = df_train[name_target]

    # --------------------------------------------------------------------
    # To specify which features are considered
    task_feat = "training/" + task + "/mmf_pgnn"

    df_featurized = get_features(
        ids=ids,
        structures=structures,
        type_features=["mm_fast", "pgnn_mm", "pgnn_ofm", "pgnn_mvl32"],
        path_features=None,
        features=None,
        n_jobs=n_jobs,
        path_saved=f"{task_feat}/features/df_featurized_final.csv.gz",
        path_dir_tmp=f"{task_feat}/tmp_df_feat",
        remove_dir_indiv_feat=False,
    )

    _ = train_fn(
        ids=ids,
        structures=structures,
        targets=targets,
        name_target=name_target,
        path_moddata=f"{task_feat}/moddata/mod.data_training",
        df_featurized=df_featurized,
        moddata=None,
        do_feature_selection=do_feature_selection,
        save_moddata=True,
        n_jobs=n_jobs,
        path_model=f"{task_feat}/models/model_ensemble_modnet.pkl",
        do_training=False,
    )

    # --------------------------------------------------------------------
    # To specify which features are considered
    task_feat = "training/" + task + "/mmf"

    df_featurized = get_features(
        ids=ids,
        structures=structures,
        type_features=["mm_fast"],
        path_features=None,
        features=None,
        n_jobs=n_jobs,
        path_saved=f"{task_feat}/features/df_featurized_final.csv.gz",
        path_dir_tmp=f"{task_feat}/tmp_df_feat",
        remove_dir_indiv_feat=False,
    )

    _ = train_fn(
        ids=ids,
        structures=structures,
        targets=targets,
        name_target=name_target,
        path_moddata=f"{task_feat}/moddata/mod.data_training",
        df_featurized=df_featurized,
        moddata=None,
        do_feature_selection=do_feature_selection,
        save_moddata=True,
        n_jobs=n_jobs,
        path_model=f"{task_feat}/models/model_ensemble_modnet.pkl",
        do_training=False,
    )

    # --------------------------------------------------------------------
    # To specify which features are considered
    task_feat = "training/" + task + "/pgnn"

    df_featurized = get_features(
        ids=ids,
        structures=structures,
        type_features=["pgnn_mm", "pgnn_ofm", "pgnn_mvl32"],
        path_features=None,
        features=None,
        n_jobs=n_jobs,
        path_saved=f"{task_feat}/features/df_featurized_final.csv.gz",
        path_dir_tmp=f"{task_feat}/tmp_df_feat",
        remove_dir_indiv_feat=False,
    )

    _ = train_fn(
        ids=ids,
        structures=structures,
        targets=targets,
        name_target=name_target,
        path_moddata=f"{task_feat}/moddata/mod.data_training",
        df_featurized=df_featurized,
        moddata=None,
        do_feature_selection=do_feature_selection,
        save_moddata=True,
        n_jobs=n_jobs,
        path_model=f"{task_feat}/models/model_ensemble_modnet.pkl",
        do_training=False,
    )

    # ====================================================================
    task = "random_125"

    # Load training + validation data
    df_train = load_full()
    df_train = df_train.drop(load_holdout(task).index, axis=0)

    ids = df_train.index.tolist()
    structures = [Structure.from_dict(s) for s in df_train["structure_rot"]]
    name_target = "dKP_full_neum"
    targets = df_train[name_target]

    # --------------------------------------------------------------------
    # To specify which features are considered
    task_feat = "training/" + task + "/mmf_pgnn"

    df_featurized = get_features(
        ids=ids,
        structures=structures,
        type_features=["mm_fast", "pgnn_mm", "pgnn_ofm", "pgnn_mvl32"],
        path_features=None,
        features=None,
        n_jobs=n_jobs,
        path_saved=f"{task_feat}/features/df_featurized_final.csv.gz",
        path_dir_tmp=f"{task_feat}/tmp_df_feat",
        remove_dir_indiv_feat=False,
    )

    _ = train_fn(
        ids=ids,
        structures=structures,
        targets=targets,
        name_target=name_target,
        path_moddata=f"{task_feat}/moddata/mod.data_training",
        df_featurized=df_featurized,
        moddata=None,
        do_feature_selection=do_feature_selection,
        save_moddata=True,
        n_jobs=n_jobs,
        path_model=f"{task_feat}/models/model_ensemble_modnet.pkl",
        do_training=False,
    )

    # --------------------------------------------------------------------
    # To specify which features are considered
    task_feat = "training/" + task + "/mmf"

    df_featurized = get_features(
        ids=ids,
        structures=structures,
        type_features=["mm_fast"],
        path_features=None,
        features=None,
        n_jobs=n_jobs,
        path_saved=f"{task_feat}/features/df_featurized_final.csv.gz",
        path_dir_tmp=f"{task_feat}/tmp_df_feat",
        remove_dir_indiv_feat=False,
    )

    _ = train_fn(
        ids=ids,
        structures=structures,
        targets=targets,
        name_target=name_target,
        path_moddata=f"{task_feat}/moddata/mod.data_training",
        df_featurized=df_featurized,
        moddata=None,
        do_feature_selection=do_feature_selection,
        save_moddata=True,
        n_jobs=n_jobs,
        path_model=f"{task_feat}/models/model_ensemble_modnet.pkl",
        do_training=False,
    )

    # --------------------------------------------------------------------
    # To specify which features are considered
    task_feat = "training/" + task + "/pgnn"

    df_featurized = get_features(
        ids=ids,
        structures=structures,
        type_features=["pgnn_mm", "pgnn_ofm", "pgnn_mvl32"],
        path_features=None,
        features=None,
        n_jobs=n_jobs,
        path_saved=f"{task_feat}/features/df_featurized_final.csv.gz",
        path_dir_tmp=f"{task_feat}/tmp_df_feat",
        remove_dir_indiv_feat=False,
    )

    _ = train_fn(
        ids=ids,
        structures=structures,
        targets=targets,
        name_target=name_target,
        path_moddata=f"{task_feat}/moddata/mod.data_training",
        df_featurized=df_featurized,
        moddata=None,
        do_feature_selection=do_feature_selection,
        save_moddata=True,
        n_jobs=n_jobs,
        path_model=f"{task_feat}/models/model_ensemble_modnet.pkl",
        do_training=False,
    )

    # ====================================================================
    task = "random_250"

    # Load training + validation data
    df_train = load_full()
    df_train = df_train.drop(load_holdout(task).index, axis=0)

    ids = df_train.index.tolist()
    structures = [Structure.from_dict(s) for s in df_train["structure_rot"]]
    name_target = "dKP_full_neum"
    targets = df_train[name_target]

    # --------------------------------------------------------------------
    # To specify which features are considered
    task_feat = "training/" + task + "/mmf_pgnn"

    df_featurized = get_features(
        ids=ids,
        structures=structures,
        type_features=["mm_fast", "pgnn_mm", "pgnn_ofm", "pgnn_mvl32"],
        path_features=None,
        features=None,
        n_jobs=n_jobs,
        path_saved=f"{task_feat}/features/df_featurized_final.csv.gz",
        path_dir_tmp=f"{task_feat}/tmp_df_feat",
        remove_dir_indiv_feat=False,
    )

    _ = train_fn(
        ids=ids,
        structures=structures,
        targets=targets,
        name_target=name_target,
        path_moddata=f"{task_feat}/moddata/mod.data_training",
        df_featurized=df_featurized,
        moddata=None,
        do_feature_selection=do_feature_selection,
        save_moddata=True,
        n_jobs=n_jobs,
        path_model=f"{task_feat}/models/model_ensemble_modnet.pkl",
        do_training=False,
    )

    # --------------------------------------------------------------------
    # To specify which features are considered
    task_feat = "training/" + task + "/mmf"

    df_featurized = get_features(
        ids=ids,
        structures=structures,
        type_features=["mm_fast"],
        path_features=None,
        features=None,
        n_jobs=n_jobs,
        path_saved=f"{task_feat}/features/df_featurized_final.csv.gz",
        path_dir_tmp=f"{task_feat}/tmp_df_feat",
        remove_dir_indiv_feat=False,
    )

    _ = train_fn(
        ids=ids,
        structures=structures,
        targets=targets,
        name_target=name_target,
        path_moddata=f"{task_feat}/moddata/mod.data_training",
        df_featurized=df_featurized,
        moddata=None,
        do_feature_selection=do_feature_selection,
        save_moddata=True,
        n_jobs=n_jobs,
        path_model=f"{task_feat}/models/model_ensemble_modnet.pkl",
        do_training=False,
    )

    # --------------------------------------------------------------------
    # To specify which features are considered
    task_feat = "training/" + task + "/pgnn"

    df_featurized = get_features(
        ids=ids,
        structures=structures,
        type_features=["pgnn_mm", "pgnn_ofm", "pgnn_mvl32"],
        path_features=None,
        features=None,
        n_jobs=n_jobs,
        path_saved=f"{task_feat}/features/df_featurized_final.csv.gz",
        path_dir_tmp=f"{task_feat}/tmp_df_feat",
        remove_dir_indiv_feat=False,
    )

    _ = train_fn(
        ids=ids,
        structures=structures,
        targets=targets,
        name_target=name_target,
        path_moddata=f"{task_feat}/moddata/mod.data_training",
        df_featurized=df_featurized,
        moddata=None,
        do_feature_selection=do_feature_selection,
        save_moddata=True,
        n_jobs=n_jobs,
        path_model=f"{task_feat}/models/model_ensemble_modnet.pkl",
        do_training=False,
    )


if __name__ == "__main__":
    main()
