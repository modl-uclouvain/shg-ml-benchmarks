import logging
from functools import partial
from modnet.preprocessing import MODData
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO)

from shg_ml_benchmarks import run_benchmark
from shg_ml_benchmarks.utils import SHG_BENCHMARK_SPLITS

from train import train_fn, get_features


def predict_fn(
        model,
        structures,
        ids,
        type_features=["mm_fast", "pgnn_mm", "pgnn_ofm", "pgnn_mvl32"],
        features=None,
        n_jobs=2,
):

    # 1. Featurize the structure with the same features as the one the model was trained on (consider that an input of the function?)
        # - Use get_features()?

    df_featurized = get_features(
        ids = ids,
        structures = structures,
        type_features = type_features,
        features = features,
        n_jobs = n_jobs,
        path_saved = None,
        remove_dir_indiv_feat = True,
    )
    # 1.b Padd the missing features by zero
    features_needed = []
    for vanilla in model.models:
        features_needed.extend(vanilla.optimal_descriptors)
    features_needed = set(features_needed)
    for feat in features_needed:
        if feat in df_featurized.columns:
            continue
        print(f"WARNING: THE FEATURE {feat} IS REQUIRED BY THE MODEL, BUT IS NOT AMONG THE FEATURES JUST OBTAINED... IT IS THUS ADDED WITH A VALUE OF NAN...")
        df_featurized[feat] = [np.nan]*len(df_featurized)

    # 2. Create the MODData to predict from df_featurized
    md = MODData(
        materials=structures,
        targets = [0.0]*len(structures),
        target_names=["dKP_full_neum"],
        structure_ids=ids,
    )
    md.df_featurized = df_featurized
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
