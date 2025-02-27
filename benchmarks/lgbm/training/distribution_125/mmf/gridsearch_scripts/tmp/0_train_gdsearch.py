def main():
    # python env: modnenv_v2

    import json
    import logging
    import os
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from lightgbm import LGBMRegressor
    from modnet.preprocessing import MODData
    from scipy.stats import spearmanr
    from sklearn.metrics import r2_score
    from sklearn.model_selection import ParameterGrid

    logging.basicConfig(level=logging.INFO)

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    # n_jobs = int(0.7*int(os.environ["SLURM_CPUS_PER_TASK"]))
    n_jobs = int(os.environ["SLURM_CPUS_PER_TASK"])

    # Trick of PP to prevent explosion of the threads
    def setup_threading():
        import os

        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
        os.environ["TF_NUM_INTEROP_THREADS"] = "1"
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    setup_threading()

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

    def compute_metrics(true_values, pred_values):
        mae = np.mean(np.abs(np.array(true_values) - np.array(pred_values)))
        rmse = np.sqrt(np.mean((np.array(true_values) - np.array(pred_values)) ** 2))
        spearmanrho = spearmanr(np.array(true_values), np.array(pred_values)).statistic
        r2score = r2_score(np.array(true_values), np.array(pred_values))

        return {
            "mae": float(mae),
            "rmse": float(rmse),
            "spearman": float(spearmanrho),
            "r2_score": float(r2score),
        }

    cwd = str(Path(os.getcwd()).parent)
    ifile = int(os.path.basename(__file__).split("_")[0])

    # Load the training features
    path_feat = Path(cwd) / "features" / "df_featurized_train.csv.gz"
    df_featurized = pd.read_csv(path_feat, index_col=[0])
    # Define the inputs
    X_train = df_featurized.to_numpy()

    # Load the training features
    path_feat = Path(cwd) / "features" / "df_featurized_val.csv.gz"
    df_featurized_val = pd.read_csv(path_feat, index_col=[0])
    df_featurized_val = replace_nans(df_featurized_val, df_featurized)
    # Define the inputs
    X_val = df_featurized_val.to_numpy()

    # Load the moddata to retrieve the targets
    path_md = Path(cwd) / "moddata" / "mod.data_training_notfeatselect"
    md = MODData.load(path_md)

    # Define the targets
    y_train = md.df_targets.filter(df_featurized.index, axis=0).to_numpy().ravel()
    assert len(y_train) == X_train.shape[0]
    y_val = md.df_targets.filter(df_featurized_val.index, axis=0).to_numpy().ravel()
    assert len(y_val) == X_val.shape[0]

    # Defining the hparams of the model and where to store the results
    param_grid = {
        "num_leaves": list(range(10, 151, 10)),
        "max_depth": list(range(2, 22, 2)),
        "min_child_samples": list(range(6, 31, 2)),
        "learning_rate": list(np.linspace(0.05, 0.5, 100, endpoint=True)),
        "random_state": [42],
        "n_jobs": [n_jobs],
    }

    path_results_gdsearch = Path(cwd) / "gridsearch" / f"{ifile}_results_gdsearch.json"
    os.makedirs(name=path_results_gdsearch.parent, exist_ok=True)
    with open(path_results_gdsearch, "w") as file:
        file.write("{")

    # Iterating over each combination of hparams
    for i, kwargs_model in enumerate(ParameterGrid(param_grid)):
        if i not in list(range(ifile * 4875, (ifile + 1) * 4875)):
            continue

        igdsearch = i
        with open(path_results_gdsearch, "a") as file:
            file.write(f'"{igdsearch}": ')

        print(f"Gridsearching at combination {igdsearch}")

        results_gdsearch_unit = {}
        results_gdsearch_unit["hparams"] = kwargs_model

        logging.info("Training a model.")
        model = LGBMRegressor(**kwargs_model).fit(X_train, y_train)

        logging.info("Predicting the validation targets.")
        preds = model.predict(X_val)
        logging.info("Computing the metrics.")
        metric = compute_metrics(y_val, preds)

        results_gdsearch_unit.update(metric)

        with open(path_results_gdsearch, "a") as file:
            file.write(json.dumps(results_gdsearch_unit, indent=4))
            file.write(",\n")

    with open(path_results_gdsearch, "a") as file:
        file.write("}")


if __name__ == "__main__":
    main()
