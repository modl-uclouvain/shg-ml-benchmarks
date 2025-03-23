# Predict the LDA KP coefficient of any MODData
# venv: shgmlenv

import os
from pathlib import Path

import pandas as pd
from modnet.models import EnsembleMODNetModel
from modnet.preprocessing import MODData

# Define the paths
for task, db_name in [
    ("gnome2025", "GNome2025"),
    ("gnome2025_v2", "GNome2025_v2"),
    ("alexandria2025", "Alexandria2025"),
]:
    task_feat = "../data/" + task + "/mmf_pgnn_pred_n_gap"
    path_pred_dir = f"{task_feat}/predictions"
    os.makedirs(path_pred_dir, exist_ok=True)
    path_pred = f"{path_pred_dir}/df_dKP_pred_unc_{task}.json.gz"
    
    # Check if predictions already saved
    if Path(path_pred).exists():
        print(f"A file already exists at {path_pred}")
    else:
        # Load the MODData to predict
        path_moddata = f"{task_feat}/moddata/mod.data_{task}_notfeatselect"
        md = MODData.load(path_moddata)
    
        # Load the LDA KP coefficient model
        path_model = "/home/vtrinquet/Documents/Doctorat/JNB_Scripts_Clusters/NLO/HT/shg_nov24/01_humanguided_al_v1/20_al_it/models/production/GA_Rf0_Nstd5-dKP_prod_v20.pkl.gz"
        model = EnsembleMODNetModel.load(path_model)
    
        # Predict the MODData
        predictions, uncertainties = model.predict(md, return_unc=True)
        predictions = predictions.filter(md.df_targets.index, axis=0)
        uncertainties = uncertainties.filter(predictions.index, axis=0)
        assert predictions.shape[0] == md.df_featurized.shape[0]
        df_pred = pd.DataFrame(index=predictions.index.tolist())
        df_pred["dKP_pred"] = predictions[predictions.columns[0]].tolist()
        df_pred["dKP_unc"] = uncertainties[uncertainties.columns[0]].tolist()
    
        # Save the predictions
        df_pred.to_json(path_pred)
