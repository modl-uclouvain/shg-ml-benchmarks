from modnet.featurizers.presets.matminer_2024_fast import Matminer2024FastFeaturizer
from modnet.preprocessing import MODData
from modnet.models import EnsembleMODNetModel
from pathlib import Path
from pymatgen.core.structure import Structure

import os
import pandas as pd


cur_v = int(Path(os.getcwd()).parent.name.split("_")[0])

# Load df_outputs_filtout
path_df_outputs_filtout = Path(f'../data/df_outputs_filtout.pkl.gz')
df_outputs_filtout = pd.read_pickle(path_df_outputs_filtout)
print(f"{df_outputs_filtout.shape = }")

# Create new MODData
path_md = f'../data/mod.data_dKP_fastfeat_n_Eg_pgnn_featselec_v{cur_v}'
if Path(path_md+".gz").exists():
    print("MODData already exists, let's load it")
    os.system(f"gunzip {path_md}.gz")
    md = MODData.load(path_md)
    os.system(f"gzip {path_md}")
else:
    md = MODData(
        materials=[Structure.from_dict(s) for s in df_outputs_filtout['structure'].tolist()],
        targets=df_outputs_filtout[['dKP']].to_numpy(),
        target_names=["dKP"],
        structure_ids=df_outputs_filtout.index.tolist(),
    )

    # Featurize it
        # md.featurizer = Matminer2024FastFeaturizer()
        # md.featurize(n_jobs=2)
    # or load from previous MODData, which also contains predicted n and src Eg
    path_md_previous = f"../../{cur_v-1}_al_it/data/mod.data_dKP_fastfeat_n_Eg_pgnn_featselec_v{cur_v-1}"
    os.system(f"gunzip {path_md_previous}.gz")
    df_featurized = MODData.load(path_md_previous).df_featurized
    os.system(f"gzip {path_md_previous}")

    # and global search space, which also contains predicted n and src Eg
    path_full_feat = "../../../00_prepa_dbs/full_search_space/data/df_full_feat.pkl.gz"
    df_full_feat = pd.read_pickle(path_full_feat)\
                    .drop(df_featurized.index, axis=0, errors='ignore')\
                    .filter(df_outputs_filtout.index, axis=0)

    # Load the pGNN features of the full search space
    path_df_pgnn_mmv1 = "../../../00_prepa_dbs/full_search_space/data/df_mmv1.pkl.gz"
    path_df_pgnn_ofm = "../../../00_prepa_dbs/full_search_space/data/df_ofm.pkl.gz"
    path_df_pgnn_mvl32 = "../../../00_prepa_dbs/full_search_space/data/df_mvl32.pkl.gz"
    df_pgnn = pd.concat([pd.read_pickle(path).drop(['structure'], axis=1, errors='ignore') for path in [path_df_pgnn_mmv1, path_df_pgnn_ofm, path_df_pgnn_mvl32]], axis=1)
    df_pgnn = df_pgnn.filter(df_full_feat.index, axis=0)
    assert len(df_full_feat)==len(df_pgnn)
    df_full_feat = pd.concat([df_full_feat, df_pgnn], axis=1)

    assert df_featurized.shape[1]==df_full_feat.shape[1]
    df_featurized = pd.concat([df_featurized, df_full_feat], axis=0)

    # Featurize missing entries
    df_missing = df_outputs_filtout.drop(df_featurized.index, axis=0, errors='ignore')
    if len(df_missing)>0:
        print("Some entries are missing features, let's compute them.")
        raise Exception(f"Need to featurize with pgnn also")
        md_missing = MODData(
            materials=[Structure.from_dict(s) for s in df_missing['structure'].tolist()],
            targets=[None]*len(df_missing),
            target_names=None,
            structure_ids=df_missing.index.tolist(),
            featurizer = Matminer2024FastFeaturizer()
        )
        md_missing.featurize(n_jobs=2)
        df_featurized_missing = md_missing.df_featurized

        # Predict n
        model_path = '/home/vtrinquet/Documents/Doctorat/JNB_Scripts_Clusters/NLO/Custom_Features_NLO/predict_n/models/production/GA_Rf0_Nstd5-refractive_index_prod_refr_idx.pkl'
        model = EnsembleMODNetModel.load(model_path)
        name_target = 'refractive_index'
        predictions, uncertainties = model.predict(md_missing, return_unc=True)
        uncertainties = uncertainties.filter(predictions.index, axis=0)
        df_pred_n = pd.DataFrame(index=predictions.index.tolist())
        df_pred_n[name_target] = predictions[predictions.columns[0]].tolist()
        df_pred_n[f"{name_target}_unc"] = uncertainties[uncertainties.columns[0]].tolist()
        df_featurized_missing = pd.concat([df_featurized_missing, df_pred_n], axis=1)
        # Get bandgap
        df_tmp = df_outputs_filtout[['src_bandgap']].filter(df_featurized_missing.index, axis=0).rename({"src_bandgap": "bandgap"}, axis=1)
        assert df_tmp.shape[0]==df_featurized_missing.shape[0]
        df_tmp['bandgap_unc'] = [0]*len(df_tmp)
        df_featurized_missing = pd.concat([df_featurized_missing, df_tmp], axis=1)

        assert df_featurized.shape[1]==df_featurized_missing.shape[1],f"{df_featurized.shape[1]} != {df_featurized_missing.shape[1]}"
        df_featurized = pd.concat([df_featurized, df_featurized_missing], axis=0)

    df_featurized = df_featurized.filter(df_outputs_filtout.index, axis=0)
    assert df_featurized.shape[0] == df_outputs_filtout.shape[0],f"{df_featurized.shape[0]} != {df_outputs_filtout.shape[0]}"
    assert df_featurized.shape[0] == md.df_targets.shape[0]

    md.df_featurized = df_featurized

    # Feature selection
    # md.feature_selection(n=-1, n_samples=len(md.df_featurized)+1, n_jobs=2)

    md.save(path_md)
    os.system(f"gzip {path_md}")
    print("NEW MODDATA CREATED AND SAVED")

print(f"{md.df_featurized.shape = }")



# # Identify the new features resulting from FastFeaturizer --> # addition of ElementFraction Z 104 --> 118
#     # Load old MODData new MODData to refeaturize fast
# path_md_training = Path(f"../data/mod.data_nl_featselec_dKP-dRMS_v{cur_v}")
# df_featurized_old = MODData.load(path_md_training).df_featurized
# print(f"{df_featurized_old.shape = }")
# df_new_feat = md.df_featurized.drop(df_featurized_old.columns, axis=1, errors='ignore')
# print(f"{df_new_feat.columns = }")
# df_new_feat.to_pickle("df_tmp_new_feat.pkl")
# Filter the features to restrict to the 197 fast features used since the beginning
    # LET'S KEEP ALL THE FEATURES FOR SIMPLICITY SINCE ALL THE NEW ONES ARE 0
