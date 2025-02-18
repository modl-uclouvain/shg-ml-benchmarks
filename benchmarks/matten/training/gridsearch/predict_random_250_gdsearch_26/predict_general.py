import os
from pathlib import Path

import pandas as pd
from matten.predict import predict
from pymatgen.core import Structure

if __name__ == "__main__":
    df = pd.read_pickle(
        "/home/vtrinquet/Softwares_Packages/Github/modl_uclouvain/shg-ml-benchmarks/data/df_rot_ieee_pmg.pkl.gz"
    )
    df = df.query("is_unique_here == True")

    type_set = "holdout"
    type_sampling = os.path.basename(Path(os.getcwd())).split("predict_")[1].replace("_gdsearch_26", "")
    # type_sampling = type_sampling.split("_")[0]
    n_sampling = int(type_sampling.split("_")[1])

    list_idx = pd.read_json(
        f"~/Softwares_Packages/Github/modl_uclouvain/shg-ml-benchmarks/data/{type_set}_id_{type_sampling}.json"
    )[0].tolist()
    path_set_matten = f"../../datasets/dataset_{type_set}_{type_sampling}.json"
    df_holdout = df.filter(list_idx, axis=0)
    assert len(df_holdout) == n_sampling

    list_tensor = predict(
        [Structure.from_dict(s) for s in df_holdout["structure_rot"]],
        model_identifier=".",
        # checkpoint="epoch=9-step=10.ckpt",
        is_elasticity_tensor=False,
    )

    df_predictions = pd.DataFrame(index=df_holdout.index)
    df_predictions["dijk_matten"] = list_tensor

    path_df_pred = "df_pred_matten_holdout.json.gz"
    if not Path(path_df_pred).exists():
        df_predictions.to_json(path_df_pred)

    # print("value:", tensor)
    # print("type:", type(tensor))
    # print("shape:", tensor.shape)
