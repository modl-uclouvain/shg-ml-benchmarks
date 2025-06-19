"Write the structures of the dataset as cif files in dataset/cif_files and build a csv with the target"

import os
from pathlib import Path

import pandas as pd
from pymatgen.core.structure import Structure

from shg_ml_benchmarks.utils import (
    SHG_BENCHMARK_SPLITS,
    load_full,
    load_holdout,
    load_train,
)

# Load the full dataset (unique only)
df_full = load_full()

# Create the dataset/cif_files dir if necessary
os.makedirs("dataset/cif_files", exist_ok=True)

# Iterate over the structures and write them as cif files
for i, s in enumerate(df_full["structure_rot"]):
    path_cif = f"dataset/cif_files/{i}.cif"
    if not Path(path_cif).exists():
        Structure.from_dict(s).to_file(filename=path_cif)

# Create a dataframe with a new index, this new ID, the name of the cif file, the target,
# the material_id, and in which subset (train, test, val) the material is depending on the split
path_id_prop = "dataset/id_prop.csv"
if not Path(path_id_prop).exists():
    df_id_prop = pd.DataFrame(
        data={
            "ID": range(len(df_full)),
            "file": [f"{i}.cif" for i in range(len(df_full))],
            "dKP_full_neum": [d for d in df_full["dKP_full_neum"]],
            "material_id": [i for i in df_full.index],
            "structure": df_full["structure_rot"].tolist(),
        }
    )

    for split in SHG_BENCHMARK_SPLITS:
        df_train = load_train(split)
        df_test = load_holdout(split)

        type_subset = []
        for ir in df_full.index:
            if ir in df_train.index.tolist():
                type_subset.append("train")
            elif ir in df_test.index.tolist():
                type_subset.append("test")
            else:
                type_subset.append("val")

        df_id_prop[split] = type_subset

    df_id_prop.to_csv(path_id_prop)
