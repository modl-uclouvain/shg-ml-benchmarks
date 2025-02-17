from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core.structure import Structure

import shg_ml_benchmarks.utils_shg as shg
from shg_ml_benchmarks.utils import DATA_DIR, load_full


def write_data(
    structures: list[Structure],
    tensors: list[np.ndarray],
    path: Path = "shg_tensors.json",
):
    """Write structures and tensors to file.

    Args:
        structures: list of pymatgen structures.
        tensors: list of 3x3x3 shg tensors.
        path: path to write the data.
    """
    data = {
        "structure": [s.as_dict() for s in structures],
        "shg_tensor_full": [t.tolist() for t in tensors],
    }
    df = pd.DataFrame(data)

    df.to_json(path)


# Load the dataset
df = load_full()

# Validation data
# ==============================================================================
type_set = "validation"
type_sampling = "distribution"
n_sampling = 125
list_idx = pd.read_json(f"{DATA_DIR}/{type_set}_id_{type_sampling}_{n_sampling}.json")[
    0
].tolist()

path_set_matten = f"dataset_{type_set}_{type_sampling}_{n_sampling}.json"
if not Path(path_set_matten).exists():
    df_tmp = df.filter(list_idx, axis=0)
    write_data(
        [Structure.from_dict(s) for s in df_tmp["structure_rot"].tolist()],
        [shg.from_voigt(d) for d in df_tmp["dijk_full_neum"].tolist()],
        path=path_set_matten,
    )
# ==============================================================================
type_set = "validation"
type_sampling = "random"
n_sampling = 125
list_idx = pd.read_json(f"{DATA_DIR}/{type_set}_id_{type_sampling}_{n_sampling}.json")[
    0
].tolist()

path_set_matten = f"dataset_{type_set}_{type_sampling}_{n_sampling}.json"
if not Path(path_set_matten).exists():
    df_tmp = df.filter(list_idx, axis=0)
    write_data(
        [Structure.from_dict(s) for s in df_tmp["structure_rot"].tolist()],
        [shg.from_voigt(d) for d in df_tmp["dijk_full_neum"].tolist()],
        path=path_set_matten,
    )
# ==============================================================================
type_set = "validation"
type_sampling = "distribution"
n_sampling = 250
list_idx = pd.read_json(f"{DATA_DIR}/{type_set}_id_{type_sampling}_{n_sampling}.json")[
    0
].tolist()

path_set_matten = f"dataset_{type_set}_{type_sampling}_{n_sampling}.json"
if not Path(path_set_matten).exists():
    df_tmp = df.filter(list_idx, axis=0)
    write_data(
        [Structure.from_dict(s) for s in df_tmp["structure_rot"].tolist()],
        [shg.from_voigt(d) for d in df_tmp["dijk_full_neum"].tolist()],
        path=path_set_matten,
    )
# ==============================================================================
type_set = "validation"
type_sampling = "random"
n_sampling = 250
list_idx = pd.read_json(f"{DATA_DIR}/{type_set}_id_{type_sampling}_{n_sampling}.json")[
    0
].tolist()

path_set_matten = f"dataset_{type_set}_{type_sampling}_{n_sampling}.json"
if not Path(path_set_matten).exists():
    df_tmp = df.filter(list_idx, axis=0)
    write_data(
        [Structure.from_dict(s) for s in df_tmp["structure_rot"].tolist()],
        [shg.from_voigt(d) for d in df_tmp["dijk_full_neum"].tolist()],
        path=path_set_matten,
    )
# ==============================================================================

# Holdout
# ==============================================================================
type_set = "holdout"
type_sampling = "distribution"
n_sampling = 125
list_idx = pd.read_json(f"{DATA_DIR}/{type_set}_id_{type_sampling}_{n_sampling}.json")[
    0
].tolist()

path_set_matten = f"dataset_{type_set}_{type_sampling}_{n_sampling}.json"
if not Path(path_set_matten).exists():
    df_tmp = df.filter(list_idx, axis=0)
    write_data(
        [Structure.from_dict(s) for s in df_tmp["structure_rot"].tolist()],
        [shg.from_voigt(d) for d in df_tmp["dijk_full_neum"].tolist()],
        path=path_set_matten,
    )
# ==============================================================================
type_set = "holdout"
type_sampling = "random"
n_sampling = 125
list_idx = pd.read_json(f"{DATA_DIR}/{type_set}_id_{type_sampling}_{n_sampling}.json")[
    0
].tolist()

path_set_matten = f"dataset_{type_set}_{type_sampling}_{n_sampling}.json"
if not Path(path_set_matten).exists():
    df_tmp = df.filter(list_idx, axis=0)
    write_data(
        [Structure.from_dict(s) for s in df_tmp["structure_rot"].tolist()],
        [shg.from_voigt(d) for d in df_tmp["dijk_full_neum"].tolist()],
        path=path_set_matten,
    )
# ==============================================================================
type_set = "holdout"
type_sampling = "distribution"
n_sampling = 250
list_idx = pd.read_json(f"{DATA_DIR}/{type_set}_id_{type_sampling}_{n_sampling}.json")[
    0
].tolist()

path_set_matten = f"dataset_{type_set}_{type_sampling}_{n_sampling}.json"
if not Path(path_set_matten).exists():
    df_tmp = df.filter(list_idx, axis=0)
    write_data(
        [Structure.from_dict(s) for s in df_tmp["structure_rot"].tolist()],
        [shg.from_voigt(d) for d in df_tmp["dijk_full_neum"].tolist()],
        path=path_set_matten,
    )
# ==============================================================================
type_set = "holdout"
type_sampling = "random"
n_sampling = 250
list_idx = pd.read_json(f"{DATA_DIR}/{type_set}_id_{type_sampling}_{n_sampling}.json")[
    0
].tolist()

path_set_matten = f"dataset_{type_set}_{type_sampling}_{n_sampling}.json"
if not Path(path_set_matten).exists():
    df_tmp = df.filter(list_idx, axis=0)
    write_data(
        [Structure.from_dict(s) for s in df_tmp["structure_rot"].tolist()],
        [shg.from_voigt(d) for d in df_tmp["dijk_full_neum"].tolist()],
        path=path_set_matten,
    )
# ==============================================================================

# Training
# ==============================================================================
type_set = "training"
type_sampling = "distribution"
n_sampling = 125

list_idx = []
for type_set_tmp in ["validation", "holdout"]:
    list_idx_tmp = pd.read_json(
        f"{DATA_DIR}/{type_set_tmp}_id_{type_sampling}_{n_sampling}.json"
    )[0].tolist()
    list_idx.extend(list_idx_tmp)

path_set_matten = f"dataset_{type_set}_{type_sampling}_{n_sampling}.json"
if not Path(path_set_matten).exists():
    df_tmp = df.drop(list_idx, axis=0)
    write_data(
        [Structure.from_dict(s) for s in df_tmp["structure_rot"].tolist()],
        [shg.from_voigt(d) for d in df_tmp["dijk_full_neum"].tolist()],
        path=path_set_matten,
    )
# ==============================================================================
type_set = "training"
type_sampling = "random"
n_sampling = 125

list_idx = []
for type_set_tmp in ["validation", "holdout"]:
    list_idx_tmp = pd.read_json(
        f"{DATA_DIR}/{type_set_tmp}_id_{type_sampling}_{n_sampling}.json"
    )[0].tolist()
    list_idx.extend(list_idx_tmp)

path_set_matten = f"dataset_{type_set}_{type_sampling}_{n_sampling}.json"
if not Path(path_set_matten).exists():
    df_tmp = df.drop(list_idx, axis=0)
    write_data(
        [Structure.from_dict(s) for s in df_tmp["structure_rot"].tolist()],
        [shg.from_voigt(d) for d in df_tmp["dijk_full_neum"].tolist()],
        path=path_set_matten,
    )
# ==============================================================================
type_set = "training"
type_sampling = "distribution"
n_sampling = 250

list_idx = []
for type_set_tmp in ["validation", "holdout"]:
    list_idx_tmp = pd.read_json(
        f"{DATA_DIR}/{type_set_tmp}_id_{type_sampling}_{n_sampling}.json"
    )[0].tolist()
    list_idx.extend(list_idx_tmp)

path_set_matten = f"dataset_{type_set}_{type_sampling}_{n_sampling}.json"
if not Path(path_set_matten).exists():
    df_tmp = df.drop(list_idx, axis=0)
    write_data(
        [Structure.from_dict(s) for s in df_tmp["structure_rot"].tolist()],
        [shg.from_voigt(d) for d in df_tmp["dijk_full_neum"].tolist()],
        path=path_set_matten,
    )
# ==============================================================================
type_set = "training"
type_sampling = "random"
n_sampling = 250

list_idx = []
for type_set_tmp in ["validation", "holdout"]:
    list_idx_tmp = pd.read_json(
        f"{DATA_DIR}/{type_set_tmp}_id_{type_sampling}_{n_sampling}.json"
    )[0].tolist()
    list_idx.extend(list_idx_tmp)

path_set_matten = f"dataset_{type_set}_{type_sampling}_{n_sampling}.json"
if not Path(path_set_matten).exists():
    df_tmp = df.drop(list_idx, axis=0)
    write_data(
        [Structure.from_dict(s) for s in df_tmp["structure_rot"].tolist()],
        [shg.from_voigt(d) for d in df_tmp["dijk_full_neum"].tolist()],
        path=path_set_matten,
    )
# ==============================================================================
