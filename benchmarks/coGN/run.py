import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from kgcnn.data.crystal import CrystalDataset
from kgcnn.literature.coGN import make_model
from kgcnn.training.schedule import KerasPolynomialDecaySchedule
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam

# The split to consider
split = "distribution_125"


# Make a Crystal Dataset
path_dataset = "/home/vtrinquet/Softwares_Packages/Github/modl_uclouvain/shg-ml-benchmarks/benchmarks/coGN/training/dataset/"
dataset = CrystalDataset(
    dataset_name="dataset",
    data_directory=path_dataset,
    file_directory="cif_files",
    file_name="id_prop.csv",
)
# dataset.prepare_data(file_column_name="file", overwrite=False)
dataset.prepare_data(file_column_name="file", overwrite=True)
dataset.read_in_memory(label_column_name="dKP_full_neum")

# Dataset is just a list of dictionaries List[Dict]
print("Length:", len(dataset))
print("Dict keys:", dataset[0].keys())


# For making graphs we use a preprocessor to store edge information.
# And apply to dataset.
from kgcnn.crystal.preprocessor import KNNAsymmetricUnitCell

preproc = KNNAsymmetricUnitCell(k=24)
# preproc = VoronoiAsymmetricUnitCell(1e-6)
dataset.set_representation(preproc)
print("Dict keys:", dataset[0].keys())

# # We can make a train-test split.
# train_indices, test_indices = train_test_split(
#     np.arange(len(dataset)), test_size=0.2, random_state=42, shuffle=True
# )
# dataset_train, dataset_test = dataset[train_indices], dataset[test_indices]

# Let's make our custom train-test split
df_id_prop = pd.read_csv(path_dataset + "id_prop.csv", index_col=[0])
test_indices = df_id_prop[df_id_prop[split] == "test"].index.tolist()
val_indices = df_id_prop[df_id_prop[split] == "val"].index.tolist()
train_indices = df_id_prop[df_id_prop[split] == "train"].index.tolist()
# In the current case, train += val
train_indices.extend(val_indices)
# TODO: TO REMOVE BECAUSE JUST FOR TESTING LOCALLY
train_indices = train_indices[:100]
test_indices = test_indices[:25]
dataset_train, dataset_test = dataset[train_indices], dataset[test_indices]

# Get Labels.
# Make sure the have a label dimension
y_train = np.expand_dims(dataset_train.get("graph_labels"), axis=-1)
y_test = np.expand_dims(dataset_test.get("graph_labels"), axis=-1)
print("Label shape", y_train.shape, y_test.shape)

# Standardize Labels
scaler = StandardScaler()
y_train = scaler.fit_transform(y_train)
y_test = scaler.transform(y_test)

# X direct as tensor.
tensors_for_keras_input = {
    "offset": {
        "shape": (None, 3),
        "name": "offset",
        "dtype": "float32",
        "ragged": True,
    },
    "cell_translation": None,
    "affine_matrix": None,
    "voronoi_ridge_area": None,
    "atomic_number": {
        "shape": (None,),
        "name": "atomic_number",
        "dtype": "int32",
        "ragged": True,
    },
    "frac_coords": None,
    "coords": None,
    "multiplicity": {
        "shape": (None,),
        "name": "multiplicity",
        "dtype": "int32",
        "ragged": True,
    },
    "lattice_matrix": None,
    "edge_indices": {
        "shape": (None, 2),
        "name": "edge_indices",
        "dtype": "int32",
        "ragged": True,
    },
    "line_graph_edge_indices": None,
}
x_train = dataset_train.tensor(tensors_for_keras_input)
x_test = dataset_test.tensor(tensors_for_keras_input)
print("Features shape", {key: value.shape for key, value in x_train.items()})


# Get the model
path_model = f"model_{split}.pkl"
if Path(path_model).exists():
    print(f"A model is already saved at {path_model}")
else:
    model = make_model(
        name="coGN",
        # name = "coNGN",
        inputs=tensors_for_keras_input,
        # All defaults else
    )

    # Compile the mode with loss and metrics.
    model.compile(
        loss="mean_absolute_error",
        optimizer=Adam(
            learning_rate=KerasPolynomialDecaySchedule(
                dataset_size=len(train_indices),
                batch_size=64,
                epochs=100,
                lr_start=0.0005,
                lr_stop=1.0e-05,
            )
        ),
        metrics=["mean_absolute_error"],  # Note targets are standard scaled.
    )

    # Fit model.
    model.fit(
        x_train,
        y_train,
        callbacks=[
            # We can use schedule instead of scheduler ...
            # LinearLearningRateScheduler(epo_min=10, epo=1000, learning_rate_start=5e-04, learning_rate_stop=1e-05)
        ],
        validation_data=(x_test, y_test),
        validation_freq=10,
        shuffle=True,
        batch_size=64,
        epochs=100,
        verbose=2,
    )
    # Save model
    with open(path_model, "wb") as f:
        pickle.dump(model, f)

    # Model prediction
    predict_test = scaler.inverse_transform(model.predict(x_test))
    y_test = scaler.inverse_transform(y_test)

    # Save predictions as json
    df_pred = pd.DataFrame(
        index=df_id_prop.iloc[test_indices]["material_id"].tolist(),
        data={"predictions": predict_test.ravel().tolist()},
    )
    path_results = f"results_{split}.json"
    if Path(path_results).exists():
        print(
            f"Some results are already saved at {path_results}... The current results are thus saved at 'results_{split}_tmp.json' instead."
        )
        path_results = f"results_{split}_tmp.json"
    df_pred.to_json(path_results)
