import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"
import pickle

import numpy as np
from kgcnn.data.crystal import CrystalDataset
from kgcnn.literature.coGN import make_model
from kgcnn.training.schedule import KerasPolynomialDecaySchedule
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam

# Make a Crystal Dataset
dataset = CrystalDataset(
    dataset_name="ExampleSmallDataset",
    data_directory="/home/vtrinquet/Documents/Doctorat/JNB_Scripts_Clusters/NLO/Graph_models/KGCNN/data/ExampleSmallDataset/",
    file_directory="cif_files",
    file_name="id_prop.csv",
)
# dataset.prepare_data(file_column_name="file", overwrite=False)
dataset.prepare_data(file_column_name="file", overwrite=True)
dataset.read_in_memory(label_column_name="dKP")

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

# We can make a train-test split.
train_indices, test_indices = train_test_split(
    np.arange(len(dataset)), test_size=0.2, random_state=42, shuffle=True
)
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
            dataset_size=159,
            batch_size=64,
            epochs=800,
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
    epochs=200,
    verbose=2,
)

# Model prediction
predict_test = scaler.inverse_transform(model.predict(x_test))
y_test = scaler.inverse_transform(y_test)
print("MAE:", np.mean(np.abs(predict_test - y_test)))
with open("predict_test.pkl", "wb") as f:
    pickle.dump(predict_test, f)
with open("test_indices.pkl", "wb") as f:
    pickle.dump(test_indices, f)
