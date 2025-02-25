import os

import h5py

os.environ["TF_USE_LEGACY_KERAS"] = "1"
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from graphlist import HDFGraphList
from kgcnn.crystal.preprocessor import VoronoiAsymmetricUnitCell
from kgcnn.graph.methods import get_angle_indices
from kgcnn.literature.coGN import make_model, model_default_nested
from kgcnn.training.schedule import KerasPolynomialDecaySchedule
from prepare_dataset_coNGN import SHGDataset
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam

# The split to consider
split = "distribution_125"
# preproc = KNNAsymmetricUnitCell(k=24)
preproc = VoronoiAsymmetricUnitCell(1e-6)


def get_input_tensors(inputs, graphlist):
    """Returns input tensors

    Args:
        inputs (list): Input layers for the model (e.g. `model.inputs`).
        graphlist (GraphList): Data to build tensors from in GraphList form.
    Returns:
        dict: Dictionary of input tensors with {tensorname: tensordata} mapping.
    """
    input_names = [input.name for input in inputs]
    input_tensors = {}
    for input_name in graphlist.edge_attributes.keys():
        if input_name in input_names:
            input_tensors[input_name] = tf.RaggedTensor.from_row_lengths(
                graphlist.edge_attributes[input_name], graphlist.num_edges
            )
    for input_name in graphlist.node_attributes.keys():
        if input_name in input_names:
            input_tensors[input_name] = tf.RaggedTensor.from_row_lengths(
                graphlist.node_attributes[input_name], graphlist.num_nodes
            )
    for input_name in graphlist.graph_attributes.keys():
        if input_name in input_names:
            input_tensors[input_name] = tf.convert_to_tensor(
                graphlist.graph_attributes[input_name]
            )
    input_tensors["edge_indices"] = tf.RaggedTensor.from_row_lengths(
        graphlist.edge_indices[:][:, [1, 0]], graphlist.num_edges
    )
    if "line_graph_edge_indices" in input_names:
        graphs_line_graph_edge_indices = []
        for g in graphlist:
            line_graph_edge_indices = get_angle_indices(
                g.edge_indices, edge_pairing="kj"
            )[2].reshape(-1, 2)  # \measuredangle e_ij e_kj
            graphs_line_graph_edge_indices.append(line_graph_edge_indices)
        line_graph_edge_indices = tf.RaggedTensor.from_row_lengths(
            np.concatenate(graphs_line_graph_edge_indices),
            [len(lg) for lg in graphs_line_graph_edge_indices],
        )
        input_tensors["line_graph_edge_indices"] = line_graph_edge_indices

    return input_tensors


def get_id_index_mapping(graphlist):
    index_mapping = {
        id_.decode(): i
        for i, id_ in enumerate(graphlist.graph_attributes["dataset_id"][:])
    }
    return index_mapping


def get_graphs(id_index_mapping, graphlist, inputs):
    idxs = [id_index_mapping[id_] for id_ in inputs.index]
    return graphlist[idxs]


# Make a Crystal Dataset
path_dataset = "../dataset/"

# Let's make our custom train-test split
df_id_prop = pd.read_csv(path_dataset + "id_prop.csv", index_col=[0])

model = make_model(**model_default_nested)

# ================================================================================================================
# Training set

# Returns file path to preprocessed crystals.
# If crystals aren't preprocessed yet with the given preprocessor, the file is creates first.
# Preprocessing a crystal may take a while.
shg_data = SHGDataset(cache_dir="../dataset_graphs")
preprocessed_crystals_file = shg_data.get_dataset_file(
    dataset_name=split,
    task_name="training",
    preprocessor=preproc,
    id_prop_file=path_dataset + "id_prop.csv",
)

# Load preprocessed cyrstals
with h5py.File(preprocessed_crystals_file, "r") as f:
    # Load as GraphList data structure
    preprocessed_crystals = HDFGraphList(f)
    ids_order = [
        id_.decode("UTF-8").replace(" ", "")
        for id_ in preprocessed_crystals.graph_attributes["dataset_id"][:]
    ]

    # Get Tensor Input
    x_train = get_input_tensors(model.inputs, preprocessed_crystals)

# Test set

# Returns file path to preprocessed crystals.
# If crystals aren't preprocessed yet with the given preprocessor, the file is creates first.
# Preprocessing a crystal may take a while.
shg_data = SHGDataset(cache_dir="../dataset_graphs")
preprocessed_crystals_file = shg_data.get_dataset_file(
    dataset_name=split,
    task_name="test",
    preprocessor=preproc,
    id_prop_file=path_dataset + "id_prop.csv",
)

# Load preprocessed cyrstals
with h5py.File(preprocessed_crystals_file, "r") as f:
    # Load as GraphList data structure
    preprocessed_crystals = HDFGraphList(f)
    ids_order_test = [
        id_.decode("UTF-8").replace(" ", "")
        for id_ in preprocessed_crystals.graph_attributes["dataset_id"][:]
    ]

    # Get Tensor Input
    x_test = get_input_tensors(model.inputs, preprocessed_crystals)

# # ================================================================================================================

# # Get Labels.
# # Make sure the have a label dimension
# y_train = np.expand_dims(dataset_train.get("graph_labels"), axis=-1)
# y_test = np.expand_dims(dataset_test.get("graph_labels"), axis=-1)

df_tmp = df_id_prop[df_id_prop[split] != "test"]
df_tmp.index = df_tmp["material_id"]
df_tmp = df_tmp.filter(ids_order, axis=0)
y_train = df_tmp["dKP_full_neum"].to_numpy().reshape(len(df_tmp), 1)

df_tmp = df_id_prop[df_id_prop[split] == "test"]
df_tmp.index = df_tmp["material_id"]
df_tmp = df_tmp.filter(ids_order_test, axis=0)
y_test = df_tmp["dKP_full_neum"].to_numpy().reshape(len(df_tmp), 1)
print("Label shape", y_train.shape, y_test.shape)

# Standardize Labels
scaler = StandardScaler()
y_train = scaler.fit_transform(y_train)
y_test = scaler.transform(y_test)

# Get the model
path_model = f"model_{split}.pkl"
if Path(path_model).exists():
    print(f"A model is already saved at {path_model}")
else:
    raise Exception("Not training locally")

    model = make_model(**model_default_nested)

    # Compile the mode with loss and metrics.
    model.compile(
        loss="mean_absolute_error",
        optimizer=Adam(
            learning_rate=KerasPolynomialDecaySchedule(
                dataset_size=y_train.shape[0],
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
        epochs=800,
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
        index=ids_order_test, data={"predictions": predict_test.ravel().tolist()}
    )
    path_results = f"results_{split}.json"
    if Path(path_results).exists():
        print(
            f"Some results are already saved at {path_results}... The current results are thus saved at 'results_{split}_tmp.json' instead."
        )
        path_results = f"results_{split}_tmp.json"
    df_pred.to_json(path_results)
