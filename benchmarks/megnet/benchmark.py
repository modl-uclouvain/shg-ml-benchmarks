from __future__ import annotations

from functools import partial
import warnings
import lightning
import torch

from pymatgen.core import Structure
from matgl.ext.pymatgen import Structure2Graph, get_element_list
from matgl.graph.data import MGLDataset, MGLDataLoader, collate_fn_graph
from matgl.layers import BondExpansion
from matgl.models import MEGNet
from matgl.utils.training import ModelLightningModule

from sklearn.model_selection import train_test_split
from shg_ml_benchmarks import run_benchmark
from shg_ml_benchmarks.utils import SHG_BENCHMARK_SPLITS

# To suppress warnings for clearer output
warnings.simplefilter("ignore")


def train_fn(train_df, target, model=None):


    structures = [Structure.from_dict(s) for s in train_df["structure"]]
    targets = train_df[target].values
    element_list = get_element_list(structures)

    # structures_train, structures_test, targets_train, targets_test =  train_test_split(structures, targets, test_size=0.1, random_state=42)

    converter = Structure2Graph(element_types=element_list, cutoff=4.0)

    # convert the raw dataset into MEGNetDataset
    train_data = MGLDataset(
        structures=structures,
        labels={"dKP_full_neum": targets},
        converter=converter,
    )

    # Note, we're using the same data for validation as well! 
    # Slightly wasteful, but we're not doing any hyperparameter opt,
    # and Matgl does not let you run without a validation set
    val_data = MGLDataset(
        structures=structures,
        labels={"dKP_full_neum": targets},
        converter=converter,
    )

    train_loader, val_loader = MGLDataLoader(
        train_data=train_data,
        val_data=val_data,
        collate_fn=collate_fn_graph,
        batch_size=len(structures),
        num_workers=0,
    )

    # setup the MEGNetTrainer
    lit_module = ModelLightningModule(model=model)

    logger = lightning.pytorch.loggers.CSVLogger("logs", name=f"MEGNet_training-{split}")
    trainer = lightning.Trainer(max_epochs=500, accelerator="gpu", logger=logger)
    trainer.fit(model=lit_module, train_dataloaders=train_loader, val_dataloaders=val_loader)  # type: ignore

    return lit_module.model

def predict_fn(model, structure) -> float:
    return float(model.predict_structure(structure))

# setup the embedding layer for node attributes
node_embed = torch.nn.Embedding(85, 16)
# define the bond expansion
bond_expansion = BondExpansion(rbf_type="Gaussian", initial=0.0, final=5.0, num_centers=100, width=0.5)

# setup the architecture of MEGNet model
model = MEGNet(
    dim_node_embedding=16,
    dim_edge_embedding=100,
    dim_state_embedding=2,
    nblocks=3,
    hidden_layer_sizes_input=(64, 32),
    hidden_layer_sizes_conv=(64, 64, 32),
    nlayers_set2set=1,
    niters_set2set=2,
    hidden_layer_sizes_output=(32, 16),
    is_classification=False,
    activation_type="softplus2",
    bond_expansion=bond_expansion,
    cutoff=4.0,
    gauss_width=0.5,
)

for split in SHG_BENCHMARK_SPLITS:
    run_benchmark(task=split, model=model, train_fn=partial(train_fn, model=model), predict_fn=predict_fn, model_label="megnet")
