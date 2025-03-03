from __future__ import annotations

from functools import partial
import warnings
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import CSVLogger

from matgl.ext.pymatgen import Structure2Graph, get_element_list
from matgl.graph.data import MGLDataset, MGLDataLoader, collate_fn_graph
from matgl.layers import BondExpansion
from matgl.models import MEGNet
from matgl.utils.training import ModelLightningModule

from shg_ml_benchmarks import run_benchmark
from shg_ml_benchmarks.utils import SHG_BENCHMARK_SPLITS

# To suppress warnings for clearer output
warnings.simplefilter("ignore")

# get element types in the dataset
element_list = list(range(1, 60))
# setup a graph converter
converter = Structure2Graph(element_types=element_list, cutoff=4.0)

def train_fn(structures, targets, model):

    # convert the raw dataset into MEGNetDataset
    train_data = MGLDataset(
        structures=structures,
        labels={"dKP_full_neum": targets},
        converter=converter,
    )

    train_loader = MGLDataLoader(
        train_data=train_data,
        val_data=train_data,
        collate_fn=collate_fn_graph,
        batch_size=1,
        num_workers=0,
    )

    # setup the MEGNetTrainer
    lit_module = ModelLightningModule(model=model)

    logger = CSVLogger("logs", name="MEGNet_training")
    trainer = pl.Trainer(max_epochs=20, accelerator="gpu", logger=logger)
    trainer.fit(model=lit_module, train_dataloaders=train_loader)  # type: ignore

    return lit_module.model

def predict_fn(structure, model=None):
    return model.predict_structure(structure)

# setup the embedding layer for node attributes
node_embed = torch.nn.Embedding(len(element_list), 16)
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

run_benchmark(model=model, train_fn=train_fn, predict_fn=partial(predict_fn, model=model), model_label="MEGNet")
