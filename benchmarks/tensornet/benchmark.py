from __future__ import annotations

import shutil
import warnings
from functools import partial
from pathlib import Path

import lightning
from matgl.ext.pymatgen import Structure2Graph
from matgl.graph.data import MGLDataLoader, MGLDataset, collate_fn_graph
from matgl.models import TensorNet
from matgl.utils.training import ModelLightningModule
from pymatgen.core import Element, Structure
from sklearn.model_selection import train_test_split

from shg_ml_benchmarks import run_benchmark
from shg_ml_benchmarks.utils import SHG_BENCHMARK_SPLITS

# To suppress warnings for clearer output
warnings.simplefilter("ignore")
ELEMENT_LIST = [str(Element.from_Z(Z)) for Z in range(1, 92)]


def train_fn(train_df, target, model=None):
    structures = [Structure.from_dict(s) for s in train_df["structure"]]
    targets = train_df[target].values

    converter = Structure2Graph(element_types=ELEMENT_LIST, cutoff=4.0)

    train_structures, val_structures, train_targets, val_targets = train_test_split(
        structures, targets, test_size=0.1, random_state=42
    )

    # convert the raw dataset into MEGNetDataset
    train_data = MGLDataset(
        structures=train_structures,
        labels={"dKP_full_neum": train_targets},
        converter=converter,
    )

    # Note, we're using the same data for validation as well!
    # Slightly wasteful, but we're not doing any hyperparameter opt,
    # and Matgl does not let you run without a validation set
    val_data = MGLDataset(
        structures=val_structures,
        labels={"dKP_full_neum": val_targets},
        converter=converter,
    )

    train_loader, val_loader = MGLDataLoader(
        train_data=train_data,
        val_data=val_data,
        collate_fn=collate_fn_graph,
        batch_size=32,
        num_workers=0,
    )

    # setup the MEGNetTrainer
    lit_module = ModelLightningModule(model=model)

    logger = lightning.pytorch.loggers.CSVLogger(
        "logs", name=f"TensorNet_training-{split}"
    )
    trainer = lightning.Trainer(max_epochs=200, accelerator="gpu", logger=logger)
    trainer.fit(
        model=lit_module, train_dataloaders=train_loader, val_dataloaders=val_loader
    )  # type: ignore

    return lit_module.model


def predict_fn(model, structure) -> float:
    return float(model.predict_structure(structure))


for equivariance_invariance_group in ("O(3)", "SO(3)"):
    model = TensorNet(equivariance_invariance_group=equivariance_invariance_group)

    for split in SHG_BENCHMARK_SPLITS:
        if Path("MGLDataset").exists():
            shutil.rmtree("MGLDataset")

        run_benchmark(
            task=split,
            model=model,
            train_fn=partial(train_fn, model=model),
            predict_fn=predict_fn,
            model_label="tensornet",
            tasks_tag=f"_{equivariance_invariance_group}",
        )
