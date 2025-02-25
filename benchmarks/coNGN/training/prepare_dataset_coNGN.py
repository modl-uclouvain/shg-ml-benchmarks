"""Prepares the training and test data as graphs and saves it."""

import json
from collections.abc import Iterable
from itertools import islice
from multiprocessing import Pool
from pathlib import Path

import h5py
import pandas as pd
from graphlist import GraphList, HDFGraphList
from kgcnn.crystal.preprocessor import (
    CrystalPreprocessor,
    VoronoiAsymmetricUnitCell,
)
from networkx import MultiDiGraph
from pymatgen.core.structure import Structure


class SHGDataset:
    """MatbenchDataset class for graph representations of the Matbench datasets.
    This class computes graph representations for crystal structures in the matbench datasets
    and caches them on the disk (in a specified directory as HDF files) for future use.
    This way crystal preprocessing is done only once for each dataset-preprocessor-combination.
    Upon instantiation of the class the user has to specify the directory where cached files are stored.
    Afterwards the user can call `get_dataset_file` to get the file to the preprocessed crystals
    for a dataset-preprocessor-combination..
    In case this preprocessor has been used for the dataset-preprocessor-combination it simply returns the file path,
    otherwise it computes the graph, stores it in a new file and returns this file path.
    """

    def __init__(self, cache_dir: str | Path):
        """Instantiates a MatbenchDataset object for a given cache directory.
        Args:
            cache_dir (Union[str, Path]): Path to the directory where preprocessed crystal graphs should be stored.
        """
        if isinstance(cache_dir, str):
            cache_dir = Path(cache_dir)
        self.cache_dir = cache_dir
        if not cache_dir.is_dir():
            cache_dir.mkdir(exist_ok=True, parents=True)

    @staticmethod
    def crystal_iterator(crystal_series: pd.Series):
        """Iterator over pymatgen structures with meta information, which is needed to create the crystal graphs.
        Args:
            crystal_series (pd.Series): Pandas Series of the crystals (as provided by matbench python package).
        Yields:
             pymatgen structures with meta information
        """
        # Needed to avoid truncating the ids (happens in graphlist)
        max_len = len(max(crystal_series.index.tolist(), key=len))

        for id_, x in zip(crystal_series.index, crystal_series):
            new_id = id_.ljust(max_len)
            setattr(x, "dataset_id", new_id.encode())
            yield x

    def dataset_exists(
        self, dataset_name: str, preprocessor: CrystalPreprocessor
    ) -> bool:
        """Checks whether a given dataset-preprocessor-combination exists as precomputed file.
        Args:
            dataset_name (str): Matbench dataset name.
            preprocessor (CrystalPreprocessor): Crystal preprocessor for graph creation.
        Returns:
            bool: Whether a given dataset-preprocessor-combination exists
        """
        preprocessor_hash = preprocessor.hash()
        dataset_cache_dir = self.cache_dir / dataset_name
        if not dataset_cache_dir.is_dir():
            return False
        hdf_file = dataset_cache_dir / (preprocessor_hash + ".h5")
        return hdf_file.is_file()

    def get_dataset_file(
        self,
        dataset_name: str,
        preprocessor: CrystalPreprocessor,
        task_name: str = "training",
        processes=2,
        batch_size=500,
        id_prop_file="dataset/id_prop.csv",
    ):
        """Returns the file path for a dataset-preprocessor-combination.
        If the dataset-preprocessor-combination was used before it returns the cached version.
        If not it first applies preprocessing to all crystals to create the graph dataset.
        Args:
            dataset_name (str): Matbench dataset name.
            preprocessor (CrystalPreprocessor): Crystal preprocessor for graph creation.
            processes (int, optional): How many parallel processes to use for graph dataset creation. Defaults to 5.
            batch_size (int, optional): How big batches are for graph dataset creation.
                On each batch several processes are started in parallel,
                after all parallel processes are done the results are written to the file,
                before the next batch is processed.
                Defaults to 1000.
        Returns:
            Path: The path to the HDF file which contains preprocessed crystals.
        """
        preprocessor_hash = preprocessor.hash()
        dataset_cache_dir = self.cache_dir / dataset_name / task_name
        if not dataset_cache_dir.is_dir():
            dataset_cache_dir.mkdir(exist_ok=True, parents=True)
        hdf_file = dataset_cache_dir / (preprocessor_hash + ".h5")
        if hdf_file.is_file():
            print("HDF file already exists... Loading...")
            return hdf_file

        df_full = pd.read_csv(id_prop_file, index_col=[0])
        if task_name == "training":
            df_full = df_full[df_full[dataset_name] != "test"]
        elif task_name == "test":
            df_full = df_full[df_full[dataset_name] == "test"]
        else:
            raise ValueError("task_name should either be 'training' or 'test'.")
        full_dataset = pd.Series(
            index=df_full["material_id"].tolist(),
            data=[Structure.from_dict(eval(s)) for s in df_full["structure"]],
        )

        hdf_file_ = create_graph_dataset(
            self.crystal_iterator(full_dataset),
            preprocessor,
            hdf_file,
            additional_graph_attributes=["dataset_id"],
            processes=processes,
            batch_size=batch_size,
        )
        with open(str(hdf_file) + ".json", "w") as meta_data_file:
            json.dump(preprocessor.get_config(), meta_data_file)
        return hdf_file_


class PreprocessorWrapper:
    """Callable that modifies the behaviour of CrystalProcessors to include extra global graph attributes.
    Returns:
        MultiDiGraph: Crystal graph with extra global graph attributes.
    """

    def __init__(self, preprocessor, additional_graph_attributes=[]):
        self.preprocessor = preprocessor
        self.additional_graph_attributes = additional_graph_attributes

    def __call__(self, crystal: MultiDiGraph | Structure):
        graph = self.preprocessor(crystal)
        for attribute in self.additional_graph_attributes:
            setattr(graph, attribute, getattr(crystal, attribute))
        return graph


def batcher(iterable, batch_size):
    """Creates batches for iterables"""
    iterator = iter(iterable)
    while batch := list(islice(iterator, batch_size)):
        yield batch


def create_graph_dataset(
    crystals: Iterable[Structure],
    preprocessor: CrystalPreprocessor,
    out_file: Path,
    additional_graph_attributes=[],
    processes=1,
    batch_size=10,
) -> Path:
    """Creates a HDF file containing crystal graphs.
    Args:
        crystals (Union[Iterable[MetaDataWrapper], Iterable[Structure]]): Iterable of pymatgen structures.
        preprocessor (CrystalPreprocessor): The preprocessor to use for crystal graph creation.
        out_file (Path): The file to write the crystal graphs to.
        additional_graph_attributes (list, optional): Strings of additional graph attributes to include in the HDF file.
            Defaults to [].
        processes (int, optional): How many processes to use for preprocessing. Defaults to None.
        batch_size (int, optional): The batch size for parallel preprocessing. Defaults to 1000.
    """
    worker = PreprocessorWrapper(
        preprocessor, additional_graph_attributes=additional_graph_attributes
    )
    with h5py.File(str(out_file), "w") as f:
        with Pool(processes) as p:
            for batch in batcher(crystals, batch_size):
                graphs = p.map(worker, batch)
                try:
                    graphlist = GraphList.from_nx_graphs(
                        graphs,
                        node_attribute_names=preprocessor.node_attributes,
                        edge_attribute_names=preprocessor.edge_attributes,
                        graph_attribute_names=preprocessor.graph_attributes
                        + additional_graph_attributes,
                    )
                except KeyError as e:
                    print(batch)
                    print(graphs[0])
                    raise e
                # Append graphlist to HDF5 file
                HDFGraphList.from_graphlist(f, graphlist)
            f.attrs["preprocessor_config"] = json.dumps(
                preprocessor.get_config(), indent=2
            )
    return out_file


if __name__ == "__main__":
    # split = "distribution_125"
    # task_name = "test"

    for split in ["distribution_125", "distribution_250", "random_125", "random_250"]:
        for task_name in ["test", "training"]:
            # preproc = KNNAsymmetricUnitCell(k=24)
            preproc = VoronoiAsymmetricUnitCell(1e-6)

            shg_data = SHGDataset(cache_dir="dataset_graphs")

            preprocessed_crystals_file = shg_data.get_dataset_file(
                dataset_name=split,
                task_name=task_name,
                preprocessor=preproc,
                processes=2,
                batch_size=3000,
            )
