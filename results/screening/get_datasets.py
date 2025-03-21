import abc
import datetime
import json
import os
from pathlib import Path

import pandas as pd
from optimade.adapters.structures import Structure as OptimadeStructure
from optimade.client import OptimadeClient


class Dataset(abc.ABC):
    """The Dataset object provides a container for OPTIMADE structures
    that are decorated with the same set of properties.

    It is assumed that each dataset can fit into memory.

    """

    id: str
    """A tag for the dataset, e.g. "Naccarato2019" or "MP2023"."""

    id_prefix: str
    """The prefix for the OPTIMADE IDs in the dataset."""

    references: list[dict] | None
    """Bibliographic references for the dataset, if available."""

    metadata: dict
    """Any additional metadata for the dataset, will be saved in the dataset directory as meta.json"""

    data: list[OptimadeStructure]
    """A list of OPTIMADE structures, decorated with target properties, where available."""

    properties: dict[str, str]
    """A dictionary mapping from a property name to the column name in the dataset."""

    targets: set[str] | None = None
    """A set of target properties for the dataset."""

    def __init__(self):
        if getattr(self, "id", None) is None:
            self.id = self.__class__.__name__.replace("Dataset", "")

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    @property
    def as_df(self):
        df = pd.DataFrame(
            [{"id": entry.id, **entry.as_dict["attributes"]} for entry in self.data]
        )
        return df.set_index("id")

    @property
    def property_df(self):
        df = pd.DataFrame(
            [
                {
                    "id": entry.id,
                    **{
                        k: getattr(entry.attributes, alias, None)
                        for k, alias in self.properties.items()
                    },
                }
                for entry in self.data
            ]
        )
        return df.set_index("id")

    @property
    def structure_df(self):
        """Returns a dataframe with the pymatgen structure and the target properties
        defined by the dataset."""
        df = pd.DataFrame(
            [
                {
                    "id": entry.id,
                    "structure": entry.as_pymatgen,
                }
                for entry in self.data
            ]
        )
        return df.set_index("id")

    @classmethod
    def load(cls) -> "Dataset":
        filename = Path("data") / f"{cls.id}" / f"{cls.id}.jsonl"
        self = cls()

        if filename.exists():
            with open(filename) as f:
                self.data = [OptimadeStructure(json.loads(s)) for s in f.readlines()]

            return self

        return None  # type: ignore

    def save(self):
        dataset_dir = Path("data") / f"{self.id}"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        with open(dataset_dir / f"{self.id}.jsonl", "w") as f:
            for entry in self.data:
                f.write(entry.as_json + "\n")

        with open(dataset_dir / "meta.json", "w") as f:
            json.dump(self.metadata, f)


class MP2023Dataset(Dataset):
    properties = {
        "hull_distance": "_mp_energy_above_hull",
        "band_gap": "_mp_band_gap",
        "refractive_index": "_mp_refractive_index",
    }

    id: str = "MP2023"

    id_prefix: str = "https://optimade.materialsproject.org/v1/structures"

    @classmethod
    def load(cls) -> "MP2023Dataset":
        """Use the MP API to load a dataset of materials with a band gap and energy above hull,
        then convert these to the OPTIMADE format and alias some properties.

        """

        self = super().load()
        if self is not None:
            return self  # type: ignore

        print(
            f"Previously created dataset not found; loading {cls.id} dataset from scratch"
        )

        from mp_api.client import MPRester

        with MPRester(os.environ["MP_API_KEY"]) as mpr:
            docs = mpr.summary.search(
                band_gap=(0.05, None),
                energy_above_hull=(0, 0.025),
            )

        optimade_docs = []

        for doc in docs:
            structure = OptimadeStructure.ingest_from(doc.structure)
            optimade_doc = structure.as_dict
            optimade_doc["attributes"]["_mp_band_gap"] = doc.band_gap
            optimade_doc["attributes"]["_mp_energy_above_hull"] = (
                doc.energy_above_hull,
            )
            optimade_doc["attributes"]["_mp_structure_origin"] = (
                "experimental" if not doc.theoretical else "predicted"
            )
            optimade_doc["attributes"]["_mp_formation_energy_per_atom"] = (
                doc.formation_energy_per_atom
            )
            optimade_doc["attributes"]["_mp_refractive_index"] = doc.n
            optimade_doc["id"] = cls.id_prefix + "/" + str(doc.material_id)
            optimade_doc["attributes"]["immutable_id"] = str(doc.material_id)
            optimade_docs.append(optimade_doc)

        self = cls()
        self.data = [OptimadeStructure(doc) for doc in optimade_docs]
        self.metadata = {"ctime": datetime.datetime.now().isoformat()}
        self.save()
        return self


class OptimadeDataset(Dataset):
    base_url: str
    filter: str
    response_fields: list[str] | None = None

    @classmethod
    def load(cls) -> "OptimadeDataset":
        self = super().load()
        if self is not None:
            return self  # type: ignore

        print(
            f"Previously created dataset not found; loading {cls.id} dataset from scratch"
        )

        client = OptimadeClient(cls.base_url, silent=False, max_results_per_provider=0)
        results = client.get(cls.filter, response_fields=cls.response_fields)

        self = cls()
        self.data = [
            OptimadeStructure(s)
            for s in results["structures"][cls.filter][cls.base_url]["data"]
        ]

        for ind, _ in enumerate(self.data):
            self.data[
                ind
            ].entry.id = f"{self.base_url}/v1/structures/{self.data[ind].entry.id}"

        self.metadata = {"ctime": datetime.datetime.now().isoformat()}

        self.save()
        return self


class Alexandria2024Dataset(OptimadeDataset):
    properties = {
        "band_gap": "_alexandria_band_gap",
        "hull_distance": "_alexandria_hull_distance",
    }
    id: str = "Alexandria2024"
    base_url: str = "https://alexandria.icams.rub.de/pbe"
    filter: str = "_alexandria_band_gap > 0.05 AND _alexandria_hull_distance <= 0.025"


class GNome2024Dataset(OptimadeDataset):
    properties = {}
    id: str = "GNome2024"
    base_url: str = "https://optimade-gnome.odbx.science"
    filter: str = ""


class odbx2024Dataset(OptimadeDataset):
    properties = {"hull_distance": "_odbx_thermodynamics.hull_distance"}
    id: str = "odbx2024"
    base_url: str = "https://optimade.odbx.science"
    filter: str = "_odbx_thermodynamics.hull_distance < 0.025"


class GNome2025Dataset(OptimadeDataset):
    properties = {"band_gap": "_gnome_bandgap", "hull_distance": "_gnome_hull_distance"}
    id = "GNome2025"
    base_url: str = "https://optimade-gnome.odbx.science"
    filter = '_gnome_bandgap > 0.05 AND _gnome_space_group_it_number HAS ANY 2,10,11,12,13,14,15,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,83,84,85,86,87,88,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,147,148,162,163,164,165,166,167,175,176,191,192,193,194,200,201,202,203,204,205,206,221,222,223,224,225,226,227,228,229,230 AND nperiodic_dimensions = 3 AND NOT elements HAS ANY "La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr"'


class Alexandria2025(OptimadeDataset):
    properties = {
        "band_gap": "_alexandria_band_gap",
        "hull_distance": "_alexandria_hull_distance",
    }
    id: str = "Alexandria2025"
    base_url: str = "https://alexandria.icams.rub.de/pbe"
    filter: str = '_alexandria_band_gap > 0.05 AND _alexandria_hull_distance <= 0.05 AND NOT _alexandria_space_group HAS ANY 2,10,11,12,13,14,15,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,83,84,85,86,87,88,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,147,148,162,163,164,165,166,167,175,176,191,192,193,194,200,201,202,203,204,205,206,221,222,223,224,225,226,227,228,229,230 AND nperiodic_dimensions = 3 AND NOT elements HAS ANY "La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr"'
    response_fields = [
        "id",
        "cartesian_site_positions",
        "_alexandria_space_group",
        "_alexandria_band_gap",
        "_alexandria_hull_distance",
        "species",
        "species_at_sites",
        "lattice_vectors",
        "chemical_formula_reduced",
        "nperiodic_dimensions",
        "nsites",
        "structure_features",
        "last_modified",
    ]


if __name__ == "__main__":
    gnome = GNome2025Dataset.load()
    alexandria = Alexandria2025.load()
