import json
from pathlib import Path

import monty
import pandas as pd
import tqdm
from optimade.adapters.structures import Structure as OptimadeStructure


def transform_to_df(fname):
    data = []
    with open(fname) as f:
        for line in tqdm.tqdm(f):
            if line:
                optimade = OptimadeStructure(json.loads(line))
                pmg = optimade.as_pymatgen
                try:
                    gap = optimade.attributes._gnome_bandgap
                except Exception:
                    gap = optimade.attributes._alexandria_band_gap
                data.append(
                    {
                        "id": optimade.id,
                        "structure": json.dumps(pmg, cls=monty.json.MontyEncoder),
                        "src_bandgap": gap,
                        "formula_reduced": pmg.composition.reduced_formula,
                    }
                )
    df = pd.DataFrame(
        data, columns=["id", "structure", "src_bandgap", "formula_reduced"]
    )
    df.set_index("id", inplace=True)
    df.to_pickle(fname.with_suffix(".pkl.gz"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Filter structures by Laue symmetry")
    parser.add_argument(
        "db_name",
        type=str,
        help="Database name to filter, path will be constructed as `data/<db-name>/<db-name>.jsonl`",
    )
    args = parser.parse_args()
    transform_to_df(
        Path(__file__).parent / "data" / args.db_name / (args.db_name + ".jsonl")
    )
