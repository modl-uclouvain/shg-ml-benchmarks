"""Some APIs do not allow you to filter on negative regexps needed
to check centrosymmetry directly, so here we filter them locally.
"""

import json
from pathlib import Path

from optimade.adapters.structures import Structure as OptimadeStructure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


def filter_by_laue(fname):
    laue_path = fname.with_name(fname.stem + "_laue.jsonl")
    non_laue_path = fname.with_name(fname.stem + "_non_laue.jsonl")

    with (
        open(fname) as f,
        open(laue_path, "w") as laue_file,
        open(non_laue_path, "w") as non_laue_file,
    ):
        laue = 0
        non_laue = 0
        while f:
            s = f.readline()
            if not s:
                break
            opt = OptimadeStructure(json.loads(s))
            pmg = opt.as_pymatgen
            spg = SpacegroupAnalyzer(pmg)
            if spg.is_laue():
                laue_file.write(s)
                laue += 1
            else:
                non_laue_file.write(s)
                non_laue += 1
            print(
                f"{laue=}\t\t\t{non_laue=}\t\t{100 * laue / (laue + non_laue)} %",
                end="\r",
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Filter structures by Laue symmetry")
    parser.add_argument(
        "db_name",
        type=str,
        help="Database name to filter, path will be constructed as `data/<db-name>/<db-name>.jsonl`",
    )
    args = parser.parse_args()
    filter_by_laue(
        Path(__file__).parent / "data" / args.db_name / (args.db_name + ".jsonl")
    )
