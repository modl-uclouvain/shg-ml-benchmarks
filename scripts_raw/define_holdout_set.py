def main():
    from pathlib import Path

    import json
    from shg_ml_benchmarks import utils

    # Equi-distribution per bin
    strategy = "distribution"
    n_holdout=500
    id_holdout = utils.get_holdout_set(
        n_holdout=n_holdout,
        strategy_holdout=strategy,
    )

    path_id = f"../data/holdout_id_{strategy}_{n_holdout}.json"
    if not Path(path_id).exists():
        with open(path_id, "w") as f:
            json.dump(id_holdout, f)

    # Random sampling
    strategy = "random"
    n_holdout=500
    id_holdout = utils.get_holdout_set(
        n_holdout=n_holdout,
        strategy_holdout=strategy,
    )

    path_id = f"../data/holdout_id_{strategy}_{n_holdout}.json"
    if not Path(path_id).exists():
        with open(path_id, "w") as f:
            json.dump(id_holdout, f)

if __name__=="__main__":
    main()
