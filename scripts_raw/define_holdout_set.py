def main():
    from pathlib import Path

    import json
    from shg_ml_benchmarks import utils

    def save_ids(id_holdout, id_validation, include_val, strategy, n_holdout):
        path_id = f"../data/holdout_id_{strategy}_{n_holdout}.json"
        if not Path(path_id).exists():
            with open(path_id, "w") as f:
                json.dump(id_holdout, f)
        if include_val:
            path_id = f"../data/validation_id_{strategy}_{n_holdout}.json"
            if not Path(path_id).exists():
                with open(path_id, "w") as f:
                    json.dump(id_validation, f)

    # Equi-distribution per bin
    strategy = "distribution"
    n_holdout=125
    include_val = True
    id_holdout, id_validation = utils.get_holdout_validation_set(
        n_holdout=n_holdout,
        strategy_holdout=strategy,
        include_validation=include_val,
    )
    save_ids(id_holdout, id_validation, include_val, strategy, n_holdout)


    strategy = "distribution"
    n_holdout=250
    include_val = True
    id_holdout, id_validation = utils.get_holdout_validation_set(
        n_holdout=n_holdout,
        strategy_holdout=strategy,
        include_validation=include_val,
    )
    save_ids(id_holdout, id_validation, include_val, strategy, n_holdout)

    # Random sampling
    strategy = "random"
    n_holdout=125
    include_val = True
    id_holdout, id_validation = utils.get_holdout_validation_set(
        n_holdout=n_holdout,
        strategy_holdout=strategy,
        include_validation=include_val,
    )
    save_ids(id_holdout, id_validation, include_val, strategy, n_holdout)

    strategy = "random"
    n_holdout=250
    include_val = True
    id_holdout, id_validation = utils.get_holdout_validation_set(
        n_holdout=n_holdout,
        strategy_holdout=strategy,
        include_validation=include_val,
    )
    save_ids(id_holdout, id_validation, include_val, strategy, n_holdout)

if __name__=="__main__":
    main()
