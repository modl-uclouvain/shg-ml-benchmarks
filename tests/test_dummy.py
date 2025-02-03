from shg_ml_benchmarks.utils import BENCHMARKS_DIR


def test_dummy():
    from shg_ml_benchmarks import run_benchmark
    from shg_ml_benchmarks.utils import DummyModel

    dummy = DummyModel()

    def train_fn(df, target):
        model = dummy.train(df, target)
        return model

    def predict_fn(model, structure):
        return model.predict(structure)

    results = run_benchmark(
        model=dummy,
        task="random_250",
        train_fn=train_fn,
        predict_fn=predict_fn,
        write_results=True,
    )

    assert results["predictions"]
    assert results["metrics"]
    assert (
        BENCHMARKS_DIR / "mean_value" / "tasks" / "random_250" / "results.json"
    ).is_file()
