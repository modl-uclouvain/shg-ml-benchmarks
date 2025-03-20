# /// script
# requires-python = ">=3.11"
# dependencies = ["pymatgen >= 2022", "shg_ml_benchmarks @ git+https://github.com/modl-uclouvain/shg-ml-benchmarks"]
# ///

from shg_ml_benchmarks import run_benchmark
from shg_ml_benchmarks.utils import SHG_BENCHMARK_SPLITS, DummyModel

if __name__ == "__main__":
    dummy = DummyModel(mode="median")

    def train_fn(structures, target):
        dummy.train(structures, target)
        return dummy

    def predict_fn(model, structures):
        return model.predict(structures)

    for task in SHG_BENCHMARK_SPLITS:
        run_benchmark(
            model=dummy,
            task=task,
            train_fn=train_fn,
            predict_fn=predict_fn,
            predict_individually=True,
        )
