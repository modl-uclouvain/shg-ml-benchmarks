
from shg_ml_benchmarks.utils import SHG_BENCHMARK_SPLITS, load_holdout, load_train, load_full

# 1. Load the full training 

df_full = load_full()
for split in SHG_BENCHMARK_SPLITS:

