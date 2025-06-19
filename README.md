# Benchmarking ML models for prediction of second-harmonic generation coefficients

The aim of the project is to use active learning to screen hypothetical compounds from OPTIMADE-enabled databases for those with high SHG coefficients for a given band gap.
This procedure generated a DFT-computed SHG dataset, SHG-25, which was then used to benchmark a series of ML models across different classes.
The results and methods of this benchmarking are found in this repository under `./benchmarks`, along with a series of utilities in the `shg-ml-benchmarks` Python package under `./src`.

This repository accompanies the preprint:

> V. Trinquet, M. L. Evans, G-M.R. Rignanese, *Accelerating the discovery of high-performance nonlinear optical materials using active learning and high-throughput screening* (2025) arXiV:[2504.01526](https://arxiv.org/abs/2504.01526) DOI: [10.48550/arXiv.2504.01526](https://doi.org/10.48550/arXiv.2504.01526)

The resulting dataset is archived on the [Materials Cloud Archive](https://archive.materialscloud.org/):

> V. Trinquet, M. L. Evans, G-M.R. Rignanese, *Accelerating the discovery of high-performance nonlinear optical materials using active learning and high-throughput screening* (2025) DOI: [10.24435/materialscloud:wk-qm](https://doi.org/10.24435/materialscloud:wk-qm)

with associated OPTIMADE API access at https://optimade.materialscloud.org/archive/wk-qm/v1/info.
