This folder provides raw python scripts and notebooks used to process the data for example. They have not been cleaned for readability and such.

- SHG_Tensor_Func.py --> contains useful functions to manipulate the SHG tensor.
- find_conventional_dijk.ipynb --> rotates and cleans the SHG tensor to recover the conventional form wrt. space group, creates ```data/df_rot_ieee_pmg.pkl.gz```
- define_holdout_set.py --> defines holdout test set to be used in benchmarking
- describe_holdout_set.py --> investigates the distribution etc of the holdout test set defined
- SHG_Tensor_to_Conventional.py --> contains functions to find the conventional form of the SHG tensors, can be called with options (see top of the script and main function) to automatically find the conventional tensors
- active learning --> directory containing the notebooks used during the AL process, they are numbered to indicate the order in which they should be run
