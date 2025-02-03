# Data on Github

## holdout_id_{strategy}_{number}.json
These files contain the ids for holdout sets to be used in the benchmarking. The sampling strategy is written in the file name as well as the number of instances in this set. They were obtained via the script ```define_holdout_set.py``` in ```../scripts_raw```. IT IS IMPORTANT TO NOTE THAT THEY WERE GENERATED FROM THE "is_unique_here == True" SUBSET OF df_rot_ieee_pmg.

# OneDrive folder

All data can be found in the re2fractive_shg/data folder on Victor's OneDrive. It can be accessed and modified [here](https://uclouvain-my.sharepoint.com/:f:/g/personal/victor_trinquet_uclouvain_be/EqB4w3awJztDkjb1Lc-Z6jgBQMfMUgHG_TPW_Qfq6o0Xcw?e=EoKG1Y) if you have been granted the permission. For some dataset, the scripts used to construct them can be found in ```scripts_raw```.

## df_rot_ieee_pmg

The dataframe ```df_rot_ieee_pmg``` has been obtained via the ```scripts_raw/find_conventional_dijk.py```. It has a lot of columns, but the main ones are:

- is_unique_here --> indicates that the compound is unique wrt. pmg StructureMatcher, TO BE USED!

- dijk --> the full SHG tensor
- epsij --> the full dielectric tensor
- structure --> the structure in a dict format (from the MP or Alexandria)
- dKP --> the Kurtz-Perry (KP) coefficient (target property)

and their "rot" counterpart which correspond to the conventional form of the dijk tensor. dKP_rot should not be different than dKP since it only entails a rotation, but sometimes it can be, due to a potential symmetrization of the tensor wrt. the structure.

- dijk_full --> the full SHG tensor, BUT only non-zero components by symmetry have been kept, while the others have been put to exactly 0.
- dKP_full --> the corresponding Kurtz-Perry coefficient, can be different than dKP and dKP_rot
- dijk_full_neum --> same as dijk_full BUT with Neumann's principle enforced, because some of the non-zero components were not equal although they should be by Neumann's principle. These values have been set to their average.
- dKP_full_neum --> the corresponding KP coefficient

Other useful columns are:

- src_bandgap --> the PBE gap as obtained from the MP or Alexandria
- n the refractive index from epsij (might require a verification bc I don't think I diagonalized epsij before calculating n... not sure anymore)

## df_outputs.pkl.gz

This dataset is the direct result of the post-processing of the DFPT workflow output. It is similar to df_rot_ieee_pmg, but it contains less columns (like the rot, full, and full_neum) and more data. The additional entries have been deemed as being outliers and have thus never been used for ML, etc. The definition of an outlier is the following:

- dKP > 170
- dKP <= 1e-5
- src_bandgap < 1e-5
- n > 20

## mod.data_dKP_fastfeat_polariz_n_Eg_pgnn_featselec_v20.gz

This MODData contains the latest dataset (thus corresponding to df_rot_ieee_pmg) and its features containing:
- the "fast" features from MODNet 2024 FastFeaturizer
- the Magpie_Polarization feature has been added (in the last AL) iteration because physically it makes sense
- the prediction of the refractive index and its uncertainty by the last re2fractive MODNet model
- the source bandgap, i.e., from the MP or Alexandria (and its uncertainty = 0)
- the pGNN features (mmv1, ofm, mvl32)

The features selection has been run and stored in self.optimal_features wrt. dKP as target.
