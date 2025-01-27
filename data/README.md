# OneDrive folder

All data can be found in the re2fractive_shg/data folder on Victor's OneDrive. It can be accessed and modified [here](https://uclouvain-my.sharepoint.com/:f:/g/personal/victor_trinquet_uclouvain_be/EqB4w3awJztDkjb1Lc-Z6jgBQMfMUgHG_TPW_Qfq6o0Xcw?e=EoKG1Y) if you have been granted the permission.

# df_rot_ieee_pmg

The dataframe ```df_rot_ieee_pmg``` has a lot of columns, but the main ones are:

- dijk --> the full SHG tensor
- epsij --> the full dielectric tensor
- structure --> the structure in a dict format (from the MP or Alexandria)
- dKP --> the Kurtz-Perry coefficient (target property)

and their "rot" counterpart which correspond to the conventional form of the dijk tensor. dKP_rot should not be different than dKP since it only entails a rotation, but sometimes it can be, due to a potential symmetrization of the tensor wrt. the structure.

- dijk_full --> the full SHG tensor, BUT only non-zero components by symmetry have been kept, while the others have been put to exactly 0.
- dKP_full --> the corresponding Kurtz-Perry coefficient, can be different than dKP and dKP_rot

Other useful columns are:

- src_bandgap --> the PBE gap as obtained from the MP or Alexandria
- n the refractive index from epsij (might require a verification bc I don't think I diagonalized epsij before calculating n... not sure anymore)

NB: This dataset will be updated with columns containing the dijk forced to respect Neumann's principle (~200 entries do not respect it up a certain number of decimals, might be due to a not so exact rotation, but difficult to find) and the corresponding dKP.
