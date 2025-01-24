The dataframe ```df_rot_ieee_pmg``` has a lot of columns, but the main ones are:

- dijk --> the full SHG tensor
- epsij --> the full dielectric tensor
- structure --> the structure in a dict format (from the MP or Alexandria)
- dKP --> the Kurtz-Perry coefficient (target property)

and their "rot" counterpart which correspond to the conventional form of the dijk tensor. dKP_rot should not be different than dKP since it only entails a rotation, but sometimes it can be, due to a potential symmetrization of the tensor wrt. the structure.

- src_bandgap --> the PBE gap as obtained from the MP or Alexandria
- n the refractive index from epsij (might require a verification bc I don't think I diagonalized epsij before calculating n... not sure anymore)

This dataset will be updated with columns containing the dijk forced into the conventional form (so the components that should be zero will be put to 0) and the corresponding dKP will be included
