There are 3 kinds of folders for Matten:

- datasets --> contains the different subsets of our data, namely, holdout, validation, and training, which are used to train Matten and to benchmark him. The json files containing the Structures and the dijk have been obtained via the python script in it.
- scripts_* --> are the folders which were on the HPC and in which Matten was trained. The config yaml file is what differs between all of them.
- predict_* --> are the corresponding folders where a trained matten model is used to predict the corresponding holdout sets (via the predict_general.py script). The notebook stats.ipynb is used to generate simple binary plots.
