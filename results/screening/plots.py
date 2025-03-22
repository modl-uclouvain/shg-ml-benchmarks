import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.family"] = "Liberation Sans"

computed_df = pd.read_pickle("../../data/df_rot_ieee_pmg.pkl.gz")

gnome_pred_df = pd.read_json("data/GNome2025/df_dKP_pred_unc_gnome2025.json.gz")
gnome_target_df = pd.read_pickle("data/GNome2025/GNome2025.pkl.gz")
alexandria_pred_df = pd.read_json(
    "data/Alexandria2025/df_dKP_pred_unc_alexandria2025.json.gz"
)
alexandria_target_df = pd.read_pickle("data/Alexandria2025/Alexandria2025.pkl.gz")
# Strip URL prefix from IDs to match pred df
alexandria_target_df.index = alexandria_target_df.index.str.split("/").str[-1]
gnome_target_df.index = gnome_target_df.index.str.split("/").str[-1]

gnome_pred_df.index = gnome_pred_df.index.str.split("gnome_").str[-1]
alexandria_pred_df.index = alexandria_pred_df.index.str.split("alexandria_").str[-1]

# Filter out any compositions already in the computed set, by matching formula_reduced string
gnome_pred_df = gnome_pred_df[
    ~gnome_target_df["formula_reduced"].isin(computed_df["formula_reduced"])
]
gnome_target_df = gnome_target_df[
    ~gnome_target_df["formula_reduced"].isin(computed_df["formula_reduced"])
]
print(f"GNome reduced to {gnome_pred_df.shape} from {gnome_target_df.shape}")

alexandria_pred_df = alexandria_pred_df[
    ~alexandria_target_df["formula_reduced"].isin(computed_df["formula_reduced"])
]
print(
    f"Alexandria reduced to {alexandria_pred_df.shape} from {alexandria_target_df.shape}"
)
alexandria_target_df = alexandria_target_df[
    ~alexandria_target_df["formula_reduced"].isin(computed_df["formula_reduced"])
]

fig, ax = plt.subplots()

ax.errorbar(
    gnome_target_df["src_bandgap"],
    gnome_pred_df["dKP_pred"],
    yerr=gnome_pred_df["dKP_unc"],
    alpha=0.2,
    c="blue",
    fmt="none",
)
ax.scatter(
    gnome_target_df["src_bandgap"],
    gnome_pred_df["dKP_pred"],
    alpha=0.5,
    c="blue",
    s=5,
    lw=0.5,
    edgecolor="black",
    zorder=1e10,
    label="GNome2025 (MODNet)",
)

ax.errorbar(
    alexandria_target_df["src_bandgap"],
    alexandria_pred_df["dKP_pred"],
    yerr=alexandria_pred_df["dKP_unc"],
    alpha=0.2,
    c="green",
    fmt="none",
)
ax.scatter(
    alexandria_target_df["src_bandgap"],
    alexandria_pred_df["dKP_pred"],
    alpha=0.5,
    s=5,
    zorder=1e9,
    lw=0.5,
    edgecolor="black",
    c="green",
    label="Alexandria2025 (MODNet)",
)

ax.scatter(
    computed_df["src_bandgap"],
    computed_df["dKP_full_neum"],
    zorder=1e11,
    alpha=0.5,
    lw=0.5,
    s=5,
    c="r",
    edgecolor="black",
    label="SHG-2k (DFT)",
)

ax.set_xlabel("PBE $E_g$ (eV)")
ax.set_ylabel(r"$d_\text{KP}$ (pm/V)")
ax.legend()

plt.savefig("figs/gnome-alex-explore.png", dpi=300)
