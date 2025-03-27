import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUTLIERS = ["mp-622018", "mp-13032", "mp-28264", "mp-13150", "mp-604884", "mp-1227604"]

colours = plt.cm.Dark2.colors
alexandria_col = colours[2]
gnome_col = colours[1]
dft_col = colours[0]

plt.rcParams["font.family"] = "Liberation Sans"

computed_df = pd.read_pickle("../../data/df_rot_ieee_pmg.pkl.gz")
computed_df = computed_df[~computed_df.index.isin(OUTLIERS)]

alex_original = pd.read_pickle("../../data/df_agm_nov24_std.pkl.gz")

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

# Filter pred df by original index
alexandria_pred_df = alexandria_pred_df[
    ~alexandria_pred_df.index.isin(alex_original.index)
]
print(
    f"Alexandria reduced to {alexandria_pred_df.shape} from {alexandria_target_df.shape}"
)
alexandria_target_df = alexandria_target_df[
    ~alexandria_target_df.index.isin(alex_original.index)
]
alexandria_target_df = alexandria_target_df[
    ~alexandria_target_df["formula_reduced"].isin(computed_df["formula_reduced"])
]

fig, axes = plt.subplots(1, 2, figsize=(8, 3))

# ax.errorbar(
#    gnome_target_df["src_bandgap"],
#    gnome_pred_df["dKP_pred"],
#    yerr=gnome_pred_df["dKP_unc"],
#    alpha=0.2,
#    c=gnome_col,
#    fmt="none",
#    rasterized=True
# )
axes[0].scatter(
    gnome_target_df["src_bandgap"],
    gnome_pred_df["dKP_pred"],
    alpha=1,
    c=gnome_col,
    # s=5,
    edgecolor="none",
    # lw=0,
    # edgecolor="black",
    marker="v",
    zorder=1e10,
    label="GNoME (MODNet)",
    rasterized=True,
)

axes[0].set_xlim(0, 10)
axes[1].set_xlim(0, 10)
axes[1].scatter(
    alexandria_target_df["src_bandgap"],
    alexandria_pred_df["dKP_pred"],
    alpha=1,
    # s=5,
    zorder=1e11,
    edgecolor="none",
    marker="^",
    # lw=0,
    c=alexandria_col,
    label="Alexandria (MODNet)",
    rasterized=True,
)

axes[0].set_yscale("log")
axes[1].set_yscale("log")

axes[0].scatter(
    computed_df["src_bandgap"],
    computed_df["dKP_full_neum"],
    alpha=0.75,
    # lw=0.5,
    zorder=0,
    marker="+",
    # s=5,
    # edgecolor="k",
    c=dft_col,
    # edgecolor="black",
    label="SHG-25 (DFT)",
    rasterized=True,
)

axes[1].scatter(
    computed_df["src_bandgap"],
    computed_df["dKP_full_neum"],
    alpha=0.75,
    # lw=0.5,
    zorder=0,
    marker="+",
    # edgecolor="black",
    # s=2,
    c=dft_col,
    # edgecolor="black",
    label="SHG-25 (DFT)",
    rasterized=True,
)

x_pareto = np.linspace(0, 10, 1000)
a = 245.338
b = -0.7378

# axes[0].plot(
#    x_pareto,
#    a * np.exp(b * x_pareto),
#    c="k",
#    lw=1.5,
#    ls="--",
#    label="$T_0$ Fitted Pareto front",
#    zorder=1e20,
# )
# axes[1].plot(
#    x_pareto,
#    a * np.exp(b * x_pareto),
#    c="k",
#    lw=1.5,
#    ls="--",
#    label="$T_0$ Fitted Pareto front",
#    zorder=1e20,
# )

axes[0].set_ylim(1e-2, 5e2)
axes[1].set_ylim(1e-2, 5e2)

axes[0].set_xlabel("PBE $E_g$ (eV)")
axes[0].set_ylabel(r"$d_\text{KP}$ (pm/V)")
axes[0].legend(loc="upper right")
axes[1].set_xlabel("PBE $E_g$ (eV)")
axes[1].set_ylabel(r"$d_\text{KP}$ (pm/V)")
axes[1].legend(loc="upper right")

axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)

plt.tight_layout()

plt.savefig("figs/gnome-alex-explore.pdf", dpi=300)
