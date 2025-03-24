import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

colours = plt.cm.Dark2.colors
alexandria_col = colours[0]
gnome_col = colours[1]
dft_col = colours[2]

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

fig, ax = plt.subplots(figsize=(3.5, 2.5))

# ax.errorbar(
#    gnome_target_df["src_bandgap"],
#    gnome_pred_df["dKP_pred"],
#    yerr=gnome_pred_df["dKP_unc"],
#    alpha=0.2,
#    c=gnome_col,
#    fmt="none",
#    rasterized=True
# )
ax.scatter(
    gnome_target_df["src_bandgap"],
    gnome_pred_df["dKP_pred"],
    alpha=0.75,
    c=gnome_col,
    s=5,
    lw=0,
    # edgecolor="black",
    marker="v",
    zorder=1e10,
    label="GNome (MODNet)",
    rasterized=True,
)

# ax.errorbar(
#    alexandria_target_df["src_bandgap"],
#    alexandria_pred_df["dKP_pred"],
#    yerr=alexandria_pred_df["dKP_unc"],
#    alpha=0.2,
#    c=alexandria_col,
#    fmt="none",
#    rasterized=True
# )
ax.set_xlim(0, 10)
ax.set_ylim(-1, 200)
ax.scatter(
    alexandria_target_df["src_bandgap"],
    alexandria_pred_df["dKP_pred"],
    alpha=0.75,
    s=5,
    zorder=1e9,
    marker="^",
    lw=0,
    c=alexandria_col,
    label="Alexandria (MODNet)",
    rasterized=True,
)

ax.scatter(
    computed_df["src_bandgap"],
    computed_df["dKP_full_neum"],
    zorder=1e11,
    alpha=0.75,
    # lw=0.5,
    marker=".",
    s=2,
    c=dft_col,
    # edgecolor="black",
    label="SHG-25 (DFT)",
    rasterized=True,
)

x_pareto = np.linspace(0, 10, 1000)
a = 245.338
b = -0.7378

ax.plot(
    x_pareto,
    a * np.exp(b * x_pareto),
    c="k",
    lw=1.5,
    ls="--",
    label="$T_0$ Fitted Pareto front",
    zorder=1e20,
)

ax.set_xlabel("PBE $E_g$ (eV)")
ax.set_ylabel(r"$d_\text{KP}$ (pm/V)")
ax.legend()

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

plt.savefig("figs/gnome-alex-explore.pdf", dpi=300)
