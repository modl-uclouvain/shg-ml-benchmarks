import matplotlib.pyplot as plt
import pandas as pd

pred_df = pd.read_json("data/GNome2025/df_dKP_pred_unc_gnome2025.json.gz")
target_df = pd.read_pickle("data/GNome2025/GNome2025.pkl.gz")
fig, ax = plt.subplots()

computed_df = pd.read_pickle("../../data/df_rot_ieee_pmg.pkl.gz")

ax.errorbar(
    target_df["src_bandgap"],
    pred_df["dKP_pred"],
    yerr=pred_df["dKP_unc"],
    alpha=0.5,
    fmt="none",
)
ax.scatter(
    target_df["src_bandgap"],
    pred_df["dKP_pred"],
    alpha=0.5,
    c="k",
    s=2,
    zorder=1e10,
    label="GNome2025 (MODNet)",
)

ax.scatter(
    computed_df["src_bandgap"],
    computed_df["dKP_full_neum"],
    zorder=10,
    alpha=0.5,
    s=2,
    c="r",
    label="SHG-2k (DFT)",
)

ax.legend()

plt.savefig("figs/gnome-explore.pdf")
