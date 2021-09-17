"""
Test DISCOVER algorithm.

- create distance matrix
- apply densMAP
- create clusters via HDBSCAN*
- search for interesting materials, for example:
     - high-target/low-density
     - materials with high-target surrounded by materials with low targets

Run using elm2d_ environment.

# Stick with ~10k elasticity datapoints
# Perform UMAP and HDBSCAN
for cluster in clusters:
    for i in [0, 1, 5, 10]:
        n = ceil(len(cluster) - i, 0)
        clustertmp = clusters
        remove cluster[0:n] from clustertmp
        train crabnet
        predict on removed cluster, store MAE
        calculate MAE for targets above some threshold
        test distance to pareto front for "kept" (?) cluster points for various metrics
        calculate the distribution of distances to the pareto front
# FIGURE: cluster-wise distributions of target values (6 clusters with highest (mean?) target value) 2x3 tile
# TODO: parameter - highest mean vs. highest single target value
# SUPP-FIGURE: cluster-wise distributions of target values for all clusters (Nx3)
# FIGURE: cluster-wise cross-val parity plots (6 clusters with highest (mean?) target value) - 2x3 tile
# SUPP-FIGURE: cluster-wise cross-val parity plots for all clusters (Nx3)

Created on Mon Sep  6 23:15:27 2021.

@author: sterg
"""
# %% Imports
from os.path import join, expanduser

import pandas as pd

from discover import Discover

disc = Discover()

# %% import validation data
data_dir = expanduser(
    join(
        "~",
        "Documents",
        "GitHub",
        "sparks-baird",
        "ElM2D",
        "CrabNet",
        "model_predictions",
    )
)
fpath = join(data_dir, "elasticity_val_output.csv")
# fpath = "C:/Users/sterg/Documents/GitHub/sparks-baird/ElM2D/CrabNet/model_predictions/example_materials_property_val_output.csv"
val_df = pd.read_csv(fpath)

n = 100
tmp_df = val_df.iloc[:n, :]
# formulas = val_df["composition"][:n]
# target = val_df["pred-0"][:n]

# %% fit
disc.fit(tmp_df)


# %% CODE GRAVEYARD
# datapath = join("ael_bulk_modulus_voigt", "train.csv")
# datapath = "train-debug.csv"
# df = pd.read_csv(datapath)
# formulas = df["formula"]
# target = df["target"]

# fig2 = px.line(x=[0, max(neigh_avg_targ)], y=[0, max(target)])
# fig3 = go.Figure(data=fig1.data + fig2.data)
# fig3.show()
# px.xlabel(r"neigh_avg_targ$^{-1}$ (GPa)$^{-1}$")
# plt.ylabel("target (GPa)")
# plt.show()


# pareto_ind = np.nonzero(
#     is_pareto_efficient_simple(np.array([1 / neigh_avg_targ, target]).T)
# )


# marker=dict(
#     opacity=0.5,
#     size=12,
#     line=dict(
#         color='Black',
#         width=1,
#         ),
#     )

# fig.add_trace(
#     go.Scatter(
#         mode='markers',
#         x=proxy.iloc[pareto_ind],
#         y=target.iloc[pareto_ind],
#         hover_data=pf_hover_data,
#         name="Pareto Front",
#         showlegend=True,
#         )
# )

# import sys

# sys.path.append("C:/Users/sterg/Documents/GitHub/sparks-baird/ElM2D/ElM2D")  # noqa
# import numpy as np

# from ElM2D.helper import Timer
# from ElM2D import ElM2D
# from utils.pareto import pareto_plot

# mapper = ElM2D(target="cuda")  # type: ignore
