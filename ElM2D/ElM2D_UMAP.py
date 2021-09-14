"""
Materials discovery using Earth Mover's Distance, densMAP embeddings, and HDBSCAN*.

- create distance matrix
- apply densMAP
- create clusters via HDBSCAN*
- search for interesting materials, for example:
     - high-target/low-density
     - materials with high-target surrounded by materials with low targets

Created on Mon Sep  6 23:15:27 2021.

@author: sterg
"""
# import sys

# sys.path.append("C:/Users/sterg/Documents/GitHub/sparks-baird/ElM2D/ElM2D")  # noqa

# from os.path import join
from operator import attrgetter

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors

import umap
import hdbscan

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# from ElM2D.helper import Timer
from ElM2D import ElM2D
from helper import Timer

plt.rcParams["text.usetex"] = True

pio.renderers.default = "browser"

mapper = ElM2D(target="cuda")


# %% import validation data
fpath = "C:/Users/sterg/Documents/GitHub/sparks-baird/ElM2D/CrabNet/model_predictions/elasticity_val_output.csv"
val_df = pd.read_csv(fpath)

formulas = val_df["composition"]
target = val_df["pred-0"]

# %% Distance Matrix
with Timer("fit-wasserstein"):
    mapper.fit(formulas)
    dm_wasserstein = mapper.dm

# %% Nearest Neighbors within Radius
mean_wasserstein, std_wasserstein = (np.mean(dm_wasserstein), np.std(dm_wasserstein))
radius = mean_wasserstein - 2 * std_wasserstein
NN = NearestNeighbors(radius=radius, metric="precomputed")
NN.fit(dm_wasserstein)
neigh_dist, neigh_ind = NN.radius_neighbors()

neigh_target = np.array([target[ind] for ind in neigh_ind])
neigh_avg_targ = np.array([np.mean(t) if t != [] else np.nan for t in neigh_target])
num_neigh = np.array([len(ind) for ind in neigh_ind])

# fig = plt.figure()

# %% Pareto-esque Front
plt_df = pd.DataFrame(
    {
        "neigh_avg_targ": neigh_avg_targ,
        "target": target,
        "formulas": formulas,
        "diff": target - neigh_avg_targ,
    }
)
fig = px.scatter(
    plt_df,
    "neigh_avg_targ",
    "target",
    color="diff",
    color_continuous_scale=px.colors.sequential.Blackbody_r,
    hover_data=["formulas"],
)
mx = np.nanmax([neigh_avg_targ, target.to_numpy()])
fig.add_trace(go.Line(x=[0, mx], y=[0, mx], name="parity"))
fig.update_layout(xaxis=dict(autorange="reversed"), legend_orientation="h")
fig.show()
fig.write_html("pareto-front.html")

# %% UMAP
dens_lambda = 1

# fit for clustering
with Timer("fit-UMAP"):
    umap_trans = umap.UMAP(
        densmap=True,
        output_dens=True,
        dens_lambda=dens_lambda,
        n_neighbors=30,
        min_dist=0,
        n_components=2,
        metric="precomputed",
    ).fit(dm_wasserstein)
# random_state=42, apparently random_state makes it slower (non-threaded?)

# extract embedding and radii
umap_emb, r_orig, r_emb = attrgetter("embedding_", "rad_orig_", "rad_emb_")(umap_trans)

# fit for visualization
std_trans = umap.UMAP(
    densmap=True, output_dens=True, dens_lambda=dens_lambda, metric="precomputed"
).fit(dm_wasserstein)

# embedding and radii
std_emb, std_r_orig, std_r_emb = attrgetter("embedding_", "rad_orig_", "rad_emb_")(
    std_trans
)

# %%
# clustering
clusterer = hdbscan.HDBSCAN(
    min_samples=1, cluster_selection_epsilon=0.63, min_cluster_size=50
).fit(umap_emb)
labels, probabilities = attrgetter("labels_", "probabilities_")(clusterer)

# plotting
ax1 = plt.scatter(
    std_emb[:, 0], std_emb[:, 1], c=labels, s=5, cmap=plt.cm.nipy_spectral, label=labels
)
plt.axis("off")
unclass_frac = np.count_nonzero(labels == -1) / len(labels)
plt.legend(["Unclassified: " + "{:.1%}".format(unclass_frac)])
lbl_ints = np.arange(np.amax(labels) + 2)
plt.colorbar(boundaries=lbl_ints - 0.5, label="Cluster ID").set_ticks(lbl_ints)
plt.show()

# cluster count histogram
col_scl = MinMaxScaler()
unique_labels = np.unique(labels)
col_trans = col_scl.fit(unique_labels.reshape(-1, 1))
scl_vals = col_trans.transform(unique_labels.reshape(-1, 1))
color = plt.cm.nipy_spectral(scl_vals)
plt.bar(*np.unique(labels, return_counts=True), color=color)
plt.show()

# target value scatter plot
plt.scatter(std_emb[:, 0], std_emb[:, 1], c=target, s=15, cmap="Spectral")
plt.axis("off")
plt.colorbar(label="Bulk Modulus (GPa)")
plt.show()

# %% multivariate normal probability summation
mn = np.amin(std_emb, axis=0)
mx = np.amax(std_emb, axis=0)
num = 200
x, y = np.mgrid[mn[0] : mx[0] : num * 1j, mn[1] : mx[1] : num * 1j]
pos = np.dstack((x, y))


def my_mvn(mu_x, mu_y, r):
    """Calculate multivariate normal at (mu_x, mu_y) with constant radius, r."""
    return multivariate_normal([mu_x, mu_y], [[r, 0], [0, r]])


mvn_list = list(map(my_mvn, std_emb[:, 0], std_emb[:, 1], np.exp(r_orig)))
pdf_list = [mvn.pdf(pos) for mvn in mvn_list]
pdf_sum = np.sum(pdf_list, axis=0)

dens = np.exp(r_orig)

# density scatter plot
plt.scatter(x, y, c=pdf_sum)
plt.axis("off")
plt.colorbar(label="Density")
plt.show()

# density and target value scatter plot
plt.scatter(x, y, c=pdf_sum)
plt.scatter(
    std_emb[:, 0],
    std_emb[:, 1],
    c=target,
    s=15,
    cmap="Spectral",
    edgecolors="none",
    alpha=0.15,
)
plt.axis("off")
plt.show()


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


# # Fairly fast for many datapoints, less fast for many costs, somewhat readable
# def is_pareto_efficient_simple(costs):
#     """
#     Find the pareto-efficient points.

#     :param costs: An (n_points, n_costs) array
#     :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
#     """
#     is_efficient = np.ones(costs.shape[0], dtype=bool)
#     for i, c in enumerate(costs):
#         if is_efficient[i]:
#             is_efficient[is_efficient] = np.any(
#                 costs[is_efficient] < c, axis=1
#             )  # Keep any point with a lower cost
#             is_efficient[i] = True  # And keep self
#     return is_efficient


# pareto_ind = np.nonzero(
#     is_pareto_efficient_simple(np.array([1 / neigh_avg_targ, target]).T)
# )
