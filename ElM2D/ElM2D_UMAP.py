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
# %% Imports
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

# from ElM2D.helper import Timer
from ElM2D import ElM2D
from helper import Timer

plt.rcParams["text.usetex"] = True

mapper = ElM2D(target="cuda")  # type: ignore


# %% import validation data
fpath = "C:/Users/sterg/Documents/GitHub/sparks-baird/ElM2D/CrabNet/model_predictions/elasticity_val_output.csv"
# fpath = "C:/Users/sterg/Documents/GitHub/sparks-baird/ElM2D/CrabNet/model_predictions/example_materials_property_val_output.csv"
val_df = pd.read_csv(fpath)

formulas = val_df["composition"] #[0:1000]
target = val_df["pred-0"] #[0:1000]

# %% Distance Matrix
with Timer("fit-wasserstein"):
    mapper.fit(formulas)
    dm_wasserstein = mapper.dm

# %% Nearest Neighbors within Radius
mean_wasserstein, std_wasserstein = (np.mean(dm_wasserstein), np.std(dm_wasserstein))
radius = mean_wasserstein - 1.5 * std_wasserstein
NN = NearestNeighbors(radius=radius, metric="precomputed")
NN.fit(dm_wasserstein)
neigh_dist, neigh_ind = NN.radius_neighbors()

neigh_target = np.array([target[ind] for ind in neigh_ind], dtype="object")
neigh_avg_targ = np.array([np.mean(t) if len(t) > 0 else float(0) for t in neigh_target])
num_neigh = np.array([len(ind) for ind in neigh_ind])

# fig = plt.figure()

# %% Pareto-esque Front

# Fairly fast for many datapoints, less fast for many costs, somewhat readable
def is_pareto_efficient_simple(costs):
    """
    Find the pareto-efficient points.

    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    mx = np.max(costs)
    costs = np.nan_to_num(costs, nan=mx)
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(
                costs[is_efficient] < c, axis=1
            )  # Keep any point with a lower cost
            is_efficient[i] = True  # And keep self
    return is_efficient

def pareto_plot(
    df,
    x="proxy",
    y="target",
    color="Peak height (GPa)",
    hover_data=["formulas"],
    fpath="pareto-front",
    reverse_x=True,
    parity_type="max-of-both",
    pareto_front=True,
):
    """Generate and save pareto plot for two variables.

    Parameters
    ----------
    df : DataFrame
        Contains relevant variables for pareto plot.
    x : str, optional
        Name of df column to use for x-axis, by default "proxy"
    y : str, optional
        Name of df column to use for y-axis, by default "target"
    color : str, optional
        Name of df column to use for colors, by default "Peak height (GPa)"
    hover_data : list of str, optional
        Name(s) of df columns to display on hover, by default ["formulas"]
    fpath : str, optional
        Filepath to which to save HTML and PNG. Specify as None if no saving
        is desired, by default "pareto-plot"
    reverse_x : bool, optional
        Whether to reverse the x-axis (i.e. for maximize y and minimize x front)
    parity_type : str, optional
        What kind of parity line to plot: "max-of-both", "max-of-each", or "none"
    """
    fig = px.scatter(
        df,
        x=x,
        y=y,
        color=color,
        color_continuous_scale=px.colors.sequential.Blackbody_r,
        hover_data=hover_data,
    )

    # unpack
    proxy = df[x]
    target = df[y]

    if pareto_front:
        if reverse_x:
            inpt = [proxy, -target]
        else:
            inpt = [-proxy, -target]
        pareto_ind = np.nonzero(
                is_pareto_efficient_simple(np.array(inpt).T)
            )
        # pf_hover_data = df.loc[:, hover_data].iloc[pareto_ind]
        # fig.add_scatter(x=proxy[pareto_ind], y=target[pareto_ind])
        # Add scatter trace with medium sized markers
        fig.add_scatter(
                        mode="markers",
                        x=proxy.iloc[pareto_ind],
                        y=target.iloc[pareto_ind],
                        marker_symbol="circle-open",
                        marker_size=10,
                        hoverinfo='skip',
                        name="Pareto Front",
                        )

    # parity line
    if parity_type == "max-of-both":
        mx = np.nanmax([proxy, target])
        mx2 = mx
    elif parity_type == "max-of-each":
        mx, mx2 = np.nanmax(proxy), np.nanmax(target)

    if parity_type != "none":
        fig.add_trace(go.Line(x=[0, mx], y=[0, mx2], name="parity"))

    # legend and reversal
    fig.update_layout(legend_orientation="h", legend_y=1.1)
    if reverse_x:
        fig.update_layout(xaxis=dict(autorange="reversed"))
    fig.show()

    # saving
    if fpath is not None:
        fig.write_image(fpath + ".png")
        fig.write_html(fpath + ".html")
    
    return fig, pareto_ind

peak_df = pd.DataFrame(
    {
        "neigh_avg_targ (GPa)": neigh_avg_targ,
        "target (GPa)": target,
        "formulas": formulas,
        "Peak height (GPa)": target - neigh_avg_targ,
    }
)

fig, pareto_ind = pareto_plot(peak_df, x="neigh_avg_targ (GPa)", y="target (GPa)", fpath="pf-peak-proxy", pareto_front=True)

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
with Timer("fit-vis-UMAP"):
    std_trans = umap.UMAP(
        densmap=True, output_dens=True, dens_lambda=dens_lambda, metric="precomputed"
    ).fit(dm_wasserstein)

# embedding and radii
std_emb, std_r_orig, std_r_emb = attrgetter("embedding_", "rad_orig_", "rad_emb_")(
    std_trans
)

# %%
# clustering
with Timer("HDBSCAN*"):
    clusterer = hdbscan.HDBSCAN(
        min_samples=1, cluster_selection_epsilon=0.63, min_cluster_size=50
    ).fit(umap_emb)
# extract
labels, probabilities = attrgetter("labels_", "probabilities_")(clusterer)

# plotting
# fig = plt.Figure()
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

# fig = plt.Figure()
plt.bar(*np.unique(labels, return_counts=True), color=color)
plt.show()

# target value scatter plot
# fig = plt.Figure()
plt.scatter(std_emb[:, 0], std_emb[:, 1], c=target, s=15, cmap="Spectral")
plt.axis("off")
plt.colorbar(label="Bulk Modulus (GPa)")
plt.show()

# %% multivariate normal probability summation
# mn = np.amin(std_emb, axis=0)
# mx = np.amax(std_emb, axis=0)
# num = 20
# x, y = np.mgrid[mn[0] : mx[0] : num * 1j, mn[1] : mx[1] : num * 1j]  # type: ignore
# pos = np.dstack((x, y))


def my_mvn(mu_x, mu_y, r):
    """Calculate multivariate normal at (mu_x, mu_y) with constant radius, r."""
    return multivariate_normal([mu_x, mu_y], [[r, 0], [0, r]])

# with Timer("pdf-summation"):
#     mvn_list = list(map(my_mvn, std_emb[:, 0], std_emb[:, 1], np.exp(r_orig)))
#     pdf_list = [mvn.pdf(pos) for mvn in mvn_list]
#     pdf_sum = np.sum(pdf_list, axis=0)

# # density scatter plot
# plt.scatter(x, y, c=pdf_sum)
# plt.axis("off")
# plt.colorbar(label="Density")
# plt.show()

# # density and target value scatter plot
# plt.scatter(x, y, c=pdf_sum)
# plt.scatter(
#     std_emb[:, 0],
#     std_emb[:, 1],
#     c=target,
#     s=15,
#     cmap="Spectral",
#     edgecolors="none",
#     alpha=0.15,
# )
# plt.axis("off")
# plt.show()

# %% density pareto plot
# density estimation for each of the input points
dens = np.exp(r_orig)
log_dens = np.log(dens)

dens_df = pd.DataFrame(
    {
        "log-density": log_dens,
        "target (GPa)": target,
        "formulas": formulas,
    }
)

fig, pareto_ind = pareto_plot(dens_df, x="log-density", y="target (GPa)", fpath="pf-dens-proxy", parity_type="none", color=None, pareto_front=True)

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