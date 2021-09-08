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
from helper import Timer
from os.path import join
from operator import attrgetter

import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.preprocessing import MinMaxScaler

import umap
import hdbscan

import matplotlib.pyplot as plt

from ElM2D import ElM2D


mapper = ElM2D()

# test data
datapath = join("ael_bulk_modulus_voigt", "train.csv")
valpath = join("ael_bulk_modulus_voigt", "val.csv")
df = pd.read_csv(datapath)
val_df = pd.read_csv(valpath)
tmp_df = pd.concat([df[0:4], val_df[0:3]])
formulas = df["formula"]

# distance matrix
with Timer("fit-wasserstein"):
    mapper.fit(formulas)
    dm_wasserstein = mapper.dm

# %% UMAP
dens_lambda = 2

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
    min_samples=1, cluster_selection_epsilon=0.63, min_cluster_size=30
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
plt.scatter(std_emb[:, 0], std_emb[:, 1], c=df["target"], s=15, cmap="Spectral")
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


mvn_list = list(map(my_mvn, std_emb[:, 0], std_emb[:, 1], 1e9 * np.exp(r_orig)))
pdf_list = [mvn.pdf(pos) for mvn in mvn_list]
pdf_sum = np.sum(pdf_list, axis=0)


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
    c=df["target"],
    s=15,
    cmap="Spectral",
    edgecolors="none",
    alpha=0.15,
)
plt.axis("off")
plt.show()
