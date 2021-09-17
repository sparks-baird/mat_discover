"""
Materials discovery using Earth Mover's Distance, densMAP embeddings, and HDBSCAN*.

- create distance matrix
- apply densMAP
- create clusters via HDBSCAN*
- search for interesting materials, for example:
     - high-target/low-density
     - materials with high-target surrounded by materials with low targets

Run using elm2d_ environment.

Created on Mon Sep  6 23:15:27 2021.

@author: sterg
"""
from operator import attrgetter
from utils.Timer import Timer, NoTimer
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn.preprocessing import MinMaxScaler

import umap
import hdbscan

from ElM2D import ElM2D

from utils.nearest_neigh import nearest_neigh_props
from utils.pareto import pareto_plot

plt.rcParams["text.usetex"] = True


def my_mvn(mu_x, mu_y, r):
    """Calculate multivariate normal at (mu_x, mu_y) with constant radius, r."""
    return multivariate_normal([mu_x, mu_y], [[r, 0], [0, r]])


class Discover:
    """Class for ElM2D, dimensionality reduction, clustering, and plotting."""

    def __init__(
        self,
        timed: bool = True,
        dens_lambda: float = 1.0,
        plotting: bool = True,
        pdf: bool = True,
        n_neighbors: int = 10,
    ):
        if timed:
            self.Timer = Timer
        else:
            self.Timer = NoTimer

        self.dens_lambda = dens_lambda
        self.plotting = plotting
        self.pdf = pdf
        self.n_neighbors = n_neighbors

        self.mapper = ElM2D(target="cuda")  # type: ignore

    def fit(self, df):
        # unpack
        self.formulas = df["composition"]
        self.target = df["pred-0"]

        # distance matrix
        with self.Timer("fit-wasserstein"):
            self.mapper.fit(self.formulas)
            X = self.mapper.dm

        # nearest neighbors
        rad_neigh_avg_targ, self.k_neigh_avg_targ = nearest_neigh_props(X, self.target)

        # UMAP
        umap_trans = self.umap_fit_cluster(X)
        std_trans = self.umap_fit_vis(X)

        self.umap_emb = self.extract_emb_rad(umap_trans)[0]
        self.std_emb, self.std_r_orig = self.extract_emb_rad(std_trans)[:2]

        # HDBSCAN
        clusterer = self.cluster(self.umap_emb)
        self.labels = self.extract_labels_probs(clusterer)[0]

        # Probability Density Function Summation
        if self.pdf:
            self.pdf_x, self.pdf_y, self.pdf_sum = self.mvn_prob_sum(
                self.std_emb, self.std_r_orig
            )

        # Plotting
        if self.plotting:
            self.plot()

    def cluster(self, umap_emb):
        with self.Timer("HDBSCAN*"):
            clusterer = hdbscan.HDBSCAN(
                min_samples=1, cluster_selection_epsilon=0.63, min_cluster_size=50
            ).fit(umap_emb)
        return clusterer

    def extract_labels_probs(self, clusterer):
        labels, probabilities = attrgetter("labels_", "probabilities_")(clusterer)
        return labels, probabilities

    def umap_fit_cluster(self, dm):
        with self.Timer("fit-UMAP"):
            umap_trans = umap.UMAP(
                densmap=True,
                output_dens=True,
                dens_lambda=self.dens_lambda,
                n_neighbors=30,
                min_dist=0,
                n_components=2,
                metric="precomputed",
            ).fit(dm)
        return umap_trans

    def umap_fit_vis(self, X):
        with self.Timer("fit-vis-UMAP"):
            std_trans = umap.UMAP(
                densmap=True,
                output_dens=True,
                dens_lambda=self.dens_lambda,
                metric="precomputed",
            ).fit(X)
        return std_trans

    def extract_emb_rad(self, trans):
        """Extract densMAP embedding and radii.

        Parameters
        ----------
        trans : class
            A fitted UMAP class.

        Returns
        -------
        emb
            UMAP embedding
        r_orig
            original radii
        r_emb
            embedded radii
        """
        emb, r_orig, r_emb = attrgetter("embedding_", "rad_orig_", "rad_emb_")(trans)
        return emb, r_orig, r_emb

    def mvn_prob_sum(self, std_emb, r_orig, n=20):
        # multivariate normal probability summation
        mn = np.amin(std_emb, axis=0)
        mx = np.amax(std_emb, axis=0)
        x, y = np.mgrid[mn[0] : mx[0] : n * 1j, mn[1] : mx[1] : n * 1j]  # type: ignore
        pos = np.dstack((x, y))

        with Timer("pdf-summation"):
            mvn_list = list(map(my_mvn, std_emb[:, 0], std_emb[:, 1], np.exp(r_orig)))
            pdf_list = [mvn.pdf(pos) for mvn in mvn_list]
            pdf_sum = np.sum(pdf_list, axis=0)
        return x, y, pdf_sum

    def plot(self):
        # peak pareto plot setup
        x = str(self.n_neighbors) + "_neigh_avg_targ (GPa)"
        y = "target (GPa)"
        peak_df = pd.DataFrame(
            {
                x: self.k_neigh_avg_targ,
                y: self.target,
                "formulas": self.formulas,
                "Peak height (GPa)": self.target - self.k_neigh_avg_targ,
            }
        )

        # peak pareto plot
        fig, pareto_ind = pareto_plot(
            peak_df,
            x=x,
            y=y,
            fpath="pf-peak-proxy",
            pareto_front=True,
        )

        # Scatter plot colored by clusters
        self.umap_cluster_scatter(self.std_emb, self.labels)

        # Histogram of cluster counts
        self.cluster_count_hist(self.labels)

        # Scatter plot colored by target values
        self.target_scatter(self.std_emb, self.target)

        # PDF evaluated on grid of points
        if self.pdf:
            self.dens_scatter(self.pdf_x, self.pdf_y, self.pdf_sum)

            self.dens_targ_scatter(
                self.std_emb, self.target, self.pdf_x, self.pdf_y, self.pdf_sum
            )

        # dens pareto plot setup
        dens = np.exp(self.std_r_orig)
        log_dens = np.log(dens)
        x = "log-density"
        y = "target (GPa)"
        dens_df = pd.DataFrame(
            {
                x: log_dens,
                y: self.target,
                "formulas": self.formulas,
            }
        )

        # dens pareto plot
        fig, pareto_ind = pareto_plot(
            dens_df,
            x=x,
            y=y,
            fpath="pf-dens-proxy",
            parity_type="none",
            color=None,
            pareto_front=True,
        )

    def umap_cluster_scatter(self, std_emb, labels):
        plt.scatter(
            std_emb[:, 0],
            std_emb[:, 1],
            c=labels,
            s=5,
            cmap=plt.cm.nipy_spectral,
            label=labels,
        )
        plt.axis("off")
        unclass_frac = np.count_nonzero(labels == -1) / len(labels)
        plt.legend(["Unclassified: " + "{:.1%}".format(unclass_frac)])
        lbl_ints = np.arange(np.amax(labels) + 2)
        if unclass_frac != 1.0:
            plt.colorbar(boundaries=lbl_ints - 0.5, label="Cluster ID").set_ticks(
                lbl_ints
            )
        plt.show()

    def cluster_count_hist(self, labels):
        col_scl = MinMaxScaler()
        unique_labels = np.unique(labels)
        col_trans = col_scl.fit(unique_labels.reshape(-1, 1))
        scl_vals = col_trans.transform(unique_labels.reshape(-1, 1))
        color = plt.cm.nipy_spectral(scl_vals)

        plt.bar(*np.unique(labels, return_counts=True), color=color)
        plt.show()

    def target_scatter(self, std_emb, target):
        plt.scatter(std_emb[:, 0], std_emb[:, 1], c=target, s=15, cmap="Spectral")
        plt.axis("off")
        plt.colorbar(label="Bulk Modulus (GPa)")
        plt.show()

    def dens_scatter(self, x, y, pdf_sum):
        plt.scatter(x, y, c=pdf_sum)
        plt.axis("off")
        plt.colorbar(label="Density")
        plt.show()

    def dens_targ_scatter(self, std_emb, target, x, y, pdf_sum):
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
# remove dependence on paths
# def load_data(self,
#               folder = "C:/Users/sterg/Documents/GitHub/sparks-baird/ElM2D/CrabNet/model_predictions",
#               name="elasticity_val_output.csv", # example_materials_property_val_output.csv
#               ):
#     fpath = join(folder, name)
#     val_df = pd.read_csv(fpath)

#     self.formulas = val_df["composition"][0:20000]
#     self.target = val_df["pred-0"][0:20000]
