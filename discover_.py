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
import os
from warnings import warn
from operator import attrgetter
from ElM2D.utils.Timer import Timer, NoTimer
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, wasserstein_distance
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.manifold import MDS

# from sklearn.decomposition import PCA

import umap
import hdbscan

from ElM2D.ElM2D_ import ElM2D

from discover.utils.nearest_neigh import nearest_neigh_props
from discover.utils.pareto import pareto_plot, get_pareto_ind

import torch

from CrabNet.train_crabnet import main as crabnet_main  # noqa
from CrabNet.train_crabnet import get_model

use_cuda = torch.cuda.is_available()

# plt.rcParams["text.usetex"] = True


def my_mvn(mu_x, mu_y, r):
    """Calculate multivariate normal at (mu_x, mu_y) with constant radius, r."""
    return multivariate_normal([mu_x, mu_y], [[r, 0], [0, r]])


class Discover:
    """Class for ElM2D, dimensionality reduction, clustering, and plotting."""

    def __init__(
        self,
        timed: bool = True,
        dens_lambda: float = 1.0,
        plotting: bool = False,
        pdf: bool = True,
        n_neighbors: int = 10,
        verbose: bool = True,
        mat_prop_name="test-property",
        dummy_run=False,
    ):
        if timed:
            self.Timer = Timer
        else:
            self.Timer = NoTimer

        self.dens_lambda = dens_lambda
        self.plotting = plotting
        self.pdf = pdf
        self.n_neighbors = n_neighbors
        self.verbose = verbose
        self.mat_prop_name = mat_prop_name
        self.dummy_run = dummy_run

        self.mapper = ElM2D(target="cuda")  # type: ignore
        self.dm = None
        # self.formula = None
        # self.target = None
        self.train_formula = None
        self.train_target = None
        self.train_df = None
        self.val_df = None

    def fit(self, train_df):
        # unpack
        self.train_df = train_df
        self.train_formula = train_df["formula"]
        self.train_target = train_df["target"]

        # HACK: remove directory structure dependence
        os.chdir("CrabNet")
        # TODO: remove the "val MAE", which is wrong (should be NaN or just not displayed)
        # turns out this is a bit more difficult because of use of self.data_loader
        # dummy_df = train_df.iloc[:1, :]
        # dummy_df["target"] = np.mean(train_df["target"])
        with Timer("train-CrabNet"):
            self.crabnet_model = get_model(
                mat_prop=self.mat_prop_name,
                train_df=train_df,
                learningcurve=False,
            )
        # crabnet_main(mat_prop="elasticity", train_df=df)
        os.chdir("..")

        # TODO: UMAP on new data (i.e. add metric != "precomputed" functionality)

    def predict(
        self,
        val_df,
        plotting=None,
        umap_random_state=None,
        pred_weight=2,  # pred_weight - how much to weight the prediction values relative to val_ratio
        Scaler=MinMaxScaler,
        dummy_run=None,
    ):
        self.val_df = val_df

        # CrabNet
        # (act, pred, formulae, uncert)
        _, train_pred, _, train_sigma = self.crabnet_model.predict(self.train_df)
        _, val_pred, _, val_sigma = self.crabnet_model.predict(self.val_df)
        pred = np.concatenate((train_pred, val_pred), axis=0)

        train_formula = self.train_df["formula"]
        train_target = self.train_df["target"]
        val_formula = self.val_df["formula"]
        if "target" in self.val_df.columns:
            val_target = self.val_df["target"]
        else:
            val_target = val_pred

        self.all_formula = pd.concat((train_formula, val_formula), axis=0)
        self.all_target = pd.concat((train_target, val_target), axis=0)

        ntrain, nval = len(train_formula), len(val_formula)
        ntot = ntrain + nval
        train_ids, val_ids = np.arange(ntrain), np.arange(ntrain, ntot)

        # distance matrix
        with self.Timer("fit-wasserstein"):
            self.mapper.fit(self.all_formula)
            self.dm = self.mapper.dm

        # UMAP (clustering and visualization)
        # TODO: swap with PCA for quick test
        if (dummy_run is None and self.dummy_run) or dummy_run:
            umap_trans = MDS(n_components=2, dissimilarity="precomputed").fit(self.dm)
            std_trans = umap_trans
            self.umap_emb = umap_trans.embedding_
            self.std_emb = umap_trans.embedding_
            self.umap_r_orig = np.random.rand(self.dm.shape[0])
            self.std_r_orig = np.random.rand(self.dm.shape[0])
        else:
            umap_trans = self.umap_fit_cluster(self.dm, random_state=umap_random_state)
            std_trans = self.umap_fit_vis(self.dm, random_state=umap_random_state)

            # TODO: swap with random numbers for quick test
            self.umap_emb, self.umap_r_orig = self.extract_emb_rad(umap_trans)[:2]
            self.std_emb, self.std_r_orig = self.extract_emb_rad(std_trans)[:2]

        # HDBSCAN*
        if (dummy_run is None and self.dummy_run) or dummy_run:
            min_cluster_size = 5
        else:
            min_cluster_size = 50
        clusterer = self.cluster(self.umap_emb, min_cluster_size=min_cluster_size)
        self.labels = self.extract_labels_probs(clusterer)[0]
        self.unique_labels = np.unique(self.labels)

        # Probability Density Function Summation
        if self.pdf:
            self.pdf_x, self.pdf_y, self.pdf_sum = self.mvn_prob_sum(
                self.std_emb, self.std_r_orig
            )

        # TODO: implement "score" method: validation density contributed by training densities
        # train_emb = self.umap_emb[:ntrain]
        # val_emb = self.umap_emb[ntrain:]

        # cluster-wise predicted average
        cluster_pred = np.array(
            [pred[self.labels == lbl] for lbl in self.unique_labels], dtype=object
        )
        self.cluster_avg = np.vectorize(np.mean)(cluster_pred).reshape(-1, 1)

        # cluster-wise validation fraction
        train_ct, val_ct = np.array(
            [
                [
                    np.count_nonzero(self.labels[ids] == lbl)
                    for lbl in self.unique_labels
                ]
                for ids in [train_ids, val_ids]
            ]
        )
        self.val_frac = (val_ct / (train_ct + val_ct)).reshape(-1, 1)

        # Scale and weight the data
        # REVIEW: Which Scaler to use? RobustScaler means no set limits on "score"
        pred_scaler = Scaler().fit(self.cluster_avg)
        # frac_scaler = Scaler().fit(self.val_frac)

        pred_scaled = pred_weight * pred_scaler.transform(self.cluster_avg)
        # frac_scaled = frac_scaler.transform(self.val_frac)
        frac_scaled = self.val_frac  # already between 0 and 1

        # combined data
        comb_data = pred_scaled + frac_scaled
        comb_scaler = Scaler().fit(comb_data)

        # scores range between 0 and 1
        self.score = comb_scaler.transform(comb_data)

        # REVIEW: val_frac OK or use density of validation data relative to training data?

        # TODO: Nearest Neighbor properties (for plotting only), incorporate as separate "score" type
        rad_neigh_avg_targ, self.k_neigh_avg_targ = nearest_neigh_props(self.dm, pred)

        # Plotting
        if (self.plotting and plotting is None) or plotting:
            self.plot()

        return self.score

    def group_cross_val(self, df, umap_random_state=None, dummy_run=None):
        # TODO: remind people to use a separate Discover() instance if they wish to access fit and gcv attributes
        self.all_formula = df["formula"]
        self.all_target = df["target"]

        # distance matrix
        with self.Timer("fit-wasserstein"):
            self.mapper.fit(self.all_formula)
            self.dm = self.mapper.dm

        # UMAP (clustering and visualization)
        if (dummy_run is None and self.dummy_run) or dummy_run:
            umap_trans = MDS(n_components=2, dissimilarity="precomputed").fit(self.dm)
            self.umap_emb = umap_trans.embedding_
            min_cluster_size = 5
        else:
            umap_trans = self.umap_fit_cluster(self.dm, random_state=umap_random_state)
            self.umap_emb = self.extract_emb_rad(umap_trans)[0]
            min_cluster_size = 50

        # HDBSCAN*
        clusterer = self.cluster(self.umap_emb, min_cluster_size=min_cluster_size)
        self.labels = self.extract_labels_probs(clusterer)[0]

        # Group Cross Validation Setup
        logo = LeaveOneGroupOut()

        n_clusters = np.max(self.labels)
        if n_clusters == -1:
            raise ValueError("no clusters")
        n_test_clusters = np.max([1, int(n_clusters * 0.1)])

        """Difficult to repeatably set aside a test set for a given clustering result;
        by changing clustering parameters, the test clusters might change, so using
        seed=42 for UMAP random_state, then seed=10 for setting aside clusters.
        Note that "unclustered" is never assigned as a test_cluster, and so is always
        included in tv_cluster_ids (tv===train_validation)."""
        np.random.default_rng(seed=10)
        test_cluster_ids = np.random.choice(n_clusters + 1, n_test_clusters)
        np.random.default_rng()

        test_ids = np.isin(self.labels, test_cluster_ids)
        tv_cluster_ids = np.setdiff1d(np.arange(-1, n_clusters + 1), test_cluster_ids)
        tv_ids = np.isin(self.labels, tv_cluster_ids)

        X, y = self.all_formula, self.all_target
        X_test, y_test = X[test_ids], y[test_ids]  # noqa
        X_tv, y_tv, labels_tv = X[tv_ids], y[tv_ids], self.labels[tv_ids]

        # Group Cross Validation
        # HACK: remove directory structure dependence
        os.chdir("CrabNet")
        # for train_index, val_index in logo.split(X_tv, y_tv, self.labels):
        avg_targ = [
            self.single_group_cross_val(X_tv, y_tv, train_index, val_index, i)
            for i, (train_index, val_index) in enumerate(
                logo.split(X_tv, y_tv, labels_tv)
            )
        ]
        out = np.array(avg_targ).T
        self.true_avg_targ, self.pred_avg_targ, self.train_avg_targ = out

        # Sorting "Error" Setup
        u = np.flip(np.cumsum(np.linspace(0, 1, len(self.true_avg_targ))))
        v = u.copy()
        # sorting indices from high to low
        self.sorter = np.flip(self.true_avg_targ.argsort())

        # sort everything by same indices to preserve cluster order
        u_weights = self.true_avg_targ[self.sorter]
        v_weights = self.pred_avg_targ[self.sorter]
        dummy_v_weights = self.train_avg_targ[self.sorter]

        # Weighted distance between CDFs as proxy for "how well sorted" (i.e. sorting metric)
        self.error = wasserstein_distance(
            u, v, u_weights=u_weights, v_weights=v_weights
        )
        if self.verbose:
            print("Weighted group cross validation error =", self.error)

        # Dummy Error (i.e. if you "guess" the mean of training targets)
        self.dummy_error = wasserstein_distance(
            u,
            v,
            u_weights=u_weights,
            v_weights=dummy_v_weights,
        )
        # HACK: remove directory structure dependence
        os.chdir("..")

        # Similar to MatBench "scaled_error"
        self.scaled_error = self.error / self.dummy_error
        if self.verbose:
            print("Weighted group cross validation scaled error =", self.scaled_error)
        return self.scaled_error

    def single_group_cross_val(self, X, y, train_index, val_index, iter):
        """
        TODO: add proper docstring.

        how do I capture a "distinct material class of high targets" in a single number?

        Average of the cluster targets vs. average of the cluster densities (or neigh_avg_targ)?

        Except that cluster densities are constant. It seems I really might need to get
        the forward transform via UMAP.
        Or I could sum the densities from all the other clusters (i.e. training data)?
        Or take the neigh_avg_targ from only the training data?


        Two questions:
        How well are the targets predicted for the val cluster?
        How well did UMAP push the val cluster away from other clusters?

        Then compare the trade-off between these two.


        Or just see if the targets are predicted well based on how certain clusters were removed.

        Then maybe just return the true mean of the targets and the predicted mean of the targets?

        This will tell me which clusters are more interesting. Then I can start looking within
        clusters at the trade-off between high-target and low-proxy to prioritize which materials
        to look at first.
        """
        Xn, yn = X.to_numpy(), y.to_numpy()
        X_train, X_val = Xn[train_index], Xn[val_index]
        y_train, y_val = yn[train_index], yn[val_index]

        train_df = pd.DataFrame({"formula": X_train, "target": y_train})
        val_df = pd.DataFrame({"formula": X_val, "target": y_val})
        # REVIEW: comment when ready for publication
        test_df = None

        # REVIEW: uncomment when ready for publication
        # test_df = pd.DataFrame({"formula": X_test, "property": y_test})

        # train CrabNet
        # train_pred_df, val_pred_df, test_pred_df = crabnet_main(
        #     mat_prop="elasticity",
        #     train_df=train_df,
        #     val_df=val_df,
        #     test_df=test_df,
        #     learningcurve=False,
        #     verbose=False,
        # )

        self.crabnet_model = get_model(
            mat_prop=self.mat_prop_name,
            train_df=train_df,
            val_df=val_df,
            learningcurve=False,
            verbose=False,
        )

        # (act, pred, formulae, uncert)
        train_true, train_pred, _, train_sigma = self.crabnet_model.predict(train_df)
        val_true, val_pred, _, val_sigma = self.crabnet_model.predict(val_df)
        if test_df is not None:
            _, test_pred, _, test_sigma = self.crabnet_model.predict(test_df)
        # true_targ = np.concatenate((train_true, val_true), axis=0)
        # pred_targ = np.concatenate((train_pred, val_pred), axis=0)

        true_avg_targ = np.mean(val_true)
        pred_avg_targ = np.mean(val_pred)

        train_avg_targ = np.mean(train_true)

        # nearest neighbors
        # rad_neigh_avg_targ, k_neigh_avg_targ = nearest_neigh_props(
        #     self.dm, self.all_target
        # )

        # pareto_ind = get_pareto_ind(k_neigh_avg_targ, target)

        # # TODO: determine pareto locations of validation cluster
        # non_pareto_ind = np.setdiff1d(range(len(target)), pareto_ind)
        # val_pareto_ind = np.intersect1d(pareto_ind, val_index)
        # if val_pareto_ind.size == 0:
        #     warn("No validation points on pareto front.")

        # val_neigh_avg_targ = k_neigh_avg_targ[val_index]

        # k_neigh_avg_targ = nearest_neigh_props(
        #     self.dm[:, val_index][val_index, :], pred_targ
        # )[1]

        # better_ind = pred_targ[
        #     pred_targ > 0.5 * np.max(pred_targ)
        #     and val_avg_targ < 0.5 * np.max(val_neigh_avg_targ)
        # ]

        # # log densMAP densities
        # log_dens = self.log_density()
        # val_dens = log_dens[val_index]

        # pkval_pareto_ind = get_pareto_ind(val_avg_targ, val_targ)
        # densval_pareto_ind = get_pareto_ind(val_dens, val_targ)

        # TODO: return (or save) the average of the training targets (for a more accurate dummy_error to be pedantically correct)
        return true_avg_targ, pred_avg_targ, train_avg_targ

    def cluster(self, umap_emb, min_cluster_size=50):
        with self.Timer("HDBSCAN*"):
            clusterer = hdbscan.HDBSCAN(
                min_samples=1,
                cluster_selection_epsilon=0.63,
                min_cluster_size=min_cluster_size,
                # allow_single_cluster=True,
            ).fit(umap_emb)
        return clusterer

    def extract_labels_probs(self, clusterer):
        labels, probabilities = attrgetter("labels_", "probabilities_")(clusterer)
        return labels, probabilities

    def umap_fit_cluster(self, dm, random_state=None):
        with self.Timer("fit-UMAP"):
            umap_trans = umap.UMAP(
                densmap=True,
                output_dens=True,
                dens_lambda=self.dens_lambda,
                n_neighbors=30,
                min_dist=0,
                n_components=2,
                metric="precomputed",
                random_state=random_state,
            ).fit(dm)
        return umap_trans

    def umap_fit_vis(self, X, random_state=None):
        with self.Timer("fit-vis-UMAP"):
            std_trans = umap.UMAP(
                densmap=True,
                output_dens=True,
                dens_lambda=self.dens_lambda,
                metric="precomputed",
                random_state=random_state,
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

    def mvn_prob_sum(self, std_emb, r_orig, n=100):
        # multivariate normal probability summation
        mn = np.amin(std_emb, axis=0)
        mx = np.amax(std_emb, axis=0)
        x, y = np.mgrid[mn[0] : mx[0] : n * 1j, mn[1] : mx[1] : n * 1j]  # type: ignore
        pos = np.dstack((x, y))

        with self.Timer("pdf-summation"):
            mvn_list = list(map(my_mvn, std_emb[:, 0], std_emb[:, 1], np.exp(r_orig)))
            pdf_list = [mvn.pdf(pos) for mvn in mvn_list]
            pdf_sum = np.sum(pdf_list, axis=0)
        return x, y, pdf_sum

    def log_density(self, std_r_orig=None):
        if std_r_orig is None:
            std_r_orig = self.std_r_orig
        dens = np.exp(std_r_orig)
        self.log_dens = np.log(dens)
        return self.log_dens

    def plot(self, return_pareto_ind=False):
        # peak pareto plot setup
        x = str(self.n_neighbors) + "_neigh_avg_targ (GPa)"
        y = "target (GPa)"
        # TODO: plot for val data only (?)
        peak_df = pd.DataFrame(
            {
                x: self.k_neigh_avg_targ,
                y: self.all_target,
                "formula": self.all_formula,
                "Peak height (GPa)": self.all_target - self.k_neigh_avg_targ,
                "cluster ID": self.labels,
            }
        )
        # peak pareto plot
        # TODO: double check if cluster labels are correct between the two pareto plots
        fig, pk_pareto_ind = pareto_plot(
            peak_df,
            x=x,
            y=y,
            color="cluster ID",
            fpath="pf-peak-proxy",
            pareto_front=True,
        )

        x = "cluster-wise validation fraction"
        y = "cluster-wise average target (GPa)"
        # cluster-wise average vs. cluster-wise validation fraction
        frac_df = pd.DataFrame(
            {
                x: self.val_frac.ravel(),
                y: self.cluster_avg.ravel(),
                "cluster ID": self.unique_labels,
            }
        )
        fig, frac_pareto_ind = pareto_plot(
            frac_df,
            x=x,
            y=y,
            hover_data=None,
            color="cluster ID",
            fpath="pf-frac-proxy",
            pareto_front=True,
            reverse_x=False,
            parity_type=None,
        )

        # Scatter plot colored by clusters
        self.umap_cluster_scatter(self.std_emb, self.labels)

        # Histogram of cluster counts
        self.cluster_count_hist(self.labels)

        # Scatter plot colored by target values
        self.target_scatter(self.std_emb, self.all_target)

        # PDF evaluated on grid of points
        if self.pdf:
            self.dens_scatter(self.pdf_x, self.pdf_y, self.pdf_sum)

            self.dens_targ_scatter(
                self.std_emb, self.all_target, self.pdf_x, self.pdf_y, self.pdf_sum
            )

        # dens pareto plot setup
        log_dens = self.log_density()
        x = "log-density"
        y = "target (GPa)"
        dens_df = pd.DataFrame(
            {
                x: log_dens,
                y: self.all_target,
                "formula": self.all_formula,
                "cluster ID": self.labels + 1,
            }
        )
        # dens pareto plot
        fig, dens_pareto_ind = pareto_plot(
            dens_df,
            x=x,
            y=y,
            fpath="pf-dens-proxy",
            parity_type=None,
            color="cluster ID",
            pareto_front=True,
        )

        if return_pareto_ind:
            return pk_pareto_ind, dens_pareto_ind

    def umap_cluster_scatter(self, std_emb, labels):
        # TODO: update plotting commands to have optional arguments (e.g. std_emb and labels)
        # plt.scatter(
        #     std_emb[:, 0],
        #     std_emb[:, 1],
        #     c=labels,
        #     s=5,
        #     cmap=plt.cm.nipy_spectral,
        #     label=labels,
        # )
        class_ids = labels != -1
        ax = plt.scatter(
            std_emb[:, 0],
            std_emb[:, 1],
            c=labels,
            s=5,
            cmap=plt.cm.nipy_spectral,
            label=labels,
        )
        unclass_ids = np.invert(class_ids)
        unclass_frac = np.sum(unclass_ids) / len(labels)
        plt.axis("off")

        if unclass_frac != 0.0:
            ax2 = plt.scatter(
                std_emb[unclass_ids, 0],
                std_emb[unclass_ids, 1],
                c=labels[unclass_ids],
                s=5,
                cmap=plt.cm.nipy_spectral,
                label=labels[unclass_ids],
            )
            plt.legend([ax2], ["Unclassified: " + "{:.1%}".format(unclass_frac)])

        lbl_ints = np.arange(np.amax(labels) + 1)
        if unclass_frac != 1.0:
            plt.colorbar(ax, boundaries=lbl_ints - 0.5, label="Cluster ID").set_ticks(
                lbl_ints
            )
        plt.show()

    def cluster_count_hist(self, labels):
        col_scl = MinMaxScaler()
        self.unique_labels = np.unique(labels)
        col_trans = col_scl.fit(self.unique_labels.reshape(-1, 1))
        scl_vals = col_trans.transform(self.unique_labels.reshape(-1, 1))
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

    # TODO: write function to visualize Wasserstein metric (barchart with height = color)


# %% CODE GRAVEYARD
# remove dependence on paths
# def load_data(self,
#               folder = "C:/Users/sterg/Documents/GitHub/sparks-baird/ElM2D/CrabNet/model_predictions",
#               name="elasticity_val_output.csv", # example_materials_property_val_output.csv
#               ):
#     fpath = join(folder, name)
#     val_df = pd.read_csv(fpath)

#     self.formula = val_df["composition"][0:20000]
#     self.target = val_df["pred-0"][0:20000]

# X_tv, X_test, y_tv, y_test = train_test_split(
#     X, y, test_size=n_test_clusters, random_state=42
# )

# from CrabNet.crabnet.model import Model
# from CrabNet.crabnet.kingcrab import CrabNet

# model = Model(
#     CrabNet(compute_device=compute_device).to(compute_device),
#     verbose=False,
# )

# self.scores = [
#     wasserstein_distance(u, v, u_weights=true_avg_targ, v_weights=pred_avg_targ)
#     for true_avg_targ, pred_avg_targ in zip(true_avg_targs, pred_avg_targs)
# ]

# self.score = np.mean(self.scores)

# if val_df is not None:
#     cat_df = pd.concat((df, val_df), axis=0)
# else:
#     cat_df = df

# REVIEW: whether to return ratio or fraction (maybe doesn't matter)
# self.val_ratio = val_ct / train_ct

# # unpack
# if df is None and self.df is not None:
#     df = self.df
# elif df is None and self.df is None:
#     raise SyntaxError(
#         "df must be assigned via fit() or supplied to group_cross_val directly"
#     )

# self.val_score = self.score[val_ids]

# val_target = np.array([np.nan] * len(val_formula))


# # targets, nearest neighbors, and pareto front indices
# true_targ = val_df["target"]
# pred_targ = val_pred_df["pred-0"]
# target_tmp = (train_df["target"], val_pred_df["pred-0"])
# target = pd.concat(target_tmp, axis=0)

# dummy_df = train_df.iloc[:1, :]
# dummy_df["target"] = np.nan
