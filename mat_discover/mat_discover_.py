"""Materials discovery using Earth Mover's Distance, DensMAP embeddings, and HDBSCAN*.

Create distance matrix, apply densMAP, and create clusters via HDBSCAN* to search for
interesting materials. For example, materials with high-target/low-density (density
proxy) or high-target surrounded by materials with low targets (peak proxy).
"""
import gc

# saving class objects: https://stackoverflow.com/a/37076668/13697228
from copy import copy
from operator import attrgetter
from os.path import join
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, MutableMapping, Optional, Union, List, Mapping
from warnings import warn

import dill as pickle
import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from chem_wasserstein.ElM2D_ import ElM2D
from chem_wasserstein.utils.Timer import NoTimer, Timer

# from crabnet.utils.composition import generate_features
from composition_based_feature_vector.composition import generate_features
from crabnet.crabnet_ import CrabNet
from ElMD import ElMD
from scipy.stats import multivariate_normal, wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from torch.cuda import empty_cache

from mat_discover.utils.data import data
from mat_discover.utils.nearest_neigh import nearest_neigh_props
from mat_discover.utils.pareto import pareto_plot  # , get_pareto_ind
from mat_discover.utils.plotting import (
    cluster_count_hist,
    dens_scatter,
    dens_targ_scatter,
    target_scatter,
    umap_cluster_scatter,
)

# from ml_matrics import density_scatter


# from sklearn.decomposition import PCA


plt.rcParams.update(
    {
        "figure.autolayout": True,
        "figure.figsize": [3.5, 3.5],
        "figure.dpi": 300,
        "xtick.direction": "in",
    }
)


def my_mvn(mu_x, mu_y, r):
    """Calculate multivariate normal at (mu_x, mu_y) with constant radius, r."""
    return multivariate_normal([mu_x, mu_y], [[r, 0], [0, r]])


def cdf_sorting_error(y_true, y_pred, y_dummy=None):
    """Cumulative distribution function sorting error via Wasserstein distance.

    Parameters
    ----------
    y_true, y_pred : list of (float or int or str)
        True and predicted values to use for sorting error, respectively.
    y_dummy : list of (float or int or str), optional
        Dummy values to use to generate a scaled error, by default None

    Returns
    -------
    error, dummy_error, scaled_error : float
        The unscaled, dummy, and scaled errors that describes the mismatch in sorting
        between the CDFs of two lists. The scaled error represents the improvement
        relative to the dummy error, such that `scaled_error = error / dummy_error`. If
        `scaled_error > 1`, the sorting error is worse than if you took the average of
        the `y_true` values as the `y_pred` values. If `scaled_error < 1`, it is better
        than this "dummy" regressor. Scaled errors closer to 0 are better.
    """
    # Sorting "Error" Setup
    n = len(y_true)
    u = np.flip(np.cumsum(np.linspace(0, 1, n)))
    v = u.copy()

    if type(y_true[0]) is str:
        lookup = {label: i for (i, label) in enumerate(y_true)}
        df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
        df.y_true = df.y_true.map(lookup)
        df.y_pred = df.y_pred.map(lookup)
        y_true, y_pred = df.data

    # sorting indices from high to low
    sorter = np.flip(y_true.argsort())

    # sort everything by same indices to preserve cluster order
    u_weights = y_true[sorter]
    v_weights = y_pred[sorter]

    if y_dummy is None:
        y_dummy = [np.mean(y_true)] * n
    dummy_v_weights = y_dummy[sorter]

    # Weighted distance between CDFs as proxy for "how well sorted" (i.e. sorting metric)
    error = wasserstein_distance(u, v, u_weights=u_weights, v_weights=v_weights)

    # Dummy Error (i.e. if you "guess" the mean of training targets)
    dummy_error = wasserstein_distance(
        u,
        v,
        u_weights=u_weights,
        v_weights=dummy_v_weights,
    )

    # Similar to Matbench "scaled_error"
    scaled_error = error / dummy_error

    return error, dummy_error, scaled_error


if TYPE_CHECKING:
    pathtype = PathLike[str]
else:
    pathtype = PathLike


class Discover:
    """
    A Materials Discovery class.

    Uses chemical-based distances, dimensionality reduction, clustering,
    and plotting to search for high performing, chemically unique compounds
    relative to training data.
    """

    def __init__(
        self,
        timed: bool = True,
        dens_lambda: float = 1.0,
        plotting: bool = False,
        pdf: bool = True,
        n_peak_neighbors: int = 10,
        verbose: bool = True,
        mat_prop_name: str = "test-property",
        dummy_run: bool = False,
        Scaler=RobustScaler,
        figure_dir: Union[str, pathtype] = "figures",
        table_dir: Union[str, pathtype] = "tables",
        target_unit: Optional[str] = None,
        use_plotly_offline: bool = True,
        novelty_learner: str = "discover",
        novelty_prop: str = "mod_petti",
        # groupby_filter="max",
        pred_weight: float = 1.0,
        proxy_weight: float = 1.0,
        device: str = "cuda",
        dist_device: Optional[str] = None,
        nscores: int = 100,
        crabnet_kwargs: Optional[MutableMapping] = None,
        umap_cluster_kwargs: Optional[MutableMapping] = None,
        umap_vis_kwargs: Optional[MutableMapping] = None,
        hdbscan_kwargs: Optional[MutableMapping] = None,
    ):
        """Initialize a Discover() class.

        Parameters
        ----------
        timed : bool, optional
            Whether or not timing is reported, by default True

        dens_lambda : float, optional
            "Controls the regularization weight of the density correlation term in
            densMAP. Higher values prioritize density preservation over the UMAP
            objective, and vice versa for values closer to zero. Setting this parameter
            to zero is equivalent to running the original UMAP algorithm." Source:
            https://umap-learn.readthedocs.io/en/latest/api.html, by default 1.0

        plotting : bool, optional
            Whether to create and save various compound-wise and cluster-wise figures,
            by default False

        pdf : bool, optional
            Whether or not probability density function values are computed, by default
            True

        n_peak_neighbors : int, optional
            Number of neighbors to consider when computing k_neigh_avg (i.e. peak
            proxy), by default 10

        verbose : bool, optional
            Whether to print verbose information, by default True

        mat_prop_name : str, optional
            A name that helps identify the training target property, by default
            "test-property"

        dummy_run : bool, optional
            Whether to use MDS instead of UMAP to run quickly for small datasets. Note
            that MDS takes longer for UMAP for large datasets, by default False

        Scaler : str or class, optional
            Scaler to use for weighted_score (i.e. weighted score of target and proxy
            values) Target and proxy are separately scaled using Scaler before taking
            the weighted sum. Possible values are "MinMaxScaler", "StandardScaler",
            "RobustScaler", or an sklearn.preprocessing scaler class, by default RobustScaler.

        figure_dir, table_dir : str, optional
            Relative or absolute path to directory at which to save figures or tables,
            by default "figures" and "tables", respectively. The directory will be
            created if it does not exist already. `if dummy_run` then append "dummy" to
            the folder via `os.path.join`.

        target_unit : Optional[str]
            Unit of target to use in various x, y, and color axes labels. If None, don't
            add a unit to the labels. By default None.

        use_plotly_offline: bool
            Whether to use `offline.plot(fig)` instead of `fig.show()`. Set to False for
            Google Colab. By default, True.

        pred_weight : int, optional
            Weighting applied to the predicted, scaled target values, by default 1 (i.e.
            equal weighting between predictions and proxies). For example, to weight the
            predicted targets at twice that of the proxy values, set to 2 (while keeping
            the default of `proxy_weight = 1`)

        novelty_learner : str or sklearn Regressor, optional
            Whether to use the DiSCoVeR algorithm (`"discover"`) or another learner for
            novelty detection (e.g. `sklearn.neighbors.LocalOutlierFactor`). By default
            "discover".

        novelty_prop : str, optional
            Which featurization scheme to use for determining novelty. `"mod_petti"` is
            is currently the only supported/tested option for the DiSCoVeR
            `novelty_learner` for speed considerations, though the other "linear"
            featurizers should technically be compatible (untested). The "vector"
            featurizers can be implemented, although with some code plumbing needed. See
            ElM2D [1]_ and ElMD supported featurizers [2]_. Possible options for
            `sklearn`-type `novelty_learner`-s are those supported by the CBFV [3]_
            package (and assuming that all elements that appear in train/val datasets
            are supported). By default "mod_petti".

        proxy_weight : int, optional
            Weighting applied to the predicted, scaled proxy values, by default 1 (i.e.
            equal weighting between predictions and proxies when using default
            `pred_weight = 1`). For example, to weight the predicted, scaled targets at
            twice that of the proxy values, set to 2 while retaining `pred_weight = 1`.

        device : str, optional
            Which device to perform the computation on. Possible values are "cpu" and
            "cuda", by default "cuda".

        dist_device : str, optional
            Which device to perform the computation on for the distance computations
            specifically. Possible values are "cpu", "cuda", and None, by default None.
            If None, default to `device`.

        nscores : int, optional
            Number of scores (i.e. compounds) to return in the CSV output files.

        crabnet_kwargs : dict, optional
            `crabnet.crabnet_.CrabNet` kwargs that are passed directly into the CrabNet
            model. By default None. For example, you can specify the number of epochs
            as {"epochs", 40} (may be useful to decrease # epochs for smaller datasets). See `CrabNet() API
            <https://crabnet.readthedocs.io/en/latest/crabnet.html#crabnet.crabnet_.CrabNet>`_.

        umap_cluster_kwargs, umap_vis_kwargs : dict, optional
            `umap.UMAP` kwargs that are passed directly into the UMAP embedder that is
            used for clustering and visualization, respectively. By default None. See
            `basic UMAP parameters
            <https://umap-learn.readthedocs.io/en/latest/parameters.html>`_ and the
            `UMAP API
            <https://umap-learn.readthedocs.io/en/latest/api.html#umap.umap_.UMAP>`_. If
            this contains `dens_lambda` key, the value in the Discover class kwarg will
            take precedence.

        hdbscan_kwargs: dict, optional
            `hdbscan.HDBSCAN` kwargs that are passed directly into the HDBSCAN
            clusterer. By default, None. See `Parameter Selection for HDBSCAN*
            <https://hdbscan.readthedocs.io/en/latest/parameter_selection.html>`_ and
            the `HDBSCAN API
            <https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.hdbscan_.HDBSCAN>`_.
            If ``min_cluster_size`` is not specified, defaults to 50. If
            ``min_samples`` is not specified, defaults to 1. If ``cluster_selection_epsilon``
            is not specified, defaults to 0.63.

        References
        ----------
        .. [1] https://github.com/lrcfmd/ElM2D
        .. [2] https://github.com/lrcfmd/ElMD/tree/v0.4.7#elemental-similarity
        .. [3] https://github.com/kaaiian/CBFV
        """
        if timed:
            self.Timer = Timer
        else:
            self.Timer = NoTimer

        self.dens_lambda = dens_lambda
        self.plotting = plotting
        self.pdf = pdf
        self.n_peak_neighbors = n_peak_neighbors
        self.verbose = verbose
        self.mat_prop_name = mat_prop_name
        self.dummy_run = dummy_run
        if dummy_run:
            figure_dir = join(figure_dir, "dummy")
            table_dir = join(table_dir, "dummy")
            self.epochs: Optional[int] = 2
            epochs_step: Optional[int] = 1
        else:
            self.epochs = None
            epochs_step = None
        self.figure_dir = figure_dir
        self.table_dir = table_dir

        self.target_unit = target_unit
        self.use_plotly_offline = use_plotly_offline

        # create dir https://stackoverflow.com/a/273227/13697228
        Path(self.figure_dir).mkdir(parents=True, exist_ok=True)

        self.novelty_learner = novelty_learner
        self.novelty_prop = novelty_prop
        self.pred_weight = pred_weight
        self.proxy_weight = proxy_weight
        self.device = device
        if dist_device is None:
            self.dist_device = self.device
        else:
            self.dist_device = dist_device

        if type(Scaler) is str:
            scalers = {
                "MinMaxScaler": MinMaxScaler,
                "StandardScaler": StandardScaler,
                "RobustScaler": RobustScaler,
            }
            self.Scaler = scalers[Scaler]
        else:
            self.Scaler = Scaler

        if self.device == "cpu":
            self.force_cpu = True
        else:
            self.force_cpu = False
        self.nscores = nscores

        self.crabnet_kwargs = dict(
            mat_prop=self.mat_prop_name,
            losscurve=False,
            learningcurve=False,
            verbose=self.verbose,
            force_cpu=self.force_cpu,
            epochs=self.epochs,
        )
        if epochs_step is not None:
            self.crabnet_kwargs["epochs_step"] = epochs_step

        # override
        if crabnet_kwargs is not None:
            for key, value in crabnet_kwargs.items():
                self.crabnet_kwargs[key] = value

        if umap_cluster_kwargs is None:
            umap_cluster_kwargs = dict(
                densmap=True,
                output_dens=True,
                dens_lambda=self.dens_lambda,
                n_neighbors=30,
                min_dist=0,
                n_components=2,
                metric="precomputed",
                random_state=None,
                low_memory=False,
            )
        else:
            if "dens_lambda" not in umap_cluster_kwargs:
                umap_cluster_kwargs["dens_lambda"] = self.dens_lambda
            else:
                self.dens_lambda = umap_cluster_kwargs["dens_lambda"]
        self.umap_cluster_kwargs = umap_cluster_kwargs

        if umap_vis_kwargs is None:
            umap_vis_kwargs = dict(
                densmap=True,
                output_dens=True,
                dens_lambda=self.dens_lambda,
                metric="precomputed",
                random_state=None,
                low_memory=False,
            )
        self.umap_vis_kwargs = umap_vis_kwargs

        if hdbscan_kwargs is None:
            if self.dummy_run:
                min_cluster_size = 5
                min_samples = 1
            else:
                min_cluster_size = 50
                min_samples = 1
            hdbscan_kwargs = dict(
                min_samples=min_samples,
                cluster_selection_epsilon=0.63,
                min_cluster_size=min_cluster_size,
            )
        else:
            if "min_cluster_size" not in hdbscan_kwargs:
                hdbscan_kwargs["min_cluster_size"] = 50
            if "min_samples" not in hdbscan_kwargs:
                hdbscan_kwargs["min_samples"] = 1
            if "cluster_selection_epsilon" not in hdbscan_kwargs:
                hdbscan_kwargs["cluster_selection_epsilon"] = 0.63
        self.hdbscan_kwargs = hdbscan_kwargs

        self.mapper = ElM2D(target=self.dist_device)  # type: ignore
        self.dm: Optional[np.ndarray] = None
        self.crabnet_model: Optional[CrabNet] = None
        self.train_formula: Optional[pd.Series] = None
        self.train_target: Optional[pd.Series] = None
        self.train_df: Optional[pd.DataFrame] = None
        self.val_df: Optional[pd.DataFrame] = None
        self.true_avg_targ: Optional[np.ndarray] = None
        self.pred_avg_targ: Optional[np.ndarray] = None
        self.train_avg_targ: Optional[np.ndarray] = None

        # create dir https://stackoverflow.com/a/273227/13697228
        Path(self.figure_dir).mkdir(parents=True, exist_ok=True)
        Path(self.table_dir).mkdir(parents=True, exist_ok=True)

    def fit(self, train_df):
        """Fit CrabNet model to training data.

        Parameters
        ----------
        train_df : DataFrame
            Should contain "formula" and "target" columns.
        """
        # unpack
        self.train_df = train_df
        self.train_formula = train_df["formula"]
        self.train_target = train_df["target"]

        if self.pred_weight != 0:
            with self.Timer("train-CrabNet"):
                if self.crabnet_model is not None:
                    # deallocate CUDA memory https://discuss.pytorch.org/t/how-can-we-release-gpu-memory-cache/14530/28
                    del self.crabnet_model
                    gc.collect()
                    empty_cache()
                self.crabnet_model = CrabNet(**self.crabnet_kwargs)
                self.crabnet_model.fit(train_df)

        # TODO: UMAP on new data (i.e. add metric != "precomputed" functionality)

    def predict(
        self,
        val_df,
        plotting: bool = None,
        umap_random_state=None,
        pred_weight=None,
        proxy_weight=None,
        dummy_run: bool = None,
        count_repeats: bool = False,
        return_peak: bool = False,
    ):
        """Predict target and proxy for validation dataset.

        Parameters
        ----------
        val_df : DataFrame
            Validation dataset containing at minimum "formula" and optionally "target"
            (targets are populated with 0's if not available).
        plotting : bool, optional
            Whether to plot, by default None
        umap_random_state : int or None, optional
            The random seed to use for UMAP, by default None
        pred_weight : int, optional
            The weight to assign to the scaled target predictions (`proxy_weight = 1`
            by default), by default None. If neither `pred_weight` nor
            `self.pred_weight` is specified, it defaults to 1.
        proxy_weight : int, optional
            The weight to assign to the scaled proxy predictions (`pred_weight` is 1 by
            default), by default None. When specified, `proxy_weight` takes precedence
            over `self.proxy_weight`. If neither `proxy_weight` nor `self.proxy_weight` is specified, it defaults to 1.
        dummy_run : bool, optional
            Whether to use MDS in place of the (typically more expensive) DensMAP, by
            default None. If neither dummy_run nor self.dummy_run is specified, it
            defaults to (effectively) being False. When specified, dummy_run takes
            precedence over self.dummy_run.
        count_repeats : bool, optional
            Whether repeat chemical formulae should intensify the local density (i.e.
            decrease the novelty) or not. By default False.
        return_peak : bool, optional
            Whether or not to return the peak scores in addition to the density-based
            scores. By default, False.


        Returns
        -------
        dens_score, peak_score
            Scaled discovery scores for density and peak proxies. Returns only
            `dens_score` if return_peak is False, which is the default.
        """
        if "target" not in val_df.columns:
            val_df["target"] = 0.0

        self.val_df = val_df

        assert (
            self.train_df is not None
        ), "`self.train_df` is None. Did you run `disc.fit(train_df)` already?"

        # CrabNet
        # TODO: parity plot
        crabnet_model = self.crabnet_model
        # CrabNet predict output format: (act, pred, formulae, uncert)
        if self.pred_weight != 0:
            assert (
                crabnet_model is not None
            ), "`crabnet_model is None. Did you run `disc.fit(train_df)` already?"
            train_pred, self.train_sigma, self.train_true = crabnet_model.predict(
                self.train_df, return_uncertainty=True, return_true=True
            )
            self.val_pred, self.val_sigma, self.val_true = crabnet_model.predict(
                self.val_df, return_uncertainty=True, return_true=True
            )
        else:
            self.train_true, train_pred, self.train_sigma = [
                np.zeros(self.train_df.shape[0])
            ] * 3
            self.val_true, self.val_pred, self.val_sigma = [
                np.zeros(self.val_df.shape[0])
            ] * 3
        pred = np.concatenate((train_pred, self.val_pred), axis=0)

        if np.any(np.isnan(self.val_pred)):
            warn("NaN in val_pred. Replacing with DummyRegressor values.")
            val_pred = np.nan_to_num(self.val_pred, nan=np.mean(self.val_true))
        else:
            val_pred = self.val_pred

        self.val_rmse = mean_squared_error(self.val_true, val_pred, squared=False)
        if self.verbose:
            print("val RMSE: ", self.val_rmse)

        train_formula = self.train_df["formula"]
        train_target = self.train_df["target"]
        self.val_formula = self.val_df["formula"]
        if "target" in self.val_df.columns:
            val_target = self.val_df["target"]
        else:
            val_target = self.val_pred

        self.all_formula = pd.concat((train_formula, self.val_formula), axis=0)
        self.all_target = pd.concat((train_target, val_target), axis=0)

        self.ntrain, self.nval = len(train_formula), len(self.val_formula)
        self.ntot = self.ntrain + self.nval
        train_ids, val_ids = np.arange(self.ntrain), np.arange(self.ntrain, self.ntot)

        if self.proxy_weight != 0 and self.novelty_learner == "discover":
            # distance matrix
            with self.Timer("fit-wasserstein"):
                self.mapper.fit(self.all_formula)
                self.dm = self.mapper.dm

            assert self.dm is not None, "`self.dm` is None despite [fit-wasserstein]"
            # TODO: look into UMAP via GPU
            # TODO: create and use fast, built-in Wasserstein UMAP method

            # UMAP (clustering and visualization) (or MDS for a quick run with small dataset)
            if (dummy_run is None and self.dummy_run) or dummy_run:
                with self.Timer("MDS"):
                    umap_trans = MDS(n_components=2, dissimilarity="precomputed").fit(
                        self.dm
                    )
                std_trans = umap_trans
                self.umap_emb = umap_trans.embedding_
                self.std_emb = umap_trans.embedding_
                self.umap_r_orig = np.random.rand(self.dm.shape[0])
                self.std_r_orig = np.random.rand(self.dm.shape[0])
            else:
                with self.Timer("DensMAP"):
                    self.umap_trans = self.umap_fit_cluster(
                        self.dm, random_state=umap_random_state
                    )
                    self.std_trans = self.umap_fit_vis(
                        self.dm, random_state=umap_random_state
                    )
                    self.umap_emb, self.umap_r_orig = self.extract_emb_rad(
                        self.umap_trans
                    )[:2]
                    self.std_emb, self.std_r_orig = self.extract_emb_rad(
                        self.std_trans
                    )[:2]

            # HDBSCAN*
            clusterer = self.cluster(self.umap_emb)
            self.labels = self.extract_labels_probs(clusterer)[0]

            # np.unique while preserving order https://stackoverflow.com/a/12926989/13697228
            lbl_ids = np.unique(self.labels, return_index=True)[1]
            self.unique_labels = [self.labels[lbl_id] for lbl_id in sorted(lbl_ids)]

            self.train_labels = self.labels[train_ids]
            self.val_labels = self.labels[val_ids]

            # Probability Density Function Summation
            if self.pdf:
                with self.Timer("gridded-pdf-summation"):
                    self.pdf_x, self.pdf_y, self.pdf_sum = self.mvn_prob_sum(
                        self.std_emb, self.std_r_orig
                    )

            # validation density contributed by training densities
            train_emb = self.umap_emb[: self.ntrain]
            train_r_orig = self.umap_r_orig[: self.ntrain]
            val_emb = self.umap_emb[self.ntrain :]
            val_r_orig = self.umap_r_orig[self.ntrain :]

            if count_repeats:
                counts = self.train_df["count"]
                train_r_orig = [r / count for (r, count) in zip(train_r_orig, counts)]

            self.train_df["emb"] = list(map(tuple, train_emb))
            self.train_df["r_orig"] = train_r_orig
            self.val_df["emb"] = list(map(tuple, val_emb))
            self.val_df["r_orig"] = val_r_orig

            with self.Timer("train-val-pdf-summation"):
                # TODO: factor in repeats (df["count"]) with flag
                mvn_list = list(
                    map(my_mvn, train_emb[:, 0], train_emb[:, 1], train_r_orig)
                )
                pdf_list = [mvn.pdf(val_emb) for mvn in mvn_list]
                self.val_dens = np.sum(pdf_list, axis=0)
                self.val_log_dens = np.log(self.val_dens)

            self.val_df["dens"] = self.val_dens

            # cluster-wise predicted average
            cluster_pred = np.array(
                [pred.ravel()[self.labels == lbl] for lbl in self.unique_labels],
                dtype=object,
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

            # # Scale and weight the cluster data
            # # REVIEW: Which Scaler to use? RobustScaler means no set limits on "score"
            self.cluster_score = self.weighted_score(self.cluster_avg, self.val_frac)

            # compound-wise scores (i.e. individual compounds)
            with self.Timer("nearest-neighbor-properties"):
                self.rad_neigh_avg_targ, self.k_neigh_avg_targ = nearest_neigh_props(
                    self.dm, pred, n_neighbors=self.n_peak_neighbors
                )
                self.val_rad_neigh_avg = self.rad_neigh_avg_targ[val_ids]
                self.val_k_neigh_avg = self.k_neigh_avg_targ[val_ids]

        elif self.proxy_weight == 0 and self.novelty_learner == "discover":
            self.val_rad_neigh_avg = np.zeros_like(self.val_pred)
            self.val_k_neigh_avg = np.zeros_like(self.val_pred)
            self.val_dens = np.zeros_like(self.val_pred)

            self.val_df["emb"] = list(
                map(tuple, np.zeros((self.val_df.shape[0], 2)).tolist())
            )
            self.val_df["dens"] = np.zeros(self.val_df.shape[0])

        elif self.proxy_weight != 0 and self.novelty_learner != "discover":
            warn(
                f"self.val_rad_neigh_avg` and `self.val_k_neigh_avg` are being assigned the same values as `val_dens` for compatibility reasons since a non-DiSCoVeR novelty learner was specified: {self.novelty_learner}."
            )
            # composition-based featurization
            X_train: Union[pd.DataFrame, np.ndarray, List] = []
            X_val: Union[pd.DataFrame, np.ndarray, List] = []
            assert self.train_formula is not None
            if self.novelty_prop == "mod_petti":
                assert isinstance(X_train, list)
                assert isinstance(X_val, list)
                for comp in self.train_formula.tolist():
                    X_train.append(ElMD(comp, metric=self.novelty_prop).feature_vector)
                for comp in self.val_formula.tolist():
                    X_val.append(ElMD(comp, metric=self.novelty_prop).feature_vector)
                X_train = np.array(X_train)
                X_val = np.array(X_val)
            else:
                train_df = pd.DataFrame(
                    {
                        "formula": self.train_formula,
                        "target": np.zeros_like(self.train_formula),
                    }
                )
                val_df = pd.DataFrame(
                    {
                        "formula": self.val_formula,
                        "target": np.zeros_like(self.val_formula),
                    }
                )
                X_train, y, formulae, skipped = generate_features(
                    train_df, elem_prop=self.novelty_prop
                )
                X_val, y, formulae, skipped = generate_features(
                    val_df, elem_prop=self.novelty_prop
                )
                # https://stackoverflow.com/a/30808571/13697228
                assert isinstance(X_train, pd.DataFrame)
                X_train = X_train.filter(regex="avg_*")
                X_val = X_val.filter(regex="avg_*")

            # novelty
            assert not isinstance(self.novelty_learner, str)
            self.novelty_learner.fit(X_train)
            self.val_labels = self.novelty_learner.predict(X_val)
            self.train_dens = self.novelty_learner.negative_outlier_factor_
            self.val_dens = self.novelty_learner.score_samples(X_val)
            mn = min(min(self.train_dens), min(self.val_dens))
            self.train_dens = self.train_dens
            self.val_dens = self.val_dens
            pos_train_dens = self.train_dens + 1.001 - mn
            pos_val_dens = self.val_dens + 1.001 - mn
            self.val_log_dens = np.log(pos_val_dens)
            self.val_rad_neigh_avg = copy(pos_val_dens)
            self.val_k_neigh_avg = copy(pos_val_dens)

            self.train_r_orig = 1 / (pos_train_dens)
            self.val_r_orig = 1 / (pos_val_dens)

            pca = PCA(n_components=2)
            X_all = np.concatenate((X_train, X_val))
            pca.fit(X_all)
            emb = pca.transform(X_all)
            self.train_df["emb"] = list(map(tuple, emb[: self.ntrain]))
            self.val_df["emb"] = list(map(tuple, emb[self.ntrain :]))
            self.val_df["dens"] = self.val_dens
            self.train_df["r_orig"] = self.train_r_orig
            self.val_df["r_orig"] = self.val_r_orig

        self.rad_score = self.weighted_score(
            self.val_pred,
            self.val_rad_neigh_avg,
            pred_weight=pred_weight,
            proxy_weight=proxy_weight,
        )
        self.peak_score = self.weighted_score(
            self.val_pred,
            self.val_k_neigh_avg,
            pred_weight=pred_weight,
            proxy_weight=proxy_weight,
        )
        self.dens_score = self.weighted_score(
            self.val_pred,
            self.val_dens,
            pred_weight=pred_weight,
            proxy_weight=proxy_weight,
        )

        # Plotting
        if (self.plotting and plotting is None) or plotting:
            self.plot()

        self.rad_score_df = self.sort(self.rad_score)
        self.peak_score_df = self.sort(self.peak_score)
        self.dens_score_df = self.sort(self.dens_score)

        # create dir https://stackoverflow.com/a/273227/13697228
        Path(self.table_dir).mkdir(parents=True, exist_ok=True)

        self.merge()

        if return_peak:
            return self.dens_score, self.peak_score
        else:
            return self.dens_score

    def weighted_score(
        self,
        pred,
        proxy,
        pred_weight=None,
        proxy_weight=None,
        pred_scaler=None,
        proxy_scaler=None,
    ):
        """Calculate weighted discovery score using the predicted target and proxy.

        Parameters
        ----------
        pred : 1D Array
            Predicted target property values.
        proxy : 1D Array
            Predicted proxy values (e.g. density or peak proxies).
        pred_weight : int, optional
            The weight to assign to the scaled predictions, by default 1
        proxy_weight : int, optional
            The weight to assign to the scaled proxies, by default 1

        Returns
        -------
        1D array
            Discovery scores.
        """
        if self.pred_weight is not None and pred_weight is None:
            pred_weight = self.pred_weight
        elif self.pred_weight is None and pred_weight is None:
            pred_weight = 1

        if self.proxy_weight is not None and proxy_weight is None:
            proxy_weight = self.proxy_weight
        elif self.proxy_weight is None and proxy_weight is None:
            proxy_weight = 1

        pred = pred.ravel().reshape(-1, 1)
        proxy = proxy.ravel().reshape(-1, 1)
        # Scale and weight the cluster data
        if pred_scaler is None:
            self.pred_scaler = self.Scaler().fit(pred)
        pred_scaled = pred_weight * self.pred_scaler.transform(pred)
        if proxy_scaler is None:
            self.proxy_scaler = self.Scaler().fit(-1 * proxy)
        proxy_scaled = proxy_weight * self.proxy_scaler.transform(-1 * proxy)

        # combined cluster data
        comb_data = pred_scaled + proxy_scaled
        comb_scaler = self.Scaler().fit(comb_data)

        # cluster scores range between 0 and 1
        score = comb_scaler.transform(comb_data).ravel()
        return score

    def sort(self, score, proxy_name="density"):
        """Sort (rank) compounds by their proxy score.

        Parameters
        ----------
        score : 1D Array
            Discovery scores for the given proxy given by `proxy_name`.

        proxy_name : string, optional
            Name of the proxy, by default "density". Possible values are "density"
            (`self.val_dens`), "peak" (`self.k_neigh_avg`), and "radius"
            (`self.val_rad_neigh_avg`).

        Returns
        -------
        DataFrame
            Contains "formula", "prediction", `proxy_name`, and "score".
        """
        proxy_lookup = {
            "density": self.val_dens,
            "peak": self.val_k_neigh_avg,
            "radius": self.val_rad_neigh_avg,
        }
        proxy = proxy_lookup[proxy_name]

        score_df = pd.DataFrame(
            {
                "formula": self.val_df.formula,
                "prediction": self.val_pred,
                proxy_name: proxy,
                "score": score,
                "index": self.val_df.index,
            }
        )
        score_df.sort_values("score", ascending=False, inplace=True)
        return score_df

    def merge(self, nscores=100):
        """Perform an outer merge of the density and peak proxy rankings.

        Returns
        -------
        DataFrame
            Outer merge of the two proxy rankings.
        """
        if self.nscores is not None:
            nscores = self.nscores
        dens_score_df = self.dens_score_df.rename(columns={"score": "Density Score"})
        peak_score_df = self.peak_score_df.rename(columns={"score": "Peak Score"})

        comb_score_df = dens_score_df[["formula", "Density Score"]].merge(
            peak_score_df[["formula", "Peak Score"]], how="outer"
        )

        comb_formula = dens_score_df.head(nscores)[["formula"]].merge(
            peak_score_df.head(nscores)[["formula"]], how="outer"
        )

        self.comb_out_df = comb_score_df[np.isin(comb_score_df.formula, comb_formula)]

        dens_score_df.head(nscores).to_csv(
            join(self.table_dir, "dens-score.csv"), index=False, float_format="%.3f"
        )
        peak_score_df.head(nscores).to_csv(
            join(self.table_dir, "peak-score.csv"), index=False, float_format="%.3f"
        )
        self.comb_out_df.to_csv(
            join(self.table_dir, "comb-score.csv"), index=False, float_format="%.3f"
        )

        return self.comb_out_df

    def group_cross_val(self, df, umap_random_state=None, dummy_run=None):
        """Perform leave-one-cluster-out cross-validation (LOCO-CV).

        Parameters
        ----------
        df : DataFrame
            Contains "formula" and "target" (all the data)
        umap_random_state : int, optional
            Random state to use for DensMAP embedding, by default None
        dummy_run : bool, optional
            Whether to perform a "dummy run" (i.e. use multi-dimensional scaling which is faster), by default None

        Returns
        -------
        float
            Scaled, weighted error based on Wasserstein distance (i.e. a sorting distance).

        Raises
        ------
        ValueError
            Needs to have at least one cluster. It is assumed that there will always be a non-cluster
            (i.e. unclassified points) if there is only 1 cluster.

        Notes
        -----
        TODO: highest mean vs. highest single target value
        """
        # TODO: remind people in documentation to use a separate Discover() instance if they wish to access fit *and* gcv attributes
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
        else:
            umap_trans = self.umap_fit_cluster(self.dm, random_state=umap_random_state)
            self.umap_emb = self.extract_emb_rad(umap_trans)[0]

        # HDBSCAN*
        clusterer = self.cluster(self.umap_emb)
        self.labels = self.extract_labels_probs(clusterer)[0]

        # Group Cross Validation Setup
        logo = LeaveOneGroupOut()

        self.n_clusters = np.max(self.labels)
        if self.n_clusters <= 0:
            raise ValueError("no clusters or only one cluster")
        n_test_clusters = np.max([1, int(self.n_clusters * 0.1)])

        """Difficult to repeatably set aside a test set for a given clustering result;
        by changing clustering parameters, the test clusters might change, so using
        seed=42 for UMAP random_state, then seed=10 for setting aside clusters.
        Note that "unclustered" is never assigned as a test_cluster, and so is always
        included in tv_cluster_ids (tv===train_validation)."""
        # np.random.default_rng(seed=10)
        # test_cluster_ids = np.random.choice(self.n_clusters + 1, n_test_clusters)
        # np.random.default_rng()

        # test_ids = np.isin(self.labels, test_cluster_ids)
        # tv_cluster_ids = np.setdiff1d(
        #     np.arange(-1, self.n_clusters + 1), test_cluster_ids
        # )
        # tv_ids = np.isin(self.labels, tv_cluster_ids)

        X, y = self.all_formula, self.all_target
        # X_test, y_test = X[test_ids], y[test_ids]  # noqa
        # X_tv, y_tv, labels_tv = X[tv_ids], y[tv_ids], self.labels[tv_ids]

        # Group Cross Validation
        # for train_index, val_index in logo.split(X_tv, y_tv, self.labels):
        # TODO: group cross-val parity plot
        if self.verbose:
            print("Number of iterations (i.e. clusters): ", self.n_clusters)

        avg_targ = [
            self.single_group_cross_val(X, y, train_index, val_index, i)
            for i, (train_index, val_index) in enumerate(logo.split(X, y, self.labels))
        ]
        out = np.array(avg_targ).T
        self.true_avg_targ, self.pred_avg_targ, self.train_avg_targ = out

        # np.unique while preserving order https://stackoverflow.com/a/12926989/13697228
        lbl_ids = np.unique(self.labels, return_index=True)[1]
        self.avg_labels = [self.labels[lbl_id] for lbl_id in sorted(lbl_ids)]

        self.error, self.dummy_error, self.scaled_error = cdf_sorting_error(
            self.true_avg_targ, self.pred_avg_targ, y_dummy=self.train_avg_targ
        )
        if self.verbose:
            print("Weighted group cross validation error: ", self.error)
            print("Weighted group cross validation scaled error: ", self.scaled_error)

        return self.scaled_error

    def single_group_cross_val(self, X, y, train_index, val_index, iter):
        """Perform leave-one-cluster-out cross-validation.

        Parameters
        ----------
        X : list of str
            Chemical formulae.
        y : 1d array of float
            Target properties.
        train_index, val_index : 1d array of int
            Training and validation indices for a given split, respectively.
        iter : int
            Iteration (i.e. how many clusters have been processed so far).

        Returns
        -------
        true_avg_targ, pred_avg_targ, train_avg_targ : 1d array of float
            True, predicted, and training average targets for each of the clusters.
            average target is used to create a "dummy" measure of performance (i.e. one
            of the the simplest "models" you can use, the average of the training data).
        """
        if self.verbose:
            print("[Iteration: ", iter, "]")

        Xn, yn = X.to_numpy(), y.to_numpy()
        X_train, X_val = Xn[train_index], Xn[val_index]
        y_train, y_val = yn[train_index], yn[val_index]

        train_df = pd.DataFrame({"formula": X_train, "target": y_train})
        val_df = pd.DataFrame({"formula": X_val, "target": y_val})
        test_df = None

        # test_df = pd.DataFrame({"formula": X_test, "property": y_test})

        self.crabnet_model = CrabNet(**self.crabnet_kwargs)
        self.crabnet_model.fit(train_df)

        # CrabNet predict output format: (act, pred, formulae, uncert)
        train_pred, train_sigma, train_true = self.crabnet_model.predict(
            train_df, return_uncertainty=True, return_true=True
        )
        val_pred, val_sigma, val_true = self.crabnet_model.predict(
            val_df, return_uncertainty=True, return_true=True
        )
        if test_df is not None:
            test_pred, test_sigma = self.crabnet_model.predict(
                test_df, return_uncertainty=True
            )

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

        return true_avg_targ, pred_avg_targ, train_avg_targ

    def cluster(self, umap_emb, min_cluster_size=50, min_samples=1):
        """Cluster using HDBSCAN*.

        Parameters
        ----------
        umap_emb : nD Array
            DensMAP embedding coordinates.
        min_cluster_size : int, optional
            "The minimum size of clusters; single linkage splits that contain fewer
            points than this will be considered points "falling out" of a cluster rather
            than a cluster splitting into two new clusters." (source: HDBSCAN* docs), by
            default 50
        min_samples : int, optional
            "The number of samples in a neighbourhood for a point to be considered a
            core point." (source: HDBSCAN* docs), by default 1

        Returns
        -------
        clusterer : HDBSCAN class
            HDBSCAN clusterer fitted to UMAP embeddings.
        """
        if min_cluster_size != 50:
            self.hdbscan_kwargs["min_cluster_size"] = min_cluster_size
        if min_samples != 1:
            self.hdbscan_kwargs["min_samples"] = min_samples
        with self.Timer("HDBSCAN*"):
            clusterer = hdbscan.HDBSCAN(**self.hdbscan_kwargs).fit(umap_emb)
        return clusterer

    def extract_labels_probs(self, clusterer):
        """Extract cluster IDs (`labels`) and `probabilities` from HDBSCAN* `clusterer`.

        Parameters
        ----------
        clusterer : HDBSCAN class
            Instantiated HDBSCAN* class for clustering.

        Returns
        -------
        labels_ : ndarray, shape (n_samples, )
            "Cluster labels for each point in the dataset given to fit(). Noisy samples
            are given the label -1." (source: HDBSCAN* docs)

        probabilities_ : ndarray, shape (n_samples, )
            "The strength with which each sample is a member of its assigned cluster.
            Noise points have probability zero; points in clusters have values assigned
            proportional to the degree that they persist as part of the cluster." (source: HDBSCAN* docs)
        """
        labels, probabilities = attrgetter("labels_", "probabilities_")(clusterer)
        return labels, probabilities

    def umap_fit_cluster(self, dm, metric="precomputed", random_state=None):
        """Perform DensMAP fitting for clustering.

        See https://umap-learn.readthedocs.io/en/latest/clustering.html.

        Parameters
        ----------
        dm : ndarray
            Pairwise Element Mover's Distance (`ElMD`) matrix within a single set of
            points.

        metric : str
            Which metric to use for DensMAP, by default "precomputed".

        random_state: int, RandomState instance or None, optional (default: None)
            "If int, random_state is the seed used by the random number generator; If
            RandomState instance, random_state is the random number generator; If None,
            the random number generator is the RandomState instance used by
            `np.random`." (source: UMAP docs)

        Returns
        -------
        umap_trans : UMAP class
            A UMAP class fitted to `dm`.

        See Also
        --------
        umap.UMAP : UMAP class.
        """
        if random_state is not None:
            self.umap_cluster_kwargs["random_state"] = random_state
        if metric != "precomputed":
            self.umap_cluster_kwargs["metric"] = metric
        self.umap_cluster_kwargs["densmap"] = True
        self.umap_cluster_kwargs["output_dens"] = True
        self.umap_cluster_kwargs["dens_lambda"] = self.dens_lambda
        self.umap_cluster_kwargs["low_memory"] = False
        with self.Timer("fit-UMAP"):
            umap_trans = umap.UMAP(**self.umap_cluster_kwargs).fit(dm)
        return umap_trans

    def umap_fit_vis(self, dm, random_state=None):
        """Perform DensMAP fitting for visualization.

        See https://umap-learn.readthedocs.io/en/latest/clustering.html.

        Parameters
        ----------
        dm : ndarray
            Pairwise Element Mover's Distance (`ElMD`) matrix within a single set of
            points.

        random_state: int, RandomState instance or None, optional (default: None)
            "If int, random_state is the seed used by the random number generator; If
            RandomState instance, random_state is the random number generator; If None,
            the random number generator is the RandomState instance used by
            `np.random`." (source: UMAP docs)

        Returns
        -------
        std_trans : UMAP class
            A UMAP class fitted to `dm`.

        See Also
        --------
        umap.UMAP : UMAP class.
        """
        if random_state is not None:
            self.umap_vis_kwargs["random_state"] = random_state
        self.umap_cluster_kwargs["densmap"] = True
        self.umap_cluster_kwargs["output_dens"] = True
        self.umap_cluster_kwargs["dens_lambda"] = self.dens_lambda
        self.umap_cluster_kwargs["low_memory"] = False
        with self.Timer("fit-vis-UMAP"):
            std_trans = umap.UMAP(**self.umap_vis_kwargs).fit(dm)
        return std_trans

    def extract_emb_rad(self, trans):
        """Extract densMAP embedding and radii.

        Parameters
        ----------
        trans : class
            A fitted UMAP class.

        Returns
        -------
        emb :
            UMAP embedding
        r_orig
            original radii
        r_emb
            embedded radii

        See Also
        --------
        umap.UMAP : UMAP class.
        """
        emb, r_orig_log, r_emb_log = attrgetter("embedding_", "rad_orig_", "rad_emb_")(
            trans
        )
        r_orig = np.exp(r_orig_log)
        r_emb = np.exp(r_emb_log)
        return emb, r_orig, r_emb

    def mvn_prob_sum(self, emb, r_orig, n=100):
        """Gridded multivariate normal probability summation.

        Parameters
        ----------
        emb : ndarray
            Clustering embedding.
        r_orig : 1d array
            Original DensMAP radii.
        n : int, optional
            Number of points along the x and y axes (total grid points = n^2), by default 100

        Returns
        -------
        x : 1d array
            x-coordinates
        y : 1d array
            y-coordinates
        pdf_sum : 1d array
            summed densities at the (`x`, `y`) locations
        """
        # multivariate normal probability summation
        mn = np.amin(emb, axis=0)
        mx = np.amax(emb, axis=0)
        x, y = np.mgrid[mn[0] : mx[0] : n * 1j, mn[1] : mx[1] : n * 1j]  # type: ignore
        pos = np.dstack((x, y))

        with self.Timer("pdf-summation"):
            mvn_list = list(map(my_mvn, emb[:, 0], emb[:, 1], r_orig))
            pdf_list = [mvn.pdf(pos) for mvn in mvn_list]
            pdf_sum = np.sum(pdf_list, axis=0)
        return x, y, pdf_sum

    def compute_log_density(self, r_orig=None):
        """Compute the log density based on the radii.

        Parameters
        ----------
        r_orig : 1d array, optional
            The original radii associated with the fitted DensMAP, by default None. If
            None, then defaults to self.std_r_orig.

        Returns
        -------
        self.dens, self.log_dens : 1d array
            Densities and log densities associated with the original radii, respectively.

        Notes
        -----
        Density is approximated as 1/r_orig
        """
        if r_orig is None:
            r_orig = self.std_r_orig
        self.dens = 1 / r_orig
        self.log_dens = np.log(self.dens)
        return self.dens, self.log_dens

    def plot(self, return_pareto_ind: bool = False):
        """Plot and save various cluster and Pareto front figures.

        Parameters
        ----------
        return_pareto_ind : bool, optional
            Whether to return the pareto front indices, by default False

        Returns
        -------
        pk_pareto_ind, dens_pareto_ind : tuple of int
            Pareto front indices for the peak and density proxies, respectively.
        """
        fig, pk_pareto_ind = self.pf_peak_proxy()
        fig, frac_pareto_ind = self.pf_train_contrib_proxy()

        self.umap_cluster_scatter()
        self.px_umap_cluster_scatter()

        self.px_targ_scatter()

        # PDF evaluated on grid of points
        if self.pdf:
            self.dens_scatter()
            self.dens_targ_scatter()

        self.pf_dens_proxy()
        self.pf_frac_proxy()

        # Group cross-validation parity plot
        if self.true_avg_targ is not None:
            fig, dens_pareto_ind = self.gcv_pareto()
        else:
            warn("Skipping group cross-validation plot")

        if return_pareto_ind:
            return pk_pareto_ind, dens_pareto_ind

    def target_scatter(self):
        """Scatter plot of DensMAP embeddings colored by target values."""
        fig = target_scatter(self.std_emb, self.all_target, figure_dir=self.figure_dir)

    def cluster_count_hist(self):
        """Histogram of cluster counts colored by cluster label."""
        fig = cluster_count_hist(self.labels, figure_dir=self.figure_dir)

    def gcv_pareto(self):
        """Cluster-wise group cross-validation parity plot."""
        x = "true cluster avg target"
        y = "pred cluster avg target"
        gcv_df = pd.DataFrame(
            {
                x: self.true_avg_targ,
                y: self.pred_avg_targ,
                "formula": None,
                "cluster ID": np.array(self.avg_labels),
            }
        )
        fig, dens_pareto_ind = pareto_plot(
            gcv_df,
            x=x,
            y=y,
            fpath=join(self.figure_dir, "gcv-pareto"),
            parity_type="max-of-both",
            color="cluster ID",
            pareto_front=False,
            reverse_x=False,
            x_unit=self.target_unit,
            y_unit=self.target_unit,
            use_plotly_offline=self.use_plotly_offline,
        )
        # fig = group_cv_parity(
        #     self.true_avg_targ, self.pred_avg_targ, self.avg_labels
        # )

        return fig, dens_pareto_ind

    def pf_frac_proxy(self):
        """Cluster-wise average vs. cluster-wise validation fraction Pareto plot.

        In other words, the average performance of a cluster vs. cluster novelty."""
        x = "cluster-wise validation fraction"
        y = "cluster-wise average target"
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
            fpath=join(self.figure_dir, "pf-frac-proxy"),
            pareto_front=True,
            reverse_x=False,
            parity_type=None,
            xrange=[0, 1],
            y_unit=self.target_unit,
            use_plotly_offline=self.use_plotly_offline,
        )
        return fig, frac_pareto_ind

    def pf_dens_proxy(self):
        """True targets vs. dens proxy pareto plot (both training and validation)."""
        dens, log_dens = self.compute_log_density()

        x = "log density"
        y = "target"
        dens_df = pd.DataFrame(
            {
                x: log_dens,
                y: self.all_target,
                "formula": self.all_formula,
                "cluster ID": self.labels,
            }
        )
        # FIXME: manually set the lower and upper bounds of the cmap here (or convert to dict)
        # TODO: make the colorscale discrete
        fig, dens_pareto_ind = pareto_plot(
            dens_df,
            x=x,
            y=y,
            fpath=join(self.figure_dir, "pf-dens-proxy"),
            parity_type=None,
            color="cluster ID",
            pareto_front=True,
            y_unit=self.target_unit,
            use_plotly_offline=self.use_plotly_offline,
        )
        return fig

    def dens_targ_scatter(self):
        """Target value scatter plot (colored by target value) overlay on densities."""
        fig = dens_targ_scatter(
            self.std_emb,
            self.all_target,
            self.pdf_x,
            self.pdf_y,
            self.pdf_sum,
            figure_dir=self.figure_dir,
        )
        return fig

    def dens_scatter(self):
        """Density scatter plot, with densities computed via probability density fn."""
        fig = dens_scatter(
            self.pdf_x, self.pdf_y, self.pdf_sum, figure_dir=self.figure_dir
        )
        return fig

    def px_targ_scatter(self):
        """Interactive `targ_scatter` plot."""
        # Interactive scatter plot colored by target values
        x = "DensMAP Dim. 1"
        y = "DensMAP Dim. 2"
        targ_df = pd.DataFrame(
            {
                x: self.std_emb[:, 0],
                y: self.std_emb[:, 1],
                "target": self.all_target,
                "formula": self.all_formula,
            }
        )
        fig = pareto_plot(
            targ_df,
            x=x,
            y=y,
            color="target",
            fpath=join(self.figure_dir, "px-targ-scatter"),
            pareto_front=False,
            parity_type=None,
            color_unit=self.target_unit,
            use_plotly_offline=self.use_plotly_offline,
        )
        return fig

    def px_umap_cluster_scatter(self):
        """Interactive scatter plot of DensMAP embeddings colored by clusters."""
        x = "DensMAP Dim. 1"
        y = "DensMAP Dim. 2"
        umap_df = pd.DataFrame(
            {
                x: self.std_emb[:, 0],
                y: self.std_emb[:, 1],
                "cluster ID": self.labels,
                "formula": self.all_formula,
            }
        )
        fig = pareto_plot(
            umap_df,
            x=x,
            y=y,
            color="cluster ID",
            fpath=join(self.figure_dir, "px-umap-cluster-scatter"),
            pareto_front=False,
            parity_type=None,
            use_plotly_offline=self.use_plotly_offline,
        )
        return fig

    def umap_cluster_scatter(self):
        """Static scatter plot colored by clusters."""
        fig = umap_cluster_scatter(
            self.std_emb, self.labels, figure_dir=self.figure_dir
        )
        return fig

    def pf_peak_proxy(self):
        """Predicted target vs. peak proxy pareto plot.

        Peak proxy gives an idea of how "surprising" the performance is (i.e. a local
        peak in the ElMD space)."""
        x = str(self.n_peak_neighbors) + "_neigh_avg_targ"
        y = "target"
        # TODO: plot for val data only (fixed?)
        peak_df = pd.DataFrame(
            {
                x: self.val_k_neigh_avg,
                y: self.val_pred,
                "formula": self.val_formula,
                "Peak height": self.val_pred - self.val_k_neigh_avg,
                "cluster ID": self.val_labels,
            }
        )
        # peak pareto plot
        # TODO: double check if cluster labels are correct between the two pareto plots
        fig, pk_pareto_ind = pareto_plot(
            peak_df,
            x=x,
            y=y,
            color="cluster ID",
            fpath=join(self.figure_dir, "pf-peak-proxy"),
            pareto_front=True,
            x_unit=self.target_unit,
            y_unit=self.target_unit,
            use_plotly_offline=self.use_plotly_offline,
        )

        return fig, pk_pareto_ind

    def pf_train_contrib_proxy(self):
        """Predicted target vs train contribution to validation log density pareto plot.

        This is only for the validation data. Training data contribution to validation
        log density is a proxy for chemical novelty (i.e. how novel is a given
        validation datapoint relative to the training data)."""
        x = "log validation density"
        y = "validation predictions"
        # cluster-wise average vs. cluster-wise validation log-density
        frac_df = pd.DataFrame(
            {
                x: self.val_log_dens.ravel(),
                y: self.val_pred.ravel(),
                "cluster ID": self.val_labels,
                "formula": self.val_formula,
            }
        )
        # FIXME: manually set the lower and upper bounds of the cmap here (or convert to dict)
        fig, frac_pareto_ind = pareto_plot(
            frac_df,
            x=x,
            y=y,
            color="cluster ID",
            fpath=join(self.figure_dir, "pf-train-contrib-proxy"),
            pareto_front=True,
            parity_type=None,
            y_unit=self.target_unit,
            use_plotly_offline=self.use_plotly_offline,
        )

        return fig, frac_pareto_ind

    # TODO: write function to visualize Wasserstein metric (barchart with height = color)

    def save(self, fpath="disc.pkl", dummy=False):
        """Save Discover model.

        Parameters
        ----------
        fpath : str, optional
            Filepath to which to save, by default "disc.pkl"

        See Also
        --------
        load : load a Discover model.
        """
        if dummy is True:
            warn("Dummy flag set to True. Overwriting fpath to dummy_disc.pkl")
            fpath = "dummy_disc.pkl"

        with open(fpath, "wb") as f:
            pickle.dump(self, f)

    def load(self, fpath="disc.pkl"):
        """Load Discover model.

        Parameters
        ----------
        fpath : str, optional
            Filepath from which to load, by default "disc.pkl"

        Returns
        -------
        Class
            Loaded Discover() model.
        """
        with open(fpath, "rb") as f:
            disc = pickle.load(f)
            return disc

    def data(self, module, **data_kwargs):
        """Grab data from within the subdirectories (modules) of mat_discover.

        Parameters
        ----------
        module : Module
            The module within mat_discover that contains e.g. "train.csv". For example,
            from crabnet.data.materials_data import elasticity
        fname : str, optional
            Filename of text file to open.
        dummy : bool, optional
            Whether to pare down the data to a small test set, by default False
        groupby : bool, optional
            Whether to use groupby_formula to filter identical compositions
        split : bool, optional
            Whether to split the data into train, val, and (optionally) test sets, by default True
        val_size : float, optional
            Validation dataset fraction, by default 0.2
        test_size : float, optional
            Test dataset fraction, by default 0.0
        random_state : int, optional
            seed to use for the train/val/test split, by default 42

        Returns
        -------
        DataFrame
            If split==False, then the full DataFrame is returned directly

        DataFrame, DataFrame
            If test_size == 0 and split==True, then training and validation DataFrames are returned.

        DataFrame, DataFrame, DataFrame
            If test_size > 0 and split==True, then training, validation, and test DataFrames are returned.
        """
        return data(module, **data_kwargs)
