"""
Construct ElM2D plot of a list of inorganic compostions via Element Movers Distance.

Copyright (C) 2021  Cameron Hargreaves

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>

--------------------------------------------------------------------------------

Python Parser Source: https://github.com/Zapaan/python-chemical-formula-parser

Periodic table JSON data: https://github.com/Bowserinator/Periodic-Table-JSON,
updated to include the Pettifor number and modified Pettifor number from
https://iopscience.iop.org/article/10.1088/1367-2630/18/9/093011

Network simplex source modified to use numba from
https://networkx.github.io/documentation/networkx-1.10/_modules/networkx/algorithms/flow/networksimplex.html#network_simplex

Requires umap which may be installed via:
    conda install -c conda-forge umap-learn
"""
# import os
# import json
# from warnings import warn
# from importlib import reload
from operator import attrgetter

# from typing import Optional
# from types import ModuleType

from numba import cuda

from multiprocessing import cpu_count, freeze_support

import numpy as np
import pandas as pd
import pickle as pk

from scipy.spatial.distance import squareform

import umap

import plotly.express as px
import plotly.io as pio

# from tqdm import tqdm
from pqdm.processes import pqdm

from ElMD import ElMD, EMD
from mat_discover.ElM2D.njit_dist_matrix_full import dist_matrix as cpu_dist_matrix

# overriden by ElM2D class if self.target is not None
use_cuda = cuda.is_available()
if use_cuda:
    target = "cuda"
else:
    target = "cpu"

# warn("target: " + target)

# with open("dist_matrix_settings.json", "w") as f:
#     json.dump(settings, f)


# from mat_discover.ElM2D import njit_dist_matrix  # noqa

# reload(njit_dist_matrix)
# # to overwrite env vars (source: https://stackoverflow.com/a/1254379/13697228)
# cpu_dist_matrix = njit_dist_matrix.dist_matrix

# REVIEW: why is it slower now?
# cuda_dist_matrix: Optional[ModuleType]
if use_cuda:
    # from mat_discover.ElM2D import cuda_dist_matrix  # noqa

    # # to overwrite env vars (source: https://stackoverflow.com/a/1254379/13697228)
    # reload(cuda_dist_matrix)
    # gpu_dist_matrix = cuda_dist_matrix.dist_matrix
    from mat_discover.ElM2D.cuda_dist_matrix_full import dist_matrix as gpu_dist_matrix
else:
    gpu_dist_matrix = None


def main():
    """
    Perform a test of basic features such as intersection, sorting, and featurization.

    Returns
    -------
    None.

    """
    df = pd.read_csv("train-debug.csv")
    df_1 = df.head(500)
    df_2 = df.tail(500)
    mapper = ElM2D(metric="mod_petti")
    mapper.intersect(df_1["composition"], df_2["composition"])
    sorted_comps = mapper.sort(df["composition"])
    sorted_comps, sorted_inds = mapper.sort(df["composition"], return_inds=True)
    mapper.featurize()
    print()


class ElM2D:
    """
    Create intercompound EMD distance matrix and embedding via list of formulas.

    Embedding types are:
        - PCA
        - UMAP
    """

    def __init__(
        self,
        formula_list=None,
        n_proc=None,
        n_components=2,
        verbose=True,
        metric="mod_petti",
        chunksize=1,
        umap_kwargs={},
        emd_algorithm="wasserstein",
        target=None,
    ):
        """
        Initialize parameters for Element Mover's Distance.

        Parameters
        ----------
        formula_list : list of str, optional
            List of chemical formulas, by default None
        n_proc : int, optional
            Number of processors to use (deprecated), by default None
        n_components : int, optional
            Number of embedding dimensions, by default 2
        verbose : bool, optional
            Whether to output verbose information, by default True
        metric : str, optional
            Which type of periodic element properties to use, by default "mod_petti"
        chunksize : int, optional
            Size of chunks for multiprocessing (deprecated), by default 1
        umap_kwargs : dict, optional
            Arguments to pass into umap_kwargs, by default {}
        emd_algorithm : str, optional
            How to compute the earth mover's distances, by default "wasserstein"
        target : str, optional
            Compute device to use: "cuda" or "cpu". If None, defaults to
            fit() "target". If fit() target value is also None, uses "cuda"
            if compatible GPU is available, otherwise "cpu", by default None
        """
        self.verbose = verbose

        if n_proc is None:
            self.n_proc = cpu_count()
        else:
            self.n_proc = n_proc

        self.formula_list = formula_list  # Input formulae

        self.metric = metric
        self.chunksize = chunksize

        self.umap_kwargs = umap_kwargs

        self.umap_kwargs["n_components"] = n_components
        self.umap_kwargs["metric"] = "precomputed"

        self.input_mat = None  # Pettifor vector representation of formula
        self.embedder = None  # For accessing UMAP object
        self.embedding = None  # Stores the last embedded coordinates
        self.dm = None  # Stores distance matrix
        self.emd_algorithm = emd_algorithm
        self.target = target  # "cuda" or "cpu"

    def save(self, filepath):
        """
        Save all variables except for the distance matrix.

        Parameters
        ----------
        filepath : str
            Filepath for which to save the pickle.

        Returns
        -------
        None.

        """
        save_dict = {k: v for k, v in self.__dict__.items()}
        f_handle = open(filepath + ".pk", "wb")
        pk.dump(save_dict, f_handle)
        f_handle.close()

    def load(self, filepath):
        """
        Load variables from pickle file.

        Parameters
        ----------
        filepath : str
            Filepath for which to load the pickle.

        Returns
        -------
        None.

        """
        f_handle = open(filepath + ".pk", "rb")
        load_dict = pk.load(f_handle)
        f_handle.close()

        for k, v in load_dict.items():
            self.__dict__[k] = v

    def plot(self, fp=None, color=None, embedding=None):
        """
        Generate plots based on embedding dimension.

        Parameters
        ----------
        fp : str, optional
            Filepath for which to write plotly html. The default is None.
        color : str, optional
            Color to use for scatter points. The default is None.
        embedding : 2D array, optional
            From self.embedding if not specified. The default is None.

        Returns
        -------
        fig : plotly figure
            Handle to the plotly figure.

        """
        if self.embedding is None:
            print("No embedding in memory, call transform() first.")
            return

        if embedding is None:
            embedding = self.embedding

        x = embedding[:, 0]
        y = embedding[:, 1]
        if embedding.shape[1] == 2:
            if color is None:
                df = pd.DataFrame({"x": x, "y": y, "formula": self.formula_list})
                fig = px.scatter(
                    df,
                    x="x",
                    y="y",
                    hover_name="formula",
                    hover_data={"x": False, "y": False},
                )

            else:
                df = pd.DataFrame(
                    {
                        "x": x,
                        "y": y,
                        "formula": self.formula_list,
                        color.name: color.to_numpy(),
                    }
                )
                fig = px.scatter(
                    df,
                    x="x",
                    y="y",
                    color=color.name,
                    hover_data={
                        "formula": True,
                        color.name: True,
                        "x": False,
                        "y": False,
                    },
                )

        elif embedding.shape[1] == 3:
            z = embedding[:, 2]
            if color is None:
                df = pd.DataFrame(
                    {"x": x, "y": y, "z": z, "formula": self.formula_list}
                )
                fig = px.scatter_3d(
                    df,
                    x="x",
                    y="y",
                    z="z",
                    hover_name="formula",
                    hover_data={"x": False, "y": False, "z": False},
                )

            else:
                df = pd.DataFrame(
                    {
                        "x": x,
                        "y": y,
                        "z": z,
                        "formula": self.formula_list,
                        color.name: color.to_numpy(),
                    }
                )
                fig = px.scatter_3d(
                    df,
                    x="x",
                    y="y",
                    z="z",
                    color=color.name,
                    hover_data={
                        "formula": True,
                        color.name: True,
                        "x": False,
                        "y": False,
                        "z": False,
                    },
                )

        elif embedding.shape[1] > 3:
            print("Too many dimensions to plot directly, using first three components")
            fig = self.plot(fp=fp, color=color, embedding=embedding[:, :3])
            return fig

        if fp is not None:
            pio.write_html(fig, fp)

        return fig

    def fit(self, X, target=None):
        """
        Construct and store an ElMD distance matrix.

        Take an input vector, either of a precomputed distance matrix, or
        an iterable of strings of composition formula, construct an ElMD distance
        matrix and store to self.dm.

        Parameters
        ----------
        X : list of str OR 2D array
            A list of compound formula strings, or a precomputed distance matrix. If
            using a precomputed distance matrix, ensure self.metric == "precomputed"


        Returns
        -------
        None.

        """
        self.formula_list = X
        n = len(X)

        if self.verbose:
            print(f"Fitting {self.metric} kernel matrix")
        if self.metric == "precomputed":
            self.dm = X

        elif n < 5:
            # Do this on a single core for smaller datasets
            distances = []
            print("Small dataset, using single CPU")
            for i in pqdm(range(n - 1)):
                x = ElMD(X[i], metric=self.metric)
                for j in range(i + 1, n):
                    distances.append(x.elmd(X[j]))

            dist_vec = np.array(distances)
            self.dm = squareform(dist_vec)

        else:
            if self.verbose:
                print("Constructing distances")
            if self.emd_algorithm == "network_simplex":
                dist_vec = self._process_list(X, n_proc=self.n_proc)
                self.dm = squareform(dist_vec)
            elif self.emd_algorithm == "wasserstein":
                self.dm = self.EM2D(X, X, target=target)

    def fit_transform(self, X, y=None, how="UMAP", n_components=2, target=None):
        """
        Successively call fit and transform.

        Parameters
        ----------
        X : list of str
            Compositions to embed.
        y : 1D numerical array, optional
            Target values to use for supervised UMAP embedding. The default is None.
        how : str, optional
            How to perform embedding ("UMAP" or "PCA"). The default is "UMAP".
        n_components : int, optional
            Number of dimensions to embed to. The default is 2.

        Returns
        -------
        embedding : TYPE
            DESCRIPTION.

        """
        self.fit(X, target=target)
        embedding = self.transform(
            how=how, n_components=self.umap_kwargs["n_components"], y=y
        )
        return embedding

    def transform(self, how="UMAP", n_components=2, y=None):
        """
        Call the selected embedding method (UMAP or PCA) and embed.

        Parameters
        ----------
        how : str, optional
            How to perform embedding ("UMAP" or "PCA"). The default is "UMAP".
            The default is "UMAP".
        n_components : int, optional
            Number of dimensions to embed to. The default is 2.
        y : 1D numerical array, optional
            Target values to use for supervised UMAP embedding. The default is None.

        Returns
        -------
        2D array
            UMAP or PCA embedding.

        """
        """

        """
        if self.dm is None:
            print("No distance matrix computed, run fit() first")
            return

        n = self.umap_kwargs["n_components"]
        if how == "UMAP":
            if y is None:
                if self.verbose:
                    print(f"Constructing UMAP Embedding to {n} dimensions")
                self.embedder = umap.UMAP(**self.umap_kwargs)
                self.embedding = self.embedder.fit_transform(self.dm)

            else:
                y = y.to_numpy(dtype=float)
                if self.verbose:
                    print(
                        f"Constructing UMAP Embedding to {n} dimensions, with \
                            a targeted embedding"
                    )
                self.embedder = umap.UMAP(**self.umap_kwargs)
                self.embedding = self.embedder.fit_transform(self.dm, y)

        elif how == "PCA":
            if self.verbose:
                print(f"Constructing PCA Embedding to {n} dimensions")
            self.embedding = self.PCA(n_components=self.umap_kwargs["n_components"])
            if self.verbose:
                print("Finished Embedding")

        return self.embedding

    def sort(self, formula_list=None):
        """
        Sorts compositions based on their ElMD similarity.

        Usage:
        mapper = ElM2D()
        sorted_comps = mapper.sort(df["formula"])
        sorted_comps, sorted_inds = mapper.sort(df["formula"], return_inds=True)

        sorted_indices = mapper.sort()
        sorted_comps = mapper.sorted_comps
        """
        if formula_list is None and self.formula_list is None:
            # TODO Exceptions?
            raise Exception(
                "Must input a list of compositions or fit a list of compositions first"
            )

        elif formula_list is None:
            formula_list = self.formula_list

        elif self.formula_list is None:
            E = ElMD(metric=self.metric)
            formula_list = map(E, formula_list, chunksize=self.chunksize)
            self.formula_list = formula_list

        sorted_comps = sorted(formula_list)
        self.sorted_comps = sorted_comps

        return sorted_comps

    def cross_validate(self, y=None, X=None, k=5, shuffle=True, seed=42):
        """
        Cross validate with K-Folds.

        Splits the formula_list into k equal sized partitions and returns five
        tuples of training and test sets. Returns a list of length k, each item
        containing 2 (4 with target data) numpy arrays of formulae of
        length n - n/k and n/k.

        Parameters
        ----------
            y=None: (optional) a numpy array of target properties to cross validate
            k=5: Number of k-folds
            shuffle=True: whether to shuffle the input formulae or not

        Usage
        -----
            cvs = mapper.cross_validate()
            for i, (X_train, X_test) in enumerate(cvs):
                sub_mapper = ElM2D()
                sub_mapper.fit(X_train)
                sub_mapper.save(f"train_elm2d_{i}.pk")
                sub_mapper.fit(X_test)
                sub_mapper.save(f"test_elm2d_{i}.pk")
            ...
            cvs = mapper.cross_validate(y=df["target"])
            for X_train, X_test, y_train, y_test in cvs:
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                errors.append(mae(y_pred, y_test))
            print(np.mean(errors))
        """
        inds = np.arange(len(self.formula_list))  # TODO Exception

        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(inds)

        if X is None:
            formulas = self.formula_list.to_numpy(str)[inds]

        else:
            formulas = X

        splits = np.array_split(formulas, k)

        X_ret = []

        for i in range(k):
            train_splits = np.delete(np.arange(k), i)
            X_train = splits[train_splits[0]]

            for index in train_splits[1:]:
                X_train = np.concatenate((X_train, splits[index]))

            X_test = splits[i]
            X_ret.append((X_train, X_test))

        if y is None:
            return X_ret

        y = y.to_numpy()[inds]
        y_splits = np.array_split(y, k)
        y_ret = []

        for i in range(k):
            train_splits = np.delete(np.arange(k), i)
            y_train = y_splits[train_splits[0]]

            for index in train_splits[1:]:
                y_train = np.concatenate((y_train, y_splits[index]))

            y_test = y_splits[i]
            y_ret.append((y_train, y_test))

        return [(X_ret[i][0], X_ret[i][1], y_ret[i][0], y_ret[i][1]) for i in range(k)]

    def _process_list(self, formula_list, n_proc):
        """
        Process a list of formulas into a pairwise distance matrix.

        Parameters
        ----------
        formula_list : list of str
            Chemical formulas.
        n_proc : int
            number of processors (i.e. CPU cores).

        Returns
        -------
        2D array
            2D distance matrix.

        Given an iterable list of formulas in composition form use multiple processes
        to convert these to pettifor ratio vector form, and then calculate the
        distance between these pairings, returning this as a condensed distance vector.
        """
        pool_list = []

        n_elements = len(ElMD(metric=self.metric).periodic_tab)
        self.input_mat = np.ndarray(
            shape=(len(formula_list), n_elements), dtype=np.float64
        )

        if self.verbose:
            print("Parsing Formula")
            for i, formula in pqdm(list(enumerate(formula_list))):
                self.input_mat[i] = ElMD(formula, metric=self.metric).ratio_vector
        else:
            for i, formula in enumerate(formula_list):
                self.input_mat[i] = ElMD(formula, metric=self.metric).ratio_vector

        # Create input pairings
        if self.verbose:
            print("Constructing joint compositional pairings")
            for i in pqdm(range(len(formula_list) - 1)):
                sublist = [(i, j) for j in range(i + 1, len(formula_list))]
                pool_list.append(sublist)
        else:
            for i in range(len(formula_list) - 1):
                sublist = [(i, j) for j in range(i + 1, len(formula_list))]
                pool_list.append(sublist)

        # Distribute amongst processes
        if self.verbose:
            print(
                "Creating Process Pool\nScattering compositions between processes \
                    and computing distances"
            )

        # scores = process_map(self._pool_ElMD, pool_list, chunksize=self.chunksize)
        scores = map(self._pool_ElMD, pool_list)

        if self.verbose:
            print("Distances computed closing processes")

        if self.verbose:
            print("Flattening sublists")
        # Flattens list of lists to single list
        distances = np.array(
            [dist for sublist in scores for dist in sublist], dtype=np.float64
        )

        return distances

    def EM2D(self, formulas, formulas2=None, target=None):
        """
        Earth Mover's 2D distances. See also EMD.

        Parameters
        ----------
        formulas : list of str
            First list of formulas for which to compute distances. If only formulas
            is specified, then a `pdist`-like array is returned, i.e. pairwise
            distances within a single set.
        formulas2 : list of str, optional
                Second list of formulas, which if specified, causes `cdist`-like
                behavior (i.e. pairwise distances between two sets).

        Returns
        -------
        2D array
            Pairwise distances.

        """
        isXY = formulas2 is None
        E = ElMD(metric=self.metric)

        def gen_ratio_vector(comp):
            """Create a numpy array from a composition dictionary."""
            if isinstance(comp, str):
                comp = E._parse_formula(comp)
                comp = E._normalise_composition(comp)

            sorted_keys = sorted(comp.keys())
            comp_labels = [E._get_position(k) for k in sorted_keys]
            comp_ratios = [comp[k] for k in sorted_keys]

            indices = np.array(comp_labels, dtype=np.int64)
            ratios = np.array(comp_ratios, dtype=np.float64)

            numeric = np.zeros(shape=len(E.periodic_tab), dtype=np.float64)
            numeric[indices] = ratios

            return numeric

        def gen_ratio_vectors(comps):
            return np.array([gen_ratio_vector(comp) for comp in comps])

        U_weights = gen_ratio_vectors(formulas)
        if isXY:
            V_weights = gen_ratio_vectors(formulas2)

        self.lookup, self.periodic_tab = attrgetter("lookup", "periodic_tab")(E)

        def get_mod_petti(x):
            mod_petti = [
                self.periodic_tab[self.lookup[a]] if b > 0 else 0
                for a, b in enumerate(x)
            ]  # FIXME: apparently might output an array of strings
            return mod_petti

        def get_mod_pettis(X):
            mod_pettis = np.array([get_mod_petti(x) for x in X]).astype(float)
            return mod_pettis

        U = get_mod_pettis(U_weights)
        if isXY:
            V = get_mod_pettis(V_weights)

        # decide whether to use cpu or cuda version
        if target is None:
            if (self.target is None or not cuda.is_available()) or self.target == "cpu":
                target = "cpu"
            elif self.target == "cuda" or cuda.is_available():
                target = "cuda"

        # if target == "cpu":
        #     dist_matrix = njit_dist_matrix
        # elif target == "cuda":
        #     dist_matrix = cuda_dist_matrix

        if isXY:
            if target == "cpu":
                distances = cpu_dist_matrix(
                    U,
                    V=V,
                    U_weights=U_weights,
                    V_weights=V_weights,
                    metric="wasserstein",
                )
            elif target == "cuda":
                distances = gpu_dist_matrix(
                    U,
                    V=V,
                    U_weights=U_weights,
                    V_weights=V_weights,
                    metric="wasserstein",
                )
        else:
            if target == "cpu":
                distances = cpu_dist_matrix(
                    U, U_weights=U_weights, metric="wasserstein"
                )
            elif target == "cuda":
                distances = gpu_dist_matrix(
                    U, U_weights=U_weights, metric="wasserstein"
                )

        # package
        self.U = U
        self.U_weights = U_weights

        if isXY:
            self.V = V
            self.V_weights = V_weights

        return distances

    def PCA(self, n_components=5):
        """
        Perform multidimensional scaling (MDS) on a matrix of interpoint distances.

        This finds a set of low dimensional points that have similar interpoint
        distances.
        Source: https://github.com/stober/mds/blob/master/src/mds.py
        """
        if self.dm == []:
            raise Exception(
                "No distance matrix computed, call fit_transform with a list of \
                    compositions, or load a saved matrix with load_dm()"
            )

        (n, n) = self.dm.shape

        if self.verbose:
            print(f"Constructing {n}x{n_components} Gram matrix")
        E = -0.5 * self.dm ** 2

        # Use this matrix to get column and row means
        Er = np.mat(np.mean(E, 1))
        Es = np.mat(np.mean(E, 0))

        # From Principles of Multivariate Analysis: A User's Perspective (page 107).
        F = np.array(E - np.transpose(Er) - Es + np.mean(E))

        if self.verbose:
            print("Computing Eigen Decomposition")
        [U, S, V] = np.linalg.svd(F)

        Y = U * np.sqrt(S)

        if self.verbose:
            print("PCA Projected Points Computed")
        self.mds_points = Y

        return Y[:, :n_components]

    def _pool_ElMD(self, input_tuple):
        """Use multiprocessing module to call the numba compiled EMD function."""
        distances = np.ndarray(len(input_tuple))
        elmd_obj = ElMD(metric=self.metric)

        for i, (input_1, input_2) in enumerate(input_tuple):
            distances[i] = EMD(
                self.input_mat[input_1],
                self.input_mat[input_2],
                elmd_obj.lookup,
                elmd_obj.periodic_tab,
                metric=self.metric,
            )

        return distances

    def __repr__(self):
        """Summary of ElM2D object: length, diversity, and max distance if dm exists."""
        if self.dm is not None:
            return f"ElM2D(size={len(self.formula_list)},  \
                chemical_diversity={np.mean(self.dm)} +/- {np.std(self.dm)}, \
                    maximal_distance={np.max(self.dm)})"
        else:
            return "ElM2D()"

    def export_dm(self, path):
        """Export distance matrix as .csv to path."""
        np.savetxt(path, self.dm, delimiter=",")

    def import_dm(self, path):
        """Import distance matrix from .csv file located at path."""
        self.dm = np.loadtxt(path, delimiter=",")

    def export_embedding(self, path):
        """Export embedding as .csv file to path."""
        np.savetxt(path, self.embedding, delimiter=",")

    def import_embedding(self, path):
        """Import embedding from .csv file located at path."""
        self.embedding = np.loadtxt(path, delimiter=",")

    def _pool_featurize(self, comp):
        """Extract the feature vector for a given composition (comp)."""
        return ElMD(comp, metric=self.metric).feature_vector

    def featurize(self, formula_list=None, how="mean"):
        """Featurize a list of formulas."""
        if formula_list is None and self.formula_list is None:
            raise Exception("You must enter a list of compositions first")

        elif formula_list is None:
            formula_list = self.formula_list

        elif self.formula_list is None:
            self.formula_list = formula_list

        # elmd_obj = ElMD(metric=self.metric)

        # if type(elmd_obj.periodic_tab["H"]) is int:
        #     vectors = np.ndarray(
        #         (len(compositions), len(elmd_obj.periodic_tab))
        #     )
        # else:
        #     vectors = np.ndarray(
        #         (len(compositions), len(elmd_obj.periodic_tab["H"]))
        #     )

        print(
            f"Constructing compositionally weighted {self.metric} feature vectors \
                for each composition"
        )
        # vectors = process_map(
        #     self._pool_featurize, formula_list, chunksize=self.chunksize
        # )
        vectors = map(self._pool_featurize, formula_list)

        print("Complete")

        return np.array(vectors)

    def intersect(self, y=None, X=None):
        """
        Find intersectional distance matrix between two formula lists.

        Takes in a second formula list, y, and computes the intersectional distance
        matrix between the two under the given metric. If a two formula lists
        are given the intersection between the two is computed, returning a
        distance matrix of the form:

              X_0             X_1            X_2            ...
        y_0  ElMD(X_0, y_0)  ElMD(X_1, y_0)  ElMD(X_2, y_0)
        y_1  ElMD(X_0, y_1)  ElMD(X_1, y_1)  ElMD(X_2, y_1)
        f_2  ElMD(X_0, y_2)  ElMD(X_1, y_2)  ElMD(X_2, y_2)
        ...
        """
        if X is None and y is None:
            raise Exception(
                "Must enter two lists of formula or fit a list of formula and enter \
                    an intersecting list of formula"
            )
        if X is None:
            X = self.formula_list

        # elmd = ElMD()
        # dm = []

        # with Pool(cpu_count()) as process_pool:
        #     for comp_1 in tqdm(X):
        #         ionic = process_pool.starmap(
        #             elmd.elmd, ((comp_1, comp_2) for comp_2 in y)
        #         )
        #         dm.append(ionic)

        # distance_matrix = np.array(dm)

        distances = self.EM2D(X, formulas2=y)

        return distances

        # Not working currently, might be faster...
        # intersection_dm = self._process_intersection(X, y, self.n_proc)

        # return intersection_dm

    def _process_intersection(self, X, y, n_proc):
        """Compute the intersection of two lists of compositions."""
        pool_list = []

        n_elements = len(ElMD(metric=self.metric).periodic_tab)
        X_mat = np.ndarray(shape=(len(X), n_elements), dtype=np.float64)
        y_mat = np.ndarray(shape=(len(y), n_elements), dtype=np.float64)

        if self.verbose:
            print("Parsing X Formula")
        for i, formula in pqdm(list(enumerate(X))):
            X_mat[i] = ElMD(formula, metric=self.metric).ratio_vector

        if self.verbose:
            print("Parsing Y Formula")
        for i, formula in pqdm(list(enumerate(y))):
            y_mat[i] = ElMD(formula, metric=self.metric).ratio_vector

        # Create input pairings
        if self.verbose:
            print("Constructing joint compositional pairings")
        for y in pqdm(range(len(y_mat))):
            sublist = [(y, x) for x in range(len(X_mat))]
            pool_list.append(sublist)

        # Distribute amongst processes
        if self.verbose:
            print(
                "Creating Process Pool\nScattering compositions between processes \
                    and computing distances"
            )
        # distances = process_map(self._pool_ElMD, pool_list, chunksize=self.chunksize)
        distances = map(self._pool_ElMD, pool_list)

        if self.verbose:
            print("Distances computed closing processes")

        # if self.verbose: print("Flattening sublists")
        # Flattens list of lists to single list
        # distances = [dist for sublist in scores for dist in sublist]

        return np.array(distances, dtype=np.float64)


if __name__ == "__main__":
    freeze_support()
    main()

# %%
# settings = {
#     "INLINE": "never",
#     "FASTMATH": "1",
#     "COLUMNS": str(n_elements),
#     "USE_64": "0",
#     "TARGET": target,
# }

# os.environ.update(settings)

# from tqdm.contrib.concurrent import process_map
# n_elements = len(ElMD(metric="mod_petti").periodic_tab)

# # number of columns of U and V and other env vars must be set as env var before import
# # HACK: define a wrapper to ElMD() so that you can use a different scale than mod_petti with cuda_dist_matrix
# os.environ["COLUMNS"] = str(n_elements)
# os.environ["USE_64"] = "0"
# os.environ["INLINE"] = "never"
# os.environ["FASTMATH"] = "1"
# os.environ["TARGET"] = "cuda"
