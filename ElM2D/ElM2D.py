"""
A class to construct an ElM2D plot of a list of inorganic compostions based on
the Element Movers Distance Between These.


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

Requies umap which may be installed via
conda install -c conda-forge umap-learn
"""
import os

from multiprocessing import cpu_count, freeze_support

from os.path import join

from itertools import product

import numpy as np
import pandas as pd
import pickle as pk

from scipy.spatial.distance import squareform
from scipy.sparse import coo_matrix

import umap

import plotly.express as px
import plotly.io as pio

from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from ElMD import ElMD, EMD

from operator import attrgetter
from importlib import reload

# number of columns of U and V must be set as env var before import dist_matrix
n_elements = len(ElMD().periodic_tab["mod_petti"])
os.environ["COLUMNS"] = str(n_elements)

# other environment variables (set before importing dist_matrix)
os.environ["USE_64"] = "0"
os.environ["INLINE"] = "never"
os.environ["FASTMATH"] = "1"
os.environ["TARGET"] = "cuda"

import dist_matrix  # noqa

# to overwrite env vars (source: https://stackoverflow.com/a/1254379/13697228)
reload(dist_matrix)
dist_matrix = dist_matrix.dist_matrix


def main():
    datapath = join("train-debug.csv")
    df = pd.read_csv(datapath)

    df_1 = df.head(500)
    df_2 = df.tail(500)
    mapper = ElM2D(metric="mod_petti")
    mapper.intersect(df_1["composition"], df_2["composition"])
    sorted_comps = mapper.sort(df["composition"])
    sorted_comps, sorted_inds = mapper.sort(df["composition"], return_inds=True)
    fts = mapper.featurize()
    print()


class ElM2D:
    """
    This class takes in a list of compound formula and creates the intercompound
    distance matrix wrt EMD and a two dimensional embedding using either PCA or
    UMAP
    """

    def __init__(
        self,
        formula_list=None,
        n_proc=None,
        n_components=None,
        verbose=True,
        metric="mod_petti",
        chunksize=100,
        umap_kwargs={},
        emd_algorithm="wasserstein",
    ):

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

        self.input_mat = []  # Pettifor vector representation of formula
        self.input_mat2 = []  # Pettifor vector representation of 2nd set of formulae
        self.embedder = None  # For accessing UMAP object
        self.embedding = None  # Stores the last embedded coordinates
        self.dm = []  # Stores distance matrix
        self.isXY = False  # Whether to compute between two sets of formula (defaults to single set)
        self.emd_algorithm = emd_algorithm  # which type of Earth Mover's distance

    def save(self, filepath):
        # Save all variables except for the distance matrix
        save_dict = {k: v for k, v in self.__dict__.items()}
        f_handle = open(filepath + ".pk", "wb")
        pk.dump(save_dict, f_handle)
        f_handle.close()

    def load(self, filepath):
        f_handle = open(filepath + ".pk", "rb")
        load_dict = pk.load(f_handle)
        f_handle.close()

        for k, v in load_dict.items():
            self.__dict__[k] = v

    def plot(self, fp=None, color=None, embedding=None):
        if self.embedding is None:
            print("No embedding in memory, call transform() first.")
            return

        if embedding is None:
            embedding = self.embedding

        if embedding.shape[1] == 2:
            if color is None:
                df = pd.DataFrame(
                    {
                        "x": embedding[:, 0],
                        "y": embedding[:, 1],
                        "formula": self.formula_list,
                    }
                )
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
                        "x": embedding[:, 0],
                        "y": embedding[:, 1],
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
            if color is None:
                df = pd.DataFrame(
                    {
                        "x": embedding[:, 0],
                        "y": embedding[:, 1],
                        "z": embedding[:, 2],
                        "formula": self.formula_list,
                    }
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
                        "x": embedding[:, 0],
                        "y": embedding[:, 1],
                        "z": embedding[:, 2],
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

    def fit(self, X):
        """
        Take an input vector, either of a precomputed distance matrix, or
        an iterable of strings of composition formula, construct an ElMD distance
        matrix and store to self.dm.

        Input
        X - A list of compound formula strings, or a precomputed distance matrix
        (ensure self.metric = "precomputed")
        """
        self.formula_list = X
        n = len(X)

        if self.verbose:
            print(f"Fitting {self.metric} kernel matrix")
        if self.metric == "precomputed":
            dm = X

        elif n < 5:
            # Do this on a single core for smaller datasets
            distances = []
            print("Small dataset, using single CPU")
            for i in tqdm(range(n - 1)):
                x = ElMD(X[i], metric=self.metric)
                for j in range(i + 1, n):
                    distances.append(x.elmd(X[j]))

            dist_vec = np.array(distances)
            dm = squareform(dist_vec)

        else:
            if self.verbose:
                print("Constructing distances")
            if self.emd_algorithm == "network_simplex":
                dist_vec = self._process_list(X, n_proc=self.n_proc)
                dm = squareform(dist_vec)
            elif self.emd_algorithm == "wasserstein":
                dm = self.EM2D(X, X)

        if self.dm == []:
            self.dm = dm

    def fit_transform(self, X, y=None, how="UMAP", n_components=None):
        """
        Successively call fit and transform

        Parameters:
        X - List of compositions to embed
        how - "UMAP" or "PCA", the embedding technique to use
        n_components - The number of dimensions to embed to
        """
        if n_components == None:
            n_components = self.umap_kwargs["n_components"]

        self.fit(X)
        embedding = self.transform(how=how, n_components=n_components, y=y)
        return embedding

    def transform(self, how="UMAP", n_components=None, y=None):
        """
        Call the selected embedding method (UMAP or PCA) and embed to
        n_components dimensions.
        """
        if self.dm == []:
            print("No distance matrix computed, run fit() first")
            return

        umap_kwargs = self.umap_kwargs
        if n_components != None:
            umap_kwargs["n_components"] = n_components

        if how == "UMAP":
            if y is None:
                if self.verbose:
                    print(
                        f"Constructing UMAP Embedding to {self.umap_kwargs['n_components']} dimensions"
                    )
                self.embedder = umap.UMAP(**umap_kwargs)
                self.embedding = self.embedder.fit_transform(self.dm)

            else:
                y = y.to_numpy(dtype=float)
                if self.verbose:
                    print(
                        f"Constructing UMAP Embedding to {self.umap_kwargs['n_components']} dimensions, with a targetted embedding"
                    )
                self.embedder = umap.UMAP(**umap_kwargs)
                self.embedding = self.embedder.fit_transform(self.dm, y)

        elif how == "PCA":
            if self.verbose:
                print(
                    f"Constructing PCA Embedding to {self.umap_kwargs['n_components']} dimensions"
                )
            self.embedding = self.PCA(n_components=n_components)
            if self.verbose:
                print("Finished Embedding")

        return self.embedding

    def xy_dist_mat(self, Y, X=[], flat_pool=[]):
        """
        Compute pairwise distance matrix between training data and new set of formulae.

        Parameters
        ----------
        Y : array
            The second set of formula with which to compare to the input features.
        X : array
            The first set of formula with which to compare to the input features. Defaults to []
        flat_pool : array of tuples
            The pairs of indices which correspond to the distances to be computed.

        Returns
        -------
        distance matrix between X (training formulae, rows) and a new set of formulae (Y, columns)

        See Also
        --------
        scipy.spatial: cdist(), distance_matrix()
        """
        if X == []:
            X = self.formula_list

        self.dm2 = self._process_list(
            X, formula_list2=Y, n_proc=self.n_proc, flat_pool=flat_pool
        )
        return self.dm2

    def EM2D(self, formulas, formulas2):
        """
        Earth Mover's 2D distances. See also EMD.

        Parameters
        ----------
        formulas : TYPE
            DESCRIPTION.
        formulas2 : TYPE
            DESCRIPTION.
        lookup : TYPE
            DESCRIPTION.
        table : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        E = ElMD()

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

            numeric = np.zeros(shape=len(E.periodic_tab[E.metric]), dtype=np.float64)
            numeric[indices] = ratios

            return numeric

        def gen_ratio_vectors(comps):
            return np.array([gen_ratio_vector(comp) for comp in comps])

        self.U_weights = gen_ratio_vectors(formulas)
        # self.V_weights = gen_ratio_vectors(formulas2)

        lookup, periodic_tab, metric = attrgetter("lookup", "periodic_tab", "metric")(E)
        ptab_metric = periodic_tab[metric]

        def get_mod_petti(x):
            return [ptab_metric[lookup[a]] if b > 0 else 0 for a, b in enumerate(x)]

        def get_mod_pettis(X):
            return np.array([get_mod_petti(x) for x in X])

        self.U = get_mod_pettis(self.U_weights)
        # self.V = get_mod_pettis(V)

        # distances = dist_matrix(
        #     U, V=V, U_weights=U_weights, V_weights=V_weights, metric="wasserstein",
        # )
        distances = dist_matrix(self.U, U_weights=self.U_weights, metric="wasserstein")
        return distances

    def PCA(self, n_components=5):
        """
        Multidimensional Scaling - Given a matrix of interpoint distances,
        find a set of low dimensional points that have similar interpoint
        distances.
        https://github.com/stober/mds/blob/master/src/mds.py
        """

        if self.dm == []:
            raise Exception(
                "No distance matrix computed, call fit_transform with a list of compositions, or load a saved matrix with load_dm()"
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
            print(f"Computing Eigen Decomposition")
        [U, S, V] = np.linalg.svd(F)

        Y = U * np.sqrt(S)

        if self.verbose:
            print(f"PCA Projected Points Computed")
        self.mds_points = Y

        return Y[:, :n_components]

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
            raise Exception(
                "Must input a list of compositions or fit a list of compositions first"
            )  # TODO Exceptions?

        elif formula_list is None:
            formula_list = self.formula_list

        elif self.formula_list is None:
            formula_list = process_map(ElMD, formula_list, chunksize=self.chunksize)
            self.formula_list = formula_list

        sorted_comps = sorted(formula_list)
        self.sorted_comps = sorted_comps

        return sorted_comps

    def cross_validate(self, y=None, X=None, k=5, shuffle=True, seed=42):
        """
        Implementation of cross validation with K-Folds.

        Splits the formula_list into k equal sized partitions and returns five
        tuples of training and test sets. Returns a list of length k, each item
        containing 2 (4 with target data) numpy arrays of formulae of
        length n - n/k and n/k.

        Parameters:
            y=None: (optional) a numpy array of target properties to cross validate
            k=5: Number of k-folds
            shuffle=True: whether to shuffle the input formulae or not

        Usage:
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

    def _gen_ratio_vector(self, formula):
        """
        Generate compositional ratio vector from a chemical formula.

        Parameters
        ----------
        formula : string
            Chemical formula for which to compute the compositional ratio vector.

        Returns
        -------
        ratio_vector : array
            Compositional ratio vector that encodes the composition of the chemical formula.

        """
        ratio_vector = ElMD(formula, metric=self.metric).ratio_vector
        return ratio_vector

    def _process_list(self, formula_list, formula_list2=[], flat_pool=[], n_proc=1):
        """
        Calculate distances between two lists of formulae in parallel.

        Parameters
        ----------
        formula_list : iterable list
            First set of formulae (rows).
        formula_list2 : iterable list, optional
            Second set of formulae (columns).
        n_proc : int, optional
            Number of processes. The default is 1.
        pool_list : iterable list, optional
            Indices of pairs for which to compute distances. The default is [].

        Returns
        -------
        ARRAY
            Pairwise distances within formula_list, between formula_list and formula_list2, or for pool_list

        """
        n_elements = len(ElMD().periodic_tab[self.metric])
        n = len(formula_list)
        self.input_mat = np.ndarray(shape=(n, n_elements), dtype=np.float64)

        formulas = np.array(formula_list)
        if self.verbose:
            print("Parsing Formulae")
            # formulas = tqdm(formulas)

        self.input_mat = process_map(
            self._gen_ratio_vector, formulas, chunksize=self.chunksize
        )
        # for i, formula in formulas:
        #     self.input_mat[i] = ElMD(formula, metric=self.metric).ratio_vector

        isXY = formula_list2 != []
        generate_pool = flat_pool == []

        if isXY:
            n2 = len(formula_list2)
        else:
            n2 = n

        if isXY:
            self.input_mat2 = np.ndarray(shape=(n2, n_elements), dtype=np.float64)
            formulas = list(enumerate(formula_list2))
            if self.verbose:
                print("Parsing Formula2")
                formulas = tqdm(formulas)

            self.input_mat2 = process_map(
                self._gen_ratio_vector, formulas, chunksize=self.chunksize
            )

        if generate_pool:
            # Create input pairings
            pool_list = []
            if self.verbose:
                print("Constructing joint compositional pairings")

            if isXY:
                # all pairs
                flat_pool = list(product(range(n), range(n2)))
            else:
                # upper triangle only
                flat_pool = list(zip(*np.triu_indices(n)))

        # package some objects for _pool_ElMD
        elmd_obj = ElMD(metric=self.metric)
        self.lookup = elmd_obj.lookup
        self.periodic_tab = elmd_obj.periodic_tab[self.metric]
        self.isXY = isXY  # overwrite

        # Distribute amongst processes
        if self.verbose:
            print("Scattering compositions between processes and computing distances")
        distances = process_map(self._pool_ElMD, flat_pool, chunksize=self.chunksize)
        # for input_tuple in flat_pool:
        #     distances = self._pool_ElMD(input_tuple)
        if self.verbose:
            print("Distances computed closing processes")

        # # Flattens list of lists to single list
        distances = np.array(distances, dtype=np.float64)

        # reassemble into pairwise distance matrix
        i, j = tuple(list(zip(*flat_pool)))
        distances = coo_matrix((distances, (i, j)), shape=(n, n2)).toarray()

        if not isXY:
            # make the matrix symmetric with zeros along diagonal
            i_lower = np.tril_indices(n, -1)
            distances[i_lower] = distances.T[i_lower]
            i_diag = np.diag_indices(n)
            distances[i_diag] = np.zeros(n)

        return distances

    def _pool_ElMD(self, input_tuple):
        """
        Use multiprocessing module to call the numba compiled EMD function.

        Parameters
        ----------
        input_tuple : tuple list
            An index pair between two sets of formulae for which to compute distances.

        Returns
        -------
        numeric scalar
            Distance between pair of formula from two sets.
        """
        # extract EMD lookup and periodic tab
        # if self.lookup == []:
        #     elmd_obj = ElMD(metric=self.metric)
        #     lookup = elmd_obj.lookup
        #     periodic_tab = elmd_obj.periodic_tab[self.metric]
        # else:
        lookup = self.lookup
        periodic_tab = self.periodic_tab

        # unpack tuple and extract input vectors
        input_id, input_id2 = input_tuple
        input_vec = self.input_mat[input_id]
        if self.isXY:
            input_vec2 = self.input_mat2[input_id2]
        else:
            input_vec2 = self.input_mat[input_id2]

        distance = EMD(input_vec, input_vec2, lookup, periodic_tab)

        return distance

    def __repr__(self):
        if self.dm is not None:
            return f"ElM2D(size={len(self.formula_list)},  chemical_diversity={np.mean(self.dm)} +/- {np.std(self.dm)}, maximal_distance={np.max(self.dm)})"
        else:
            return f"ElM2D()"

    def export_dm(self, path):
        np.savetxt(path, self.dm, delimiter=",")

    def import_dm(self, path):
        self.dm = np.loadtxt(path, delimiter=",")

    def export_embedding(self, path):
        np.savetxt(path, self.embedding, delimiter=",")

    def import_embedding(self, path):
        self.embedding = np.loadtxt(path, delimiter=",")

    def _pool_featurize(self, comp):
        return ElMD(comp, metric=self.metric).feature_vector

    def featurize(self, formula_list=None, how="mean"):
        if formula_list is None and self.formula_list is None:
            raise Exception("You must enter a list of compositions first")

        elif formula_list is None:
            formula_list = self.formula_list

        elif self.formula_list is None:
            self.formula_list = formula_list

        elmd_obj = ElMD(metric=self.metric)

        # if type(elmd_obj.periodic_tab[self.metric]["H"]) is int:
        #     vectors = np.ndarray((len(compositions), len(elmd_obj.periodic_tab[self.metric])))
        # else:
        #     vectors = np.ndarray((len(compositions), len(elmd_obj.periodic_tab[self.metric]["H"])))

        print(
            f"Constructing compositionally weighted {self.metric} feature vectors for each composition"
        )
        vectors = process_map(
            self._pool_featurize, formula_list, chunksize=self.chunksize
        )

        print("Complete")

        return np.array(vectors)


if __name__ == "__main__":
    freeze_support()
    main()

"""Code Graveyard"""
"""
# for i, (input_1, input_2) in enumerate(input_tuple):
#     distances[i] = EMD(self.input_mat[input_1],
#                        self.input_mat2[input_2],
#                        elmd_obj.lookup,
#                        elmd_obj.periodic_tab[self.metric])

        #if self.verbose: print("Flattening sublists")
        #distances = [dist for sublist in scores for dist in sublist]
        #distances = distances.reshape((n, n2))

                #distances = np.ndarray(len(input_tuple))

    # def _pool_ElMD(self, input_tuple):
    #     '''
    #     Uses multiprocessing module to call the numba compiled EMD function
    #     '''
    #     distances = np.ndarray(len(input_tuple))
    #     elmd_obj = ElMD(metric=self.metric)

    #     for i, (input_1, input_2) in enumerate(input_tuple):
    #         distances[i] = EMD(self.input_mat[input_1],
    #                            self.input_mat[input_2],
    #                            elmd_obj.lookup,
    #                            elmd_obj.periodic_tab[self.metric])
    #         if distances[i] == None: raise ValueError("Distance should be a real value, not None")

    #     return distances


    # def _process_list(self, formula_list, n_proc, pool_list = []):
    #     '''
    #     Given an iterable list of formulas in composition form
    #     use multiple processes to convert these to pettifor ratio
    #     vector form, and then calculate the distance between these
    #     pairings, returning this as a condensed distance vector
    #     '''

    #     n_elements = len(ElMD().periodic_tab[self.metric])
    #     self.input_mat = np.ndarray(shape=(len(formula_list), n_elements), dtype=np.float64)

    #     if self.verbose:
    #         print("Parsing Formula")
    #         for i, formula in tqdm(list(enumerate(formula_list))):
    #             self.input_mat[i] = ElMD(formula, metric=self.metric).ratio_vector
    #     else:
    #         for i, formula in enumerate(formula_list):
    #             self.input_mat[i] = ElMD(formula, metric=self.metric).ratio_vector

    #     # get_ratio_vector = lambda formula: ElMD(formula, metric=self.metric).ratio_vector
    #     # if self.verbose: print("Parsing Formula")
    #     # self.input_mat = process_map(get_ratio_vector, formula_list, chunksize=self.chunksize)

    #     # Create input pairings
    #     if self.verbose:
    #         print("Constructing joint compositional pairings")
    #         for i in tqdm(range(len(formula_list) - 1)):
    #             sublist = [(i, j) for j in range(i + 1, len(formula_list))]
    #             pool_list.append(sublist)
    #     else:
    #         for i in range(len(formula_list) - 1):
    #             sublist = [(i, j) for j in range(i + 1, len(formula_list))]
    #             pool_list.append(sublist)

    #     # Distribute amongst processes
    #     if self.verbose: print("Creating Process Pool\nScattering compositions between processes and computing distances")

    #     scores = process_map(self._pool_ElMD, pool_list, chunksize=self.chunksize)

    #     if self.verbose: print("Distances computed closing processes")

    #     if self.verbose: print("Flattening sublists")
    #     # Flattens list of lists to single list
    #     distances = [dist for sublist in scores for dist in sublist]
    #     distances = np.array(distances, dtype=np.float64)

    #     return np.array(distances, dtype=np.float64)

    # def intersect(self, y=None, X=None):
    #
    #     Takes in a second formula list, y, and computes the intersectional distance
    #     matrix between the two under the given metric. If a two formula lists
    #     are given the intersection between the two is computed, returning a
    #     distance matrix of the form:

    #           X_0             X_1            X_2            ...
    #     y_0  ElMD(X_0, y_0)  ElMD(X_1, y_0)  ElMD(X_2, y_0)
    #     y_1  ElMD(X_0, y_1)  ElMD(X_1, y_1)  ElMD(X_2, y_1)
    #     y_2  ElMD(X_0, y_2)  ElMD(X_1, y_2)  ElMD(X_2, y_2)
    #     ...
    #
    #     if X is None and y is None:
    #         raise Exception("Must enter two lists of formula or fit a list of formula and enter an intersecting list of formula")
    #     if X is None:
    #         X = self.formula_list

    #     elmd = ElMD()
    #     dm = []
    #     process_pool = Pool(cpu_count())

    #     for comp_1 in tqdm(X):
    #         ionic = process_pool.starmap(elmd.elmd, ((comp_1, comp_2) for comp_2 in y))
    #         dm.append(ionic)

    #     distance_matrix = np.array(dm)

    #     return distance_matrix

    #     # Not working currently, might be faster...
    #     # intersection_dm = self._process_intersection(X, y, self.n_proc)

    #     # return intersection_dm

    # def _process_intersection(self, X, y, n_proc):
    #     '''
    #     Compute the intersection of two lists of compositions
    #     '''
    #     pool_list = []

    #     n_elements = len(ElMD().periodic_tab[self.metric])
    #     X_mat = np.ndarray(shape=(len(X), n_elements), dtype=np.float64)
    #     y_mat = np.ndarray(shape=(len(y), n_elements), dtype=np.float64)

    #     if self.verbose: print("Parsing X Formula")
    #     for i, formula in tqdm(list(enumerate(X))):
    #         X_mat[i] = ElMD(formula, metric=self.metric).ratio_vector

    #     if self.verbose: print("Parsing Y Formula")
    #     for i, formula in tqdm(list(enumerate(y))):
    #         y_mat[i] = ElMD(formula, metric=self.metric).ratio_vector

    #     # Create input pairings
    #     if self.verbose: print("Constructing joint compositional pairings")
    #     for y in tqdm(range(len(y_mat))):
    #         sublist = [(y, x) for x in range(len(X_mat))]
    #         pool_list.append(sublist)

    #     # Distribute amongst processes
    #     if self.verbose: print("Creating Process Pool\nScattering compositions between processes and computing distances")
    #     distances = process_map(self._pool_ElMD, pool_list, chunksize=self.chunksize)

    #     if self.verbose: print("Distances computed closing processes")

    #     # if self.verbose: print("Flattening sublists")
    #     # Flattens list of lists to single list
    #     # distances = [dist for sublist in scores for dist in sublist]

    #     return np.array(distances, dtype=np.float64)

        #df = load_dataset("matbench_expt_gap").head(1001)
        
                    #if self.verbose: print("Constructing joint compositional pairings")
                #id_list = tqdm(id_list)


                # id_list = range(n)
                # id_list2 = range(n2)
                
                # all pairs
                # i, j = np.meshgrid(range(n), range(n2))
                # flat_pool = list(zip(i.ravel(), j.ravel()))
                
                # for i in id_list:
                #     sublist = [(i, j) for j in id_list2]
                #     pool_list.append(sublist)
                
                    #id_list = range(n-1)
                    # for i in id_list:
    #     sublist = [(i, j) for j in range(i + 1, n)]
    #     pool_list.append(sublist)
    
    
                # for i, formula in formulas:
            #     self.input_mat2[i] = ElMD(formula, metric=self.metric).ratio_vector
            
                #id_list = tqdm(id_list)
                
                

            if self.verbose: print("Flattening pool sublists")
            flat_pool = [pool for sublist in pool_list for pool in sublist]
"""
