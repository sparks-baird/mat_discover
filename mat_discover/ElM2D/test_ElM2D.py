"""
Test Element Mover's 2D Distance Matrix via network simplex and "wasserstein" methods.

This test ensures that the fast implementation "wasserstein" produces "close" values
to the original network simplex method.
"""
import os

# import json
import unittest
from importlib import reload

from os.path import join, dirname, relpath

from numpy.testing import assert_allclose

# import numpy as np
# from numba import cuda

from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import squareform

from mat_discover.utils.Timer import Timer
from ElM2D import ElM2D as pip_ElM2D

# from EleMD import EleMD
import pandas as pd

from ElMD import ElMD

target = "cuda"

n_elements = len(ElMD(metric="mod_petti").periodic_tab)

# # number of columns of U and V and other env vars must be set as env var before import
os.environ["COLUMNS"] = str(n_elements)
os.environ["USE_64"] = "0"
os.environ["INLINE"] = "never"
os.environ["FASTMATH"] = "1"
os.environ["TARGET"] = target

from mat_discover.ElM2D import ElM2D_  # noqa

reload(ElM2D_)

custom_ElM2D = ElM2D_.ElM2D


class Testing(unittest.TestCase):
    def test_dm_close(self):
        mapper = custom_ElM2D(metric="mod_petti")
        # df = pd.read_csv("train-debug.csv")
        df = pd.read_csv(join(dirname(relpath(__file__)), "stable-mp.csv"))
        formulas = df["formula"]
        sub_formulas = formulas[0:100]
        with Timer("dm_wasserstein"):
            mapper.fit(sub_formulas, target=target)
            dm_wasserstein = mapper.dm

        # FIXME: njit_dist_matrix not inheriting env vars, are env vars even necessary for njit
        # mapper.fit(sub_formulas, target="cpu")
        # dm_wasserstein = mapper.dm

        # TODO: replace check with a CSV file
        with Timer("dm_network"):
            mapper2 = pip_ElM2D(sub_formulas)
            # mapper2 = ElM2D(emd_algorithm="network_simplex")
            mapper2.fit(sub_formulas)
            dm_network = mapper2.dm

        # with Timer("dm_elemd"):
        #     mod_petti = EleMD(scale="mod_pettifor")
        #     dm_elemd = np.zeros_like(dm_wasserstein)
        #     for i, comp1 in enumerate(sub_formulas):
        #         for j, comp2 in enumerate(sub_formulas):
        #             if j < i:
        #                 dm_elemd[i, j] = mod_petti.elemd(comp1, comp2)
        #                 dm_elemd[j, i] = dm_elemd[i, j]

        # print(
        #     "dm_wasserstein vs. dm_elemd RMSE: ",
        #     mean_squared_error(
        #         squareform(dm_wasserstein), squareform(dm_elemd), squared=False
        #     ),
        # )

        print(
            "dm_wasserstein vs. dm_network RMSE: ",
            mean_squared_error(
                squareform(dm_wasserstein), squareform(dm_network), squared=False
            ),
        )

        # print(
        #     "dm_network vs. dm_elemed RMSE: ",
        #     mean_squared_error(
        #         squareform(dm_network), squareform(dm_elemd), squared=False
        #     ),
        # )

        assert_allclose(
            dm_wasserstein,
            dm_network,
            atol=1e-3,
            err_msg="wasserstein did not match network simplex.",
        )

        # assert_allclose(
        #     dm_wasserstein,
        #     dm_elemd,
        #     atol=1e-3,
        #     err_msg="wasserstein did not match EleMD simplex.",
        # )

        # assert_allclose(
        #     dm_network,
        #     dm_elemd,
        #     atol=1e-3,
        #     err_msg="network did not match EleMD.",
        # )


if __name__ == "__main__":
    unittest.main()
# network_simplex gives a strange error (freeze_support() on Spyder IPython);
# however dist_matrix does not give this error

# %%
# Note: "network_simplex" emd_algorithm produces a "freeze_support()" error in
# Spyder IPython and Spyder external terminal without the if __name == "__main__" on
# Windows. This is likely due to the way Pool is used internally within ElM2D.

# n_elements = len(ElMD(metric="mod_petti").periodic_tab)

# settings = {
#     "INLINE": "never",
#     "FASTMATH": True,
#     "COLUMNS": n_elements,
#     "USE_64": False,
#     "TARGET": "cuda",
# }

# with open("dist_matrix_settings.json", "w") as f:
#     json.dump(settings, f)

# if cuda.is_available():
#     target = "cuda"
# else:
#     target = "cpu"
