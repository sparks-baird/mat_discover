"""
Test Element Mover's 2D Distance Matrix via network simplex and "wasserstein" methods.

This test ensures that the fast implementation "wasserstein" produces "close" values
to the original network simplex method.

Note: "network_simplex" emd_algorithm produces a "freeze_support()" error in
Spyder IPython and Spyder external terminal without the if __name == "__main__" on
Windows. This is likely due to the way Pool is used internally within ElM2D.

Created on Mon Sep  6 22:02:36 2021

@author: sterg
"""
import unittest
from numpy.testing import assert_allclose
from helper import Timer
import pandas as pd


class Testing(unittest.TestCase):
    def test_dm_close(self):
        from ElM2D import ElM2D

        mapper = ElM2D()

        # df = pd.read_csv("train-debug.csv")
        df = pd.read_csv("stable-mp.csv")
        formulas = df["formula"]
        sub_formulas = formulas[0:5000]
        with Timer("fit-wasserstein"):
            mapper.fit(sub_formulas)
            dm_wasserstein = mapper.dm

        mapper2 = ElM2D(emd_algorithm="network_simplex")

        with Timer("fit-network_simplex"):
            mapper2.fit(sub_formulas)
            dm_network = mapper2.dm
        assert_allclose(
            dm_wasserstein,
            dm_network,
            atol=1e-3,
            err_msg="wasserstein did not match network simplex.",
        )


if __name__ == "__main__":
    unittest.main()
# network_simplex gives a strange error (freeze_support() on Spyder IPython);
# however dist_matrix does not give this error
