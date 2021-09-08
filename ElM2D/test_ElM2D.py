# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 22:02:36 2021

@author: sterg
"""
from ElM2D import ElM2D
from helper import Timer
from os.path import join
import pandas as pd

mapper = ElM2D()

datapath = join("ael_bulk_modulus_voigt", "train.csv")
valpath = join("ael_bulk_modulus_voigt", "val.csv")
df = pd.read_csv(datapath)
val_df = pd.read_csv(valpath)
tmp_df = pd.concat([df[0:4], val_df[0:3]])

sub_formulas = df["formula"][0:100]

formulas = df["formula"]
with Timer("fit-wasserstein"):
    mapper.fit(formulas)
    dm_wasserstein = mapper.dm

# network_simplex gives an error (freeze_support() on Spyder IPython)


# %% CODE GRAVEYARD
# from numpy.testing.utils import assert_allclose

# mapper2 = ElM2D(emd="network_simplex")

# with Timer("fit-network_simplex"):
#     mapper2.fit(sub_formulas)
#     dm_network = mapper2.dm

# assert_allclose(
#     dm_wasserstein, dm_network, err_msg="wasserstein did not match network simplex."
# )
