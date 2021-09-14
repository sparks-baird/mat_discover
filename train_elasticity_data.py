"""
Train CrabNet using elasticity data from Materials Project.

Created on Sat Sep 11 21:12:54 2021

@author: sterg
"""
import os

# HACK
os.chdir("CrabNet")
from CrabNet.train_crabnet import main  # noqa

if __name__ == "__main__":
    main(mat_prop="elasticity")
    os.chdir("..")
