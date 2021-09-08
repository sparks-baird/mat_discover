# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 16:14:14 2021.

@author: sterg
"""
from multiprocessing import freeze_support
from os.path import join
import pandas as pd
from ElM2D import ElM2D

def main():
    """
    Run ElM2D fitting which is compatible with Spyder for debugging purposes.

    Returns
    -------
    None.

    """
    mapper = ElM2D()
    #datapath = join("..","..","discovery","ael_bulk_modulus_voigt","train.csv")
    datapath = "train-debug.csv"
    df = pd.read_csv(datapath)
    mapper.fit(df["formula"])
    mapper.dm
    
if __name__ == '__main__':
    freeze_support()
    main()