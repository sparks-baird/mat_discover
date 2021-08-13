# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 16:14:14 2021

@author: sterg
"""
from multiprocessing import freeze_support
from os.path import join
import pandas as pd
from ElM2D import ElM2D

def main():
    mapper = ElM2D()
    datapath = join("train-debug.csv")
    df = pd.read_csv(datapath)
    mapper.fit(df["formula"])
    
if __name__ == '__main__':
    freeze_support()
    main()