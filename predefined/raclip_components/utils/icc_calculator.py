from sklearn.metrics import confusion_matrix
import os
import numpy as np
from matplotlib import pyplot as plt
import pingouin as pg
import pandas as pd


def icc_calculator(G:np.array, P:np.array) -> float:
    if len(G.shape)==3:
        G = np.sum(G, axis=(1,2))
        P = np.sum(P, axis=(1,2))
    elif len(G.shape)==4:
        G = np.sum(G, axis=(1,2,3))
        P = np.sum(P, axis=(1,2,3))
    elif len(G.shape)==2:
        G = np.sum(G, axis=(1))
        P = np.sum(P, axis=(1))
    index = list(range(len(G)))
    rater = [1] * len(G)
    rater2 = [0] * len(P)
    assert len(rater)==len(rater2)
    G_dict = {'ID':index, 'Score':G, 'rater':rater}
    P_dict = {'ID':index, 'Score':P, 'rater':rater2}
    Gdf = pd.DataFrame(G_dict)
    Pdf = pd.DataFrame(P_dict)

    data = pd.concat([Gdf, Pdf], axis=0)

    icc = pg.intraclass_corr(data=data, targets='ID', raters='rater', ratings='Score').round(8)# 
    icc = icc.set_index("Type")
    icc = icc.loc['ICC2']['ICC']
    return icc
    