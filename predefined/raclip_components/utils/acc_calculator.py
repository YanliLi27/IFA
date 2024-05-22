import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, roc_curve, r2_score, mean_squared_error
from typing import Union, Tuple
from scipy import stats


def acc_calculator(G:np.array, P:np.array, num_scores_per_site:int=43):
    P = np.asarray(np.round(2*P), dtype=int)
    G = np.asarray(2*G, dtype=int)
    divided_acc = accuracy_score(G, P)

    if len(G.shape) >= 2:
        if G.shape[1] == 1:  # only when the input array is flatten
            G = G.reshape((-1, num_scores_per_site))  # from [batch * num_scores_per_site] to [batch, num_scores_per_site]
    else:
        G = G.reshape((-1, num_scores_per_site))  # from [batch * num_scores_per_site] to [batch, num_scores_per_site]
    if len(P.shape) >= 2:
        if P.shape[1] == 1:  # only when the input array is flatten
            P = P.reshape((-1, num_scores_per_site))  # from [batch * num_scores_per_site] to [batch, num_scores_per_site]
    else:
        P = P.reshape((-1, num_scores_per_site))  # from [batch * num_scores_per_site] to [batch, num_scores_per_site]
    sum_acc = []
    for i in range(G.shape[1]):
        sum_acc.append(accuracy_score(G[:,i], P[:,i]))
    return divided_acc, sum_acc


def regularized_corr_calculator(G:np.array, P:np.array, num_scores_per_site:int=43):
    Parray = np.maximum(np.asarray(np.round(2*P), dtype=int),0)
    Garray= np.maximum(np.asarray(2*G, dtype=int), 0)
    if len(Garray.shape) >= 2:
        if Garray.shape[1] == 1:  # only when the input array is flatten
            Garray = Garray.reshape((-1, num_scores_per_site))  # from [batch * num_scores_per_site] to [batch, num_scores_per_site]
    else:
        Garray = Garray.reshape((-1, num_scores_per_site))  # from [batch * num_scores_per_site] to [batch, num_scores_per_site]
    if len(Parray.shape) >= 2:
        if Parray.shape[1] == 1:
            Parray = Parray.reshape((-1, num_scores_per_site))
    else:
        Parray = Parray.reshape((-1, num_scores_per_site))  # from [batch * num_scores_per_site] to [batch, num_scores_per_site]
    
    # normality make sure:
    summed_G = np.sum(Garray, axis=1)
    _, Pvalue =  stats.kstest(summed_G, 'norm', (np.mean(summed_G), np.std(summed_G)))
    if Pvalue>0.05:
        print('normaliy distributed')
        corr, p_value = stats.pearsonr(np.sum(Garray, axis=1), np.sum(Parray, axis=1))
    else:
        wrong_corr, _ = stats.spearmanr(np.sum(Garray, axis=1), np.sum(Parray, axis=1))
        # print(f'not normaliy distributed, if use spearman: {wrong_corr}')
        corr, p_value = stats.pearsonr(np.sum(Garray, axis=1), np.sum(Parray, axis=1)) 
    return corr, p_value