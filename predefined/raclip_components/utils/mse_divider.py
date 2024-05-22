import numpy as np
from sklearn.metrics import mean_squared_error


def mse_divider(G:np.array, P:np.array, len_region:int=43):
    assert len(G) == len(P)
    G_div = np.reshape(G, (-1, len_region))  # [num_sample, 43]
    P_div = np.reshape(P, (-1, len_region))  # [num_sample, 43]
    mse_matrix = []
    for i in range(len_region):
        mse = mean_squared_error(G_div[:, i], P_div[:, i])
        mse_matrix.append(mse)
    assert len(mse_matrix) == len_region
    return mse_matrix
