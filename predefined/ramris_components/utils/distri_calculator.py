import numpy as np
from typing import Union, Tuple
from matplotlib import pyplot as plt
from scipy import stats
import statsmodels.api as sm


def _sort_distri(Garray:np.array, save_name:str='./models/figs/example.jpg', save:bool=False):
    Garray = np.sort(Garray)
    plt.clf()
    plt.hist(Garray, bins=range(11))
    plt.title('histogram')
    if save:
        plt.savefig(save_name)
    else:
        plt.show()
    plt.clf()

def distri_calculators(Garray:np.array, num_scores_per_site:int=43, division:bool=False, div_target:Union[int, str]=0,
                      save_name:str='./models/figs/example.jpg'):
    # division: the length of scores per sample
    # Garray -- array [batch * num_scores_per_site]
    if len(Garray.shape) >= 2:
        if Garray.shape[1] == 1:  # only when the input array is flatten
            Garray = Garray.reshape((-1, num_scores_per_site))  # from [batch * num_scores_per_site] to [batch, num_scores_per_site]
    else:
        Garray = Garray.reshape((-1, num_scores_per_site))  # from [batch * num_scores_per_site] to [batch, num_scores_per_site]
    if division:
        if type(div_target)==int:
            _sort_distri(Garray[:, div_target], save_name, save=True)  # from  [batch, num_scores_per_site] to [batch, (1)]
        elif type(div_target)==str:
            for i in range(num_scores_per_site):
                _sort_distri(Garray[: i], save_name, save=True)
        else:
            raise ValueError('not supported div target')
    else:
        _sort_distri(np.sum(Garray, axis=1), save_name, save=True)  # from  [batch, num_scores_per_site] to [batch, (1)])



def _corr_sort_distri(Garray:np.array, Parray:np.array, mode:str='corr', save_path:str='./models/figs/ba_pred_gt.jpg'):
    # Parray 是mse时，则会给出MSE与Score大小的correlation关系
    # Parray 是prediction时， 会给出Pred和GT的correlation关系
    # correlation 是否需要排序？
    # Index = np.argsort(Garray)
    # Garray = np.sort(Garray)
    # Psorted = []
    # for index in Index:
    #     Psorted.append(Parray[index])
    assert mode in ['corr', 'blandaltman']
    if mode == 'corr':
        corr, p_value = stats.pearsonr(Garray, Parray)       
        return corr, p_value
    elif mode == 'blandaltman':
        plt.clf()
        f, ax = plt.subplots(1, figsize = (8,5))
        plt.title('Bland altman of G and P',fontsize='large', fontweight='bold')
        sm.graphics.mean_diff_plot(Garray, Parray, ax = ax)
        # plt.show()
        plt.savefig(save_path)
        plt.clf()
        print('bland altman saved')
        return 0, 0


def corr_calculator(Garray:np.array, Parray:np.array, num_scores_per_site:int=43,
                    division:Union[None, int]=None, div_target:Union[int, str]=0,
                    mode:str='corr', save_path:str='./models/figs/ba_pred_gt.jpg'):
    # G -- the reference, P -- the MSE or Pred.
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
    if division:
        if type(div_target)==int:
            corr, p_value = _corr_sort_distri(Garray[:, div_target], Parray[:, div_target], mode, save_path)
        elif type(div_target)==str:
            corr, p_value = 0, 0
            for i in range(num_scores_per_site):
                corr_i, p_value_i = _corr_sort_distri(Garray[:, i], Parray[:, i], mode, save_path)
                corr += corr_i
                p_value += p_value_i
            corr /= num_scores_per_site
            p_value /= num_scores_per_site
        else:
            raise ValueError('not supported div target')
    else:
        corr, p_value = _corr_sort_distri(np.sum(Garray, axis=1), np.sum(Parray, axis=1), mode, save_path)
    return corr, p_value


def _scatter_plot(Garray:np.array, Parray:np.array, mode:str='save', save_path:str='./models/figs/scatter_gt_pr.jpg'):
    assert mode in ['save', 'print']
    plt.clf()
    # set the fit curve
    linear_model=np.polyfit(Garray, Parray, 1)
    linear_model_fn=np.poly1d(linear_model)
    x_s=np.arange(0,16)
    plt.plot(x_s,linear_model_fn(x_s),color="red")
    # set the reference curve
    plt.plot([0, 15], [0, 15],color="green")
    # scatter plot
    plt.scatter(Garray, Parray, s=5, c=None, marker=None, cmap=None, norm=None, alpha=0.5, linewidths=None)
    plt.savefig(save_path)
    plt.clf()


def sca_calculator(Garray:np.array, Parray:np.array, num_scores_per_site:int=43,
                    division:Union[None, int]=None, div_target:Union[int, str]=0,
                    mode:str='save', save_path:str='./models/figs/scatter_gt_pr.jpg'):
    # G -- the reference, P -- the MSE or Pred.
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
    if division:
        if type(div_target)==int:
            _scatter_plot(Garray[:, div_target], Parray[:, div_target], mode)
        elif type(div_target)==str:
            for i in range(num_scores_per_site):
                _scatter_plot(Garray[:, i], Parray[:, i], mode)
        else:
            raise ValueError('not supported div target')
    else:
        _scatter_plot(np.sum(Garray, axis=1), np.sum(Parray, axis=1), mode, save_path)