from predefined.synaug_components.generators.synaug_list import createcombination
from predefined.synaug_components.generators.dataset.synaugdataset import SynaugReg
from sklearn.model_selection import KFold
from typing import Tuple
import numpy as np
import pickle
import os


class Synaug_generator:
    def __init__(self, maxfold:int=5, score_sum:bool=False) -> None:
        savepath = r'./predefined/synaug_components/generators/save.pkl'
        if os.path.exists(savepath):
            with open(savepath, "rb") as tf:
                self.datalist = pickle.load(tf)
        else:
            self.datalist = createcombination()  # [N* [origin, aug]]
            self.datalist = self._split_generator(self.datalist, shuffle=True, maxfold=maxfold)
            with open(savepath, "wb") as tf:
                pickle.dump(self.datalist, tf)
        self.maxfold = maxfold

    def _split_generator(self, list_ids:list, shuffle:bool=True, maxfold:int=5) ->list:
        # input: dict {'id':label -length==5N}
        # return: list [list[id, id,...], list[id, id,...], list[id, id,...], ...]
        list_ids  # [id, id, id]
        kf = KFold(n_splits=maxfold, shuffle=shuffle)
        target_split_list = []
        for train_index, val_index in kf.split(range(len(list_ids))):
            # val_index -- the ids in val set
            split_sublist = []
            split_list = list(np.array(list_ids)[val_index])  # [ids, ...]
            for id in split_list:
                split_sublist.append(id)
            target_split_list.append(split_sublist)
        return target_split_list
    

    def _split_definer(self, split_list:list, fold_order:int) ->Tuple[list, list]:
        # list [[id[], id[], ...] *5] 5 fold
        # -> train list [id[], id[], id[], ...], val list [id[], id[], id[], ...]
        # in aimira:
        # list [[id[tp1[...],tp2[...],..], id[...]] *5]  fold
        # -> train list [id[tp1,tp2,...], id[], id[], ...], val list [id[tp1,tp2,...], id[], id[], ...]
        train_split_dict = []
        val_split_dict = []
        for i in range(len(split_list)):
            if i == fold_order:
                val_split_dict.extend(split_list[i])
            elif i != fold_order:
                train_split_dict.extend(split_list[i])  # [LIST*4]
        return train_split_dict, val_split_dict
    
    def _val_split_definer(self, split_list:list) ->list:
        # list [[id[], id[], ...] *5] 5 fold
        # -> train list [id[], id[], id[], ...], val list [id[], id[], id[], ...]
        val_split_dict = []
        for i in range(len(split_list)):
            val_split_dict.extend(split_list[i])
        return val_split_dict
    

    def returner(self, train_flag:bool=False, fold_order:int=0, path_flag:bool=True):
        if train_flag:
            traindata, valdata = self._split_definer(self.datalist, fold_order)
            traindataset = SynaugReg(traindata, None, False, path_flag=path_flag, cleanup=False)
            valdataset = SynaugReg(valdata, None, False, path_flag=path_flag, cleanup=False)
        else:
            valdata = self._val_split_definer(self.datalist)
            valdataset = SynaugReg(valdata, None, False, path_flag=path_flag, cleanup=False)
            traindataset = None
        return traindataset, valdataset
        
