import pandas as pd
import os
from typing import Any, Union, Literal


class Record():
    def __init__(self, *Args:Union[str, list[str], tuple[str], None]) -> None:
        self.record = {}
        self.record['index'] = []
        if Args is not None:
            for arg in Args:
                self.record[arg] = []
        # create the save space
        self.keys = self.record.keys()

        # index_type:Literal['epoch', 'item']='epoch'
        # self.index_type = index_type  # to control the output type -- item csv [index, pred, y, path, extra]
        # -- epoch csv [epoch, trainloss, valloss, metric1, metric2, confusion matrix, ...]


    def __call__(self, **kwds) -> None:  # metrix1=xxxx  
        # while having new keys
        for key, _ in kwds.items():
            if key in self.keys:
                continue
            else:
                self.record[str(key)] = []
                print(f'create save for {key}')

        updated_keys = ['index']
        if self.record['index'] == []:
                    self.record['index'].append(0)
        else:
            self.record['index'].append(self.record['index'][-1]+1)

        for key, value in kwds.items():
            self.record[str(key)].append(value)
            updated_keys.append(str(key))

        left_keys = [item for item in self.keys if item not in updated_keys]
        for key in left_keys:
            self.record[str(key)].append('noitem')

        # update the keys, if new key added
        self.keys = self.record.keys()


    def summary(self, save_path:Union[str, None]=None, sort_key:Union[str, None]=None):
        df = pd.DataFrame(self.record)
        df.set_index(keys='index', drop=False)
        if save_path:
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            df.to_csv(save_path)
        else:
            print(df)

        if sort_key:
            # print('if use losses as the sort index, use - key')
            if '-' in sort_key:
                sort_key = sort_key.replace('-', '')
                max_index = self.record[sort_key].index(min(self.record[sort_key]))
            else:
                max_index = self.record[sort_key].index(max(self.record[sort_key]))
            for key in self.keys:
                print(f'{key}: {df.at[max_index, key]}')


if __name__ == '__main__':
    record = Record('trainloss', 'valloss', 'dice', 'confusion_matrix')
    record(trainloss=0.86, valloss=0.94, names='beta')
    print(record.record)
    record(trainloss=0.66, valloss=0.74, names='gama')
    print(record.record)
    record.summary(sort_key='-valloss')