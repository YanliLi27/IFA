import os
from generators.init_utils.dataset_scanner import ESMIRA_scanner
from generators.init_utils.input_filter import input_filter
from generators.init_utils.split_generator import class_generator, split_generator, split_definer, balancer, val_split_definer
from generators.init_utils.central_slice import central_slice_generator
from generators.init_utils.split_saver import split_saver
from dataset.datasets import ESMIRADataset2D
from torch.utils.data import Dataset
from typing import Union, Tuple
# from torchvision import transforms
import pickle
import pandas as pd


class ESMIRA_generator:
    def __init__(self, data_root:str, target_category:list=['EAC', 'ATL'], target_site:list=['Wrist'],
                 target_dirc:list=['TRA', 'COR'], maxfold:int=5) ->None:
        # The common ids: common_list is the dict of EAC/CSA/ATL, that patients exist in all target_site
        # {'EAC':[LIST], 'CSA':[LIST], 'ATL':[LIST]}
        self.data_root = data_root
        self.target_category = target_category
        self.default_id_path = f'./dataset/dicts/{data_root[-4:]}_id_list.pkl'
        if os.path.isfile(self.default_id_path):
            with open(self.default_id_path, "rb") as tf:
                self.common_dict = pickle.load(tf)
        else:
            self.common_dict = ESMIRA_scanner(self.data_root, target_category)  # {'EAC':[LIST], 'CSA':[LIST], 'ATL':[LIST]} [LIST] -- ['Csa842_CSA', ...]
            # EACNUMM	MRI_T1	MRIdate_T1	MRI_T2	MRIdate_T2	MRI_T4	MRIdate_T4	MRI_T6	MRIdate_T6	RA_baseline	RA_1yr	658	112	770
            # path: D:\ESMIRA\SPSS data\Copy of EAC RAdiags excel_changeInRA.xlsx
            # create an id list
            if 'EAC' in target_category:
                target_id_lsit = self._id_finder_eac(path='D:\\ESMIRA\\SPSS data\\Copy of EAC RAdiags excel_changeInRA.xlsx')
                self.common_dict['EAC'] = self._id_filter_eac(self.common_dict['EAC'], target_id_lsit)

            with open(self.default_id_path, "wb") as tf:
                pickle.dump(self.common_dict, tf)
            
        
        # calculate the central slices
        self.default_cs_path = f'./dataset/dicts/{data_root[-4:]}_name2central_list.pkl'
        if os.path.isfile(self.default_cs_path):
            with open(self.default_cs_path, "rb") as tf:
                self.common_cs_dict = pickle.load(tf)
            print('saved dataset+central slice dict:{}'.format(self.common_cs_dict.keys()))
            print('--------found saved info--------')
        else:
            self.common_cs_dict = central_slice_generator(self.data_root, self.common_dict, target_category)
            # From now, no more ids, but the specific name and path /此处开始不再是ids，而是具体的名称和路径
            # {'EAC_Wrist_TRA':[LIST-'Names_label.mha:10to15'], ..., 'CSA_MCP_COR':[LIST-'Names_label.mha:8to13'], ...}
            with open(self.default_cs_path, "wb") as tf:
                pickle.dump(self.common_cs_dict, tf)
            df = pd.DataFrame(self.common_cs_dict)
            df.to_csv(path_or_buf=self.default_cs_path.replace('.pkl', '.csv'))

        # --------------------------------------------------------main generator-------------------------------------------------------- #
        # input selection:
        # common_cs_dict  {'EAC_XXX_XXX':[LIST-'Names_label.mha:10to15'], ..., 'ATL_XXX_XXX':[LIST-'Names_label.mha:8to13']}
        self.common_cs_dict = input_filter(self.common_cs_dict, target_category, target_site, target_dirc)
        print('Remained keys: ', self.common_cs_dict.keys())
        # common_cs_dict  {'EAC_Wrist_TRA':[LIST-'Names_label.mha:10to15'], 'ATL_Wrist_TRA':[LIST-'Names_label.mha:8to13']}


        self.repr_target_split_path = split_saver(data_root[-4:], target_category, target_site, target_dirc, True)
        self.repr_atlas_split_path = split_saver(data_root[-4:], target_category, target_site, target_dirc, False)
        if os.path.isfile(self.repr_target_split_path) and os.path.isfile(self.repr_atlas_split_path):
            with open(self.repr_target_split_path, "rb") as tf:
                self.target_split = pickle.load(tf)
            with open(self.repr_atlas_split_path, "rb") as tf2:
                self.atlas_split = pickle.load(tf2)
            print('--------found saved split--------')
        # label generation and fold-split    
        else:
            self.target_split, self.atlas_split = class_generator(self.common_cs_dict, target_category)
            # target_split -- {'EAC_XXX_XXX':[LIST--subname+names.mha:10to15:label], ...}
            # atlas_split -- {'ATL_XXX_XXX':[LIST--subname+names.mha:10to15:label], ...}

            # in each sub-list 'Cate_XXX_XXX', split into 5 fold:
            self.target_split = split_generator(self.target_split, random=True, maxfold=maxfold)
            self.atlas_split = split_generator(self.atlas_split, random=True, maxfold=maxfold)
            # {'EAC_XXX_XXX':[5*[LIST--subname+names.mha:10to15:1]], 'EAC_XXX_XXX':[5*[LIST--subname+names.mha:10to15:1]], ...}
            # {'ATL_XXX_XXX':[5*[LIST--subname+names.mha:10to15:0]], 'ATL_XXX_XXX':[5*[LIST--subname+names.mha:10to15:0]], ...}
            with open(self.repr_target_split_path, "wb") as tf:
                pickle.dump(self.target_split, tf)
            with open(self.repr_atlas_split_path, "wb") as tf2:
                pickle.dump(self.atlas_split, tf2)
        print('-------------------------------> Dataset Initialization Finished <-------------------------------')

    def _id_finder_eac(path:str=''):
        target_id_list = []
        df = pd.read_excel(path)
        assert df.columns == ['EACNUMM', 'MRI_T1', 'MRIdate_T1', 'MRI_T2', 'MRIdate_T2', 'MRI_T4', 'MRIdate_T4',
                              	'MRI_T6', 'MRIdate_T6', 'RA_baseline', 'RA_1yr', '658', '112', '770']
        df1 = df.loc[df['770']==1]
        target_id_list = df1['EACNUMM'].values
        print(target_id_list.dtype)
        print(target_id_list)
        return target_id_list
    
    def _id_filter_eac(init_list:list, target_id_list:list):
        # 3393 --> 'Arth3393_EAC' in init_list
        return_list = []
        for item in init_list:
            item_short = item.replace('Arth', '')
            item_short = item_short.replace('_EAC', '')
            if item_short in target_id_list:
                return_list.append(item)
        print(f'length of target list: {len(target_id_list)}')
        print(f'length of init list: {len(init_list)}')
        print((f'length of return list: {len(return_list)}'))
        return return_list


    def returner(self, phase:str='train', fold_order:int=0, mean_std:bool=False, 
                 monai:bool=True, full_img:Union[bool, int]=5, path_flag:bool=False,
                 test_balance:bool=True, dimension:str='2D') ->Tuple[Union[None, Dataset], Dataset]:
        if monai:
            from monai import transforms
            transform = [transforms.Compose([
                                            # transforms.RandGaussianNoise(0.2, 0, 0.1),
                                            transforms.RandFlip(0.5, 0),
                                            transforms.RandRotate((30), prob=0.7),
                                            transforms.RandAffine(prob=1.0, translate_range=(40, 40)),
                                            # transforms.RandShiftIntensity(offsets=0.1, safe=True, prob=0.2),
                                            # transforms.RandStdShiftIntensity(factors=0.1, prob=0.2),
                                            # transforms.RandBiasField(degree=2, coeff_range=(0, 0.1), prob=0.2),
                                            # transforms.RandAdjustContrast(prob=0.5, gamma=(0.9, 1.1)),
                                            transforms.RandHistogramShift(num_control_points=10, prob=0.2),
                                            transforms.RandZoom(prob=0.7, min_zoom=0.8, max_zoom=1.1, keep_size=True)
                                            ]),
                        None
                            ]           
        else:
            from torchvision import transforms
            transform = [transforms.Compose([transforms.RandomHorizontalFlip(0.5),
                                            transforms.RandomRotation((10),),
                                            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(1, 1), shear=None, fill=0),
                                        ]),  
                       None
                       ]

        if phase=='train':
            target_train_dict, target_val_dict = split_definer(self.target_split, fold_order) 
            # {'EAC_XXX_XXX':[4*LIST--subname+names.mha:10to15:1], 'EAC_XXX_XXX':[4*LIST--subname+names.mha:10to15:1], ...}
            # e.g: 'EAC_XXX_XXX':[EAC_XXX_XXX+names.mha:10to15:1, ...]
            # {'EAC_XXX_XXX':[LIST--subname+names.mha:10to15:1], 'EAC_XXX_XXX':[LIST--subname+names.mha:10to15:1], ...}
            atlas_train_dict, atlas_val_dict = split_definer(self.atlas_split, fold_order)
            # {'ATL_XXX_XXX':[4*LIST--subname+names.mha:10to15:0], 'ATL_XXX_XXX':[4*LIST--subname+names.mha:10to15:0], ...}
            # {'ATL_XXX_XXX':[LIST--subname+names.mha:10to15:0], 'ATL_XXX_XXX':[LIST--subname+names.mha:10to15:0], ...}
            
            train_dict = balancer(target_train_dict, atlas_train_dict, self.target_category)
            val_dict = balancer(target_val_dict, atlas_val_dict, self.target_category, balance=test_balance)
            # {'site_dirc':[LIST(Target+Atlas): subdir\names.mha:cs:label ], ...}
            
            train_dataset = ESMIRADataset2D(self.data_root,train_dict, transform[0], mean_std, 
                                            full_img=full_img, path_flag=path_flag, dimension=dimension)
            val_dataset = ESMIRADataset2D(self.data_root, val_dict, transform[1], mean_std, 
                                          full_img=full_img, path_flag=path_flag, dimension=dimension)
        else:
            target_val_dict = val_split_definer(self.target_split)
            atlas_val_dict = val_split_definer(self.atlas_split)
            val_dict = balancer(target_val_dict, atlas_val_dict, self.target_category, balance=test_balance)
            val_dataset = ESMIRADataset2D(self.data_root, val_dict, transform[1], mean_std, 
                                          full_img=full_img, path_flag=path_flag, dimension=dimension)
            train_dataset = None
        return train_dataset, val_dataset  # [N*5, 512, 512] + int(label)