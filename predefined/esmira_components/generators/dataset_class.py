import os
from predefined.esmira_components.generators.init_utils.dataset_scanner import ESMIRA_scanner
from predefined.esmira_components.generators.init_utils.input_filter import input_filter
from predefined.esmira_components.generators.init_utils.split_generator import class_generator, split_generator, split_definer, balancer, val_split_definer
from predefined.esmira_components.generators.init_utils.central_slice import central_slice_generator
from predefined.esmira_components.generators.init_utils.split_saver import split_saver
from predefined.esmira_components.dataset.datasets import ESMIRADataset2D
from torch.utils.data import Dataset
from typing import Union, Tuple
from torchvision import transforms
import pickle


class ESMIRA_generator:
    def __init__(self, data_root:str, target_category:list=['EAC', 'ATL'], target_site:list=['Wrist'],
                 target_dirc:list=['TRA', 'COR']) ->None:
        # The common ids: common_list is the dict of EAC/CSA/ATL, that patients exist in all target_site
        # {'EAC':[LIST], 'CSA':[LIST], 'ATL':[LIST]}
        self.data_root = data_root
        self.target_category = target_category
        self.default_id_path = './predefined/esmira_components/dataset/dicts/id_list.pkl'
        if os.path.isfile(self.default_id_path):
            with open(self.default_id_path, "rb") as tf:
                self.common_dict = pickle.load(tf)
        else:
            self.common_dict = ESMIRA_scanner(self.data_root)
            with open(self.default_id_path, "wb") as tf:
                pickle.dump(self.common_dict, tf)
            
        
        # calculate the central slices
        self.default_cs_path = './predefined/esmira_components/dataset/dicts/name2central_list.pkl'
        if os.path.isfile(self.default_cs_path):
            with open(self.default_cs_path, "rb") as tf:
                self.common_cs_dict = pickle.load(tf)
            print('saved dataset+central slice dict:{}'.format(self.common_cs_dict.keys()))
            print('--------found saved info--------')
        else:
            self.common_cs_dict = central_slice_generator(self.data_root, self.common_dict)
            # From now, no more ids, but the specific name and path /此处开始不再是ids，而是具体的名称和路径
            # {'EAC_Wrist_TRA':[LIST-'Names_label.mha:10to15'], ..., 'CSA_MCP_COR':[LIST-'Names_label.mha:8to13'], ...}
            with open(self.default_cs_path, "wb") as tf:
                pickle.dump(self.common_cs_dict, tf)

        # --------------------------------------------------------main generator-------------------------------------------------------- #
        # input selection:
        # common_cs_dict  {'EAC_XXX_XXX':[LIST-'Names_label.mha:10to15'], ..., 'ATL_XXX_XXX':[LIST-'Names_label.mha:8to13']}
        self.common_cs_dict = input_filter(self.common_cs_dict, target_category, target_site, target_dirc)
        print('Remained keys: ', self.common_cs_dict.keys())
        # common_cs_dict  {'EAC_Wrist_TRA':[LIST-'Names_label.mha:10to15'], 'ATL_Wrist_TRA':[LIST-'Names_label.mha:8to13']}

        self.repr_target_split_path = split_saver(target_category, target_site, target_dirc, True)
        self.repr_atlas_split_path = split_saver(target_category, target_site, target_dirc, False)
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
            self.target_split = split_generator(self.target_split)
            self.atlas_split = split_generator(self.atlas_split)
            # {'EAC_XXX_XXX':[5*[LIST--subname+names.mha:10to15:1]], 'EAC_XXX_XXX':[5*[LIST--subname+names.mha:10to15:1]], ...}
            # {'ATL_XXX_XXX':[5*[LIST--subname+names.mha:10to15:0]], 'ATL_XXX_XXX':[5*[LIST--subname+names.mha:10to15:0]], ...}
            with open(self.repr_target_split_path, "wb") as tf:
                pickle.dump(self.target_split, tf)
            with open(self.repr_atlas_split_path, "wb") as tf2:
                pickle.dump(self.atlas_split, tf2)
        print('-------------------------------> Dataset Initialization Finished <-------------------------------')


    def returner(self, phase:str='train', fold_order:int=0, mean_std:bool=False) ->Tuple[Union[None, Dataset], Dataset]:
        if phase=='train':
            target_train_dict, target_val_dict = split_definer(self.target_split, fold_order) 
            # {'EAC_XXX_XXX':[4*LIST--subname+names.mha:10to15:1], 'EAC_XXX_XXX':[4*LIST--subname+names.mha:10to15:1], ...}
            # e.g: 'EAC_XXX_XXX':[EAC_XXX_XXX+names.mha:10to15:1, ...]
            # {'EAC_XXX_XXX':[LIST--subname+names.mha:10to15:1], 'EAC_XXX_XXX':[LIST--subname+names.mha:10to15:1], ...}
            atlas_train_dict, atlas_val_dict = split_definer(self.atlas_split, fold_order)
            # {'ATL_XXX_XXX':[4*LIST--subname+names.mha:10to15:0], 'ATL_XXX_XXX':[4*LIST--subname+names.mha:10to15:0], ...}
            # {'ATL_XXX_XXX':[LIST--subname+names.mha:10to15:0], 'ATL_XXX_XXX':[LIST--subname+names.mha:10to15:0], ...}
            
            train_dict = balancer(target_train_dict, atlas_train_dict, self.target_category)
            val_dict = balancer(target_val_dict, atlas_val_dict, self.target_category)
            # {'site_dirc':[LIST(Target+Atlas): subdir\names.mha:cs:label ], ...}
            
            transform = transforms.Compose([
                                            transforms.RandomHorizontalFlip(0.5),
                                            transforms.RandomRotation((10),),
                                            transforms.RandomAffine(0, translate=(0.05, 0), scale=(1, 1), shear=None, fill=0),
                                        ]) 
            val_transform = transforms.Compose([
                                            transforms.RandomAffine(0, translate=(0.05, 0), scale=(1, 1), shear=None, fill=0),
                                        ])
            train_dataset = ESMIRADataset2D(self.data_root,train_dict, transform, mean_std)
            val_dataset = ESMIRADataset2D(self.data_root, val_dict, val_transform, mean_std)
        else:
            target_val_dict = val_split_definer(self.target_split)
            atlas_val_dict = val_split_definer(self.atlas_split)
            val_dict = balancer(target_val_dict, atlas_val_dict, self.target_category)
            val_transform = transforms.Compose([
                                            transforms.RandomAffine(0, translate=(0.05, 0), scale=(1, 1), shear=None, fill=0),
                                        ]) 
            val_dataset = ESMIRADataset2D(self.data_root, val_dict, val_transform, mean_std)
            train_dataset = None
        return train_dataset, val_dataset  # [N*5, 512, 512] + int(label)