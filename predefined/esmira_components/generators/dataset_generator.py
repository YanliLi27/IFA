import os
from predefined.esmira_components.generators.init_utils.dataset_scanner import ESMIRA_scanner
from predefined.esmira_components.generators.init_utils.input_filter import input_filter
from predefined.esmira_components.generators.init_utils.split_generator import class_generator, split_generator, split_definer, balancer, val_split_definer
from predefined.esmira_components.generators.init_utils.central_slice import central_slice_generator
from dataset.datasets import ESMIRADataset2D
from torch.utils.data import Dataset
from typing import Union, Tuple
from torchvision import transforms
import pickle


def ESMIRA_generator(data_root:str, target_category:list=['EAC', 'ATL'], target_site:list=['Wrist'],
                     target_dirc:list=['TRA', 'COR'],
                     phase:str='train', fold_order:int=0, mean_std:bool=False) ->Tuple[Union[None, Dataset], Dataset]:
    # 1. Decide the dirname with phase: train/test -- notice: all the seperate test set were already moved to a individual dirctory
    # 2. Scan the dirname and get the masked patient id
    # 3. 


    # The common ids: common_list is the dict of EAC/CSA/ATL, that patients exist in all target_site
    # {'EAC':[LIST], 'CSA':[LIST], 'ATL':[LIST]}
    default_id_path = './predefined/esmira_components/dataset/dicts/id_list.pkl'
    if os.path.isfile(default_id_path):
        with open(default_id_path, "rb") as tf:
            common_dict = pickle.load(tf)
    else:
        common_dict = ESMIRA_scanner(data_root)
        with open(default_id_path, "wb") as tf:
            pickle.dump(common_dict, tf)
        
    
    # calculate the central slices
    default_cs_path = './predefined/esmira_components/dataset/dicts/name2central_list.pkl'
    if os.path.isfile(default_cs_path):
        with open(default_cs_path, "rb") as tf:
            common_cs_dict = pickle.load(tf)
            print('saved dataset+central slice dict:{}'.format(common_cs_dict.keys()))
            print('--------found saved info--------')
    else:
        common_cs_dict = central_slice_generator(data_root, common_dict)
        # From now, no more ids, but the specific name and path /此处开始不再是ids，而是具体的名称和路径
        # {'EAC_Wrist_TRA':[LIST-'Names_label.mha:10to15'], ..., 'CSA_MCP_COR':[LIST-'Names_label.mha:8to13'], ...}
        with open(default_cs_path, "wb") as tf:
            pickle.dump(common_cs_dict, tf)

    # --------------------------------------------------------main generator-------------------------------------------------------- #
    # input selection:
    # common_cs_dict  {'EAC_XXX_XXX':[LIST-'Names_label.mha:10to15'], ..., 'ATL_XXX_XXX':[LIST-'Names_label.mha:8to13']}
    common_cs_dict = input_filter(common_cs_dict, target_site, target_dirc)
    print('Remained keys: ', common_cs_dict.keys())
    # common_cs_dict  {'EAC_Wrist_TRA':[LIST-'Names_label.mha:10to15'], 'ATL_Wrist_TRA':[LIST-'Names_label.mha:8to13']}

    # label generation and fold-split    
    if phase=='train':
        # define labels in the list
        target_split, atlas_split = class_generator(common_cs_dict, target_category)
        # target_split -- {'EAC_XXX_XXX':[LIST--subname+names.mha:10to15:label], ...}
        # atlas_split -- {'ATL_XXX_XXX':[LIST--subname+names.mha:10to15:label], ...}

        # in each sub-list 'Cate_XXX_XXX', split into 5 fold:
        target_split = split_generator(target_split)
        atlas_split = split_generator(atlas_split)
        # {'EAC_XXX_XXX':[5*[LIST--subname+names.mha:10to15:1]], 'EAC_XXX_XXX':[5*[LIST--subname+names.mha:10to15:1]], ...}
        # {'ATL_XXX_XXX':[5*[LIST--subname+names.mha:10to15:0]], 'ATL_XXX_XXX':[5*[LIST--subname+names.mha:10to15:0]], ...}

        target_train_dict, target_val_dict = split_definer(target_split, fold_order) 
        # {'EAC_XXX_XXX':[4*LIST--subname+names.mha:10to15:1], 'EAC_XXX_XXX':[4*LIST--subname+names.mha:10to15:1], ...}
        # e.g: 'EAC_XXX_XXX':[EAC_XXX_XXX+names.mha:10to15:1, ...]
        # {'EAC_XXX_XXX':[LIST--subname+names.mha:10to15:1], 'EAC_XXX_XXX':[LIST--subname+names.mha:10to15:1], ...}
        atlas_train_dict, atlas_val_dict = split_definer(atlas_split, fold_order)
        # {'ATL_XXX_XXX':[4*LIST--subname+names.mha:10to15:0], 'ATL_XXX_XXX':[4*LIST--subname+names.mha:10to15:0], ...}
        # {'ATL_XXX_XXX':[LIST--subname+names.mha:10to15:0], 'ATL_XXX_XXX':[LIST--subname+names.mha:10to15:0], ...}
        train_dict = balancer(target_train_dict, atlas_train_dict, target_category)
        val_dict = balancer(target_val_dict, atlas_val_dict, target_category)
        # {'site_dirc':[LIST(Target+Atlas): subdir\names.mha:cs:label ], ...}
        
        transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.RandomHorizontalFlip(0.5),
                                        transforms.RandomRotation((10),),
                                        transforms.RandomAffine(0, translate=(0.05, 0), scale=(1, 1), shear=None, fill=0),
                                    ]) 
        val_transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.RandomAffine(0, translate=(0.05, 0), scale=(1, 1), shear=None, fill=0),
                                    ]) 
        train_dataset = ESMIRADataset2D(data_root,train_dict, transform, mean_std)
        val_dataset = ESMIRADataset2D(data_root, val_dict, val_transform, mean_std)
    else:
        target_split, atlas_split = class_generator(common_cs_dict, target_category)
        target_split = split_generator(target_split)
        atlas_split = split_generator(atlas_split)
        target_val_dict = val_split_definer(target_split)
        atlas_val_dict = val_split_definer(atlas_split)
        val_dict = balancer(target_val_dict, atlas_val_dict, target_category)
        val_transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.RandomAffine(0, translate=(0.05, 0), scale=(1, 1), shear=None, fill=0),
                                    ]) 
        val_dataset = ESMIRADataset2D(data_root, val_dict, val_transform, mean_std)
        train_dataset = None
    return train_dataset, val_dataset  # [N*5, 512, 512] + int(label)