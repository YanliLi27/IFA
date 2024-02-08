from sklearn.model_selection import KFold
from typing import Tuple
import numpy as np
import os


def class_generator(common_dict:dict, target_category:list) ->dict:
    # please get the target_category with the order of [EAC, CSA, ATL]
    # 此处classification和prediction有所不同，一个是按照名称划分。一个是按照key划分，因此根据target_category应构成两个不同的function
    # TODO 要不干脆直接在这个阶段把标签添加到名称里面，变成：Subname+Names.mha:10to15:1 -- 即‘路径：切片：标签’
    # 并且由于此处已经过了保存pickle的部分，因此保存的pickle不会随之变化
    # target_split -- {'EAC_Wrist_TRA':[5*[LIST--subname+names.mha:10to15:label]], 'EAC_MCP_TRA':...}
    # atlas_split -- {'EAC_Wrist_TRA':[5*[LIST--subname+names.mha:10to15:label]], ...}
    target_split = {}
    atlas_split = {}
    if len(target_category) == 1:  # Prediction task:  same subdir(cate), different sites and dirc
        for key in common_dict.keys():
            target_split[key] = []
            atlas_split[key] = []
            filenames = common_dict[key]  # [LIST--subname+names.mha:10to15]
            for filename in filenames:
                if '_1.mha' in filename:  # filename: str
                    filename = key + '\\' + filename + ':1'
                    target_split[key].append(filename)
                else:  # _0.mha and _8.mha
                    filename = key + '\\' + filename + ':0'
                    atlas_split[key].append(filename)

    elif len(target_category) == 2:
        target_split = {}
        atlas_split = {}
        for key in common_dict.keys():
            if target_category[0] in key:
                target_split[key] = []
                filenames = common_dict[key]
                for filename in filenames:
                    filename = key + '\\' + filename + ':1'
                    target_split[key].append(filename)
            elif target_category[1] in key:
                atlas_split[key] = []
                filenames = common_dict[key]
                for filename in filenames:
                    filename = key + '\\' + filename + ':0'
                    atlas_split[key].append(filename)
            else:
                raise ValueError('something wrong with key and category')
    else:
        raise ValueError('More than two classes required')
    return target_split, atlas_split
    # target_split = {'EAC_XXX_XXX':[LIST--subname+names.mha:10to15:1], ...}
    # atlas_split = {'ATL_XXX_XXX':[LIST--subname+names.mha:10to15:0], ...}


def split_generator(target_split:dict) ->dict:
    # target_split = {'EAC_XXX_XXX':[LIST--subname+names.mha:10to15:1], 'EAC_XXX_XXX':[LIST--subname+names.mha:10to15:1], ...}
    for key in target_split.keys():
        indiv_split = []
        list_key = target_split[key]  # [LIST]
        kf = KFold(n_splits=5, shuffle=True)
        for train_index, val_index in kf.split(range(len(list_key))):
            # val_index -- the index of ids in list_key
            indiv_split.append(np.array(list_key)[val_index])  # [5* [LIST]]
        target_split[key] = indiv_split
    return target_split  # {'EAC_XXX_XXX':[5*[LIST--subname+names.mha:10to15:1]], 'EAC_XXX_XXX':[5*[LIST--subname+names.mha:10to15:1]], ...}


def split_definer(split_list:dict, fold_order:int) ->Tuple[dict, dict]:
    # {'EAC_XXX_XXX':[5*[LIST]], 'EAC_XXX_XXX':[5*[LIST]], ...}
    # -> train {'EAC_XXX_XXX':[4LIST], 'EAC_XXX_XXX':[4LIST], ...}, val {'EAC_XXX_XXX':[LIST], 'EAC_XXX_XXX':[LIST] ...}
    train_split_dict = {}
    val_split_dict = {}
    for key in split_list.keys():
        val_split = list(split_list[key][fold_order])  # [LIST]
        train_split = []
        for i in range(5):
            if i != fold_order:
                train_split.extend(split_list[key][i])  # [LIST*4]
        val_split_dict[key] = val_split
        train_split_dict[key] = train_split
    return train_split_dict, val_split_dict


def balancer(target_list:dict, atlas_list:dict, target_category:list) ->dict:
    # 从subdir:names变成subdir+names
    # balance the data in atlas and target split
    keyname = list(target_list.keys())[0]
    at_keyname = list(atlas_list.keys())[0]
    assert (len(target_list.keys())==len(atlas_list.keys()))  # site and dirc should be the same
    # keyname can be different for ATL/EACorCSA,
    # but the length in EAC_XXX_XXX, should be the same
    length_target = len(target_list[keyname])
    length_atlas = len(atlas_list[at_keyname])
    capacity = [length_target, length_atlas]
    capacity_max = np.max(capacity)
    target_repeat = capacity_max // length_target
    atlas_repeat = capacity_max // length_atlas
    for key in target_list.keys():
        target_list[key] = target_list[key] * target_repeat
    for key in atlas_list.keys():
        atlas_list[key] = atlas_list[key] * atlas_repeat
    
    # merge the target and atlas train/val split to get the final train set name:cs:label list
    merged_list = {}
    for key in target_list.keys():
        if len(target_category) == 1:
            # prediction task, target and atlas share the same subdir 'EAC_XXX_XXX'
            merged_list[key] = target_list[key] + atlas_list[key]
        elif len(target_category) == 2:
            # classification task: from 'EAC_XXX_XXX' TO 'ATL_XXX_XXX'
            # target_category -- ['EAC', 'CSA', 'ATL']
            merge_name = key.replace(f'{target_category[0]}_', '')  # site_dirc
            at_name = key.replace(f'{target_category[0]}', f'{target_category[1]}')
            merged_list[merge_name] = target_list[key] + atlas_list[at_name]
    assert (len(target_list.keys())==len(merged_list.keys()))
    return merged_list  # {'site_dirc':[LIST(Target+Atlas): subdir\names.mha:cs:label ], ...}


def val_split_definer(split_list:dict) ->Tuple[dict, dict]:
    # {'EAC_XXX_XXX':[5*[LIST]], 'EAC_XXX_XXX':[5*[LIST]], ...}
    # -> train {'EAC_XXX_XXX':[4LIST], 'EAC_XXX_XXX':[4LIST], ...}, val {'EAC_XXX_XXX':[LIST], 'EAC_XXX_XXX':[LIST] ...}
    val_split_dict = {}
    for key in split_list.keys():
        val_split = []
        for i in range(5):
            val_split.extend(split_list[key][i])  # [LIST*5]
        val_split_dict[key] = val_split
    return val_split_dict