import os
import numpy as np
import pandas as pd
import pickle


def _id_finder(names:list) ->list:
    ids = []
    for name in names:
        # name = 'ESMIRA-LUMC-Csa842_CSA-20210505-RightMCP_PostTRAT1f_0.mha'
        namelist = name.split('-')  # namelist = ['ESMIRA', 'LUMC', 'Csa842_CSA', '20210505', 'RightMCP_PostTRAT1f_0.mha']
        ids.append(namelist[2])  # add 'Csa842_CSA'
    return ids # list of ids 


def _common_list_generator(init_dict:dict) ->dict:
    common_list = {}
    # init_dict{'EAC_Wrist_TRA':idlist, ...}
    keys = init_dict.keys()  # 'EAC_Wrist_TRA', 'EAC_Wrist_COR'

    # find groups: not limited in EAC/CSA/ATL
    groups = []
    for key in keys:
        key_name = key.split('_')
        key_group = key_name[0]
        if key_group not in groups:
            groups.append(key_group)   
    # groups = ['EAC', 'CSA', 'ATL']
    assert len(groups)==3  # TODO for now, only 3
    for group in groups:
        group_id_list = []
        # 'EAC' as example,
        for key in keys:
            if group in key:
                group_id_list.append(init_dict[key])  # list of ids
        # group_id_list [[], [], [], [], ...] 6* [ids_length]
        group_common = set(group_id_list[0]).intersection(*group_id_list[1:])
        common_list[group] = group_common
    return common_list  # {'EAC':[LIST], 'CSA':[LIST], 'ATL':[LIST]}


def _common_list_finder(init_dict:dict) ->dict:
    dict_path = 'D:\\ESMIRAcode\\RA_CLIP\\generators/scanner/logs/img_id_list.pkl'
    if os.path.isfile(dict_path):
        with open(dict_path, "rb") as tf:
            common_list = pickle.load(tf)
        print('saved dataset dict:{}'.format(common_list.keys()))
    else:
        common_list = _common_list_generator(init_dict)
        with open(dict_path, "wb") as tf:
            pickle.dump(common_list, tf)
        print('------> common list saved <------')
    return common_list


def img_id_scanner(data_root:str) ->dict:
    target_category = ['EAC', 'CSA', 'ATL']
    target_site = ['Wrist', 'MCP', 'Foot']
    target_dirc = ['TRA', 'COR']
    # scan the data_root and got all the names with keys
    init_dict = {}
    for cate in target_category:
        for site in target_site:
            for dirc in target_dirc:
                dict_key = f'{cate}_{site}_{dirc}'
                root = os.path.join(data_root, dict_key)
                datalist = os.listdir(root)  # -- ['full name']
                idlist = _id_finder(datalist)
                init_dict[dict_key] = idlist
                print(f'-- {dict_key} id finished --')
    # init_dict{'EAC_Wrist_TRA':idlist, ...}
    # find the common list of sites:
    common_list = _common_list_finder(init_dict)  # {'EAC':[LIST], 'CSA':[LIST], 'ATL':[LIST]}
    return common_list


def ramris_id_filter(img_id_dict:dict) ->dict:
    target_category = ['EAC', 'CSA', 'ATL']
    # img_id_dict # ['EAC':[List of ids], 'CSA':[List of ids], 'ATL':[List of ids]]
    fused_id_dict = {}
    for cate in target_category:
        id_list = img_id_dict[cate]  # [list of ids]
        selected_id_list = []
        # get the RAMRIS
        RAMRIS_dir = 'D:\\ESMIRA\\SPSS data'
        RAMRIS_list = ['1. EAC baseline.csv', '3. Atlas.csv', '5. CSA_T1_MRI_scores_SPSS.csv']
        if cate == 'EAC':
            chosen_ramris = RAMRIS_list[0]
        elif cate == 'CSA':
            chosen_ramris = RAMRIS_list[2]
        elif cate == 'ATL':
            chosen_ramris = RAMRIS_list[1]
        else:
            raise ValueError('not valid class')
        RAMRIS_path = os.path.join(RAMRIS_dir, chosen_ramris)
        df = pd.read_csv(RAMRIS_path, sep=';')
        titles = [column for column in df]
        ramris_ids = df[titles[0]].values  # type: np.ndarray
        ramris_ids_regularized = []
        for id in ramris_ids:
            if cate == 'EAC':  # Arth3393_EAC
                id = 'Arth' + str(id).zfill(4) + '_EAC'
                ramris_ids_regularized.append(id)
            elif cate == 'CSA':  # Csa014_CSA
                id = 'Csa' + str(id).zfill(3) + '_CSA'
                ramris_ids_regularized.append(id) 
            elif cate == 'ATL':  # Atlas002_ATL
                id = 'Atlas' + str(id).zfill(3) + '_ATL'
                ramris_ids_regularized.append(id)
            else:
                raise ValueError('NoT VALID CLASS')
            
        for id in id_list:
            if id in ramris_ids_regularized:
                selected_id_list.append(id)
        fused_id_dict[cate] = selected_id_list


    return fused_id_dict  # dict ['EAC':[List of ids], 'CSA':[List of ids], 'ATL':[List of ids]]
