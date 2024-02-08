import os
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
    dict_path = './predefined/esmira_components/dataset/dicts/common_ids.pkl'
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


def ESMIRA_scanner(data_root:str) ->dict:
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


