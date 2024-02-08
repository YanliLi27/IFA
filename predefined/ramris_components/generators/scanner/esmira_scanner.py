import os
import re
import numpy as np
import pandas as pd
# from scanner.scanner_utils.central_slice import central_selector
# from scanner.scanner_utils.ramris_reader import ramris_reader
# from scanner.scanner_utils.ramris_search import ramris_search
from predefined.ramris_components.generators.scanner.scanner_utils.central_slice import central_selector
from predefined.ramris_components.generators.scanner.scanner_utils.ramris_reader import ramris_reader
from predefined.ramris_components.generators.scanner.scanner_utils.ramris_search import ramris_search
from typing import Union, Tuple


def _create_blank_list(site_list:list, extra_list:list, extra_list2:Union[None, list]=None, img_flag:bool=True) ->dict:
    # extra can be dirc or reader
    if img_flag:
        u_flag = 'IMG'
    else:
        u_flag = 'RAMRIS'
    dictionary = {}
    for site in site_list:
        for extra in extra_list:
            if extra_list2:
                for extra2 in extra_list2:
                    list_name = f'{u_flag}_{site}_{extra}_{extra2}'
                    dictionary[list_name] = []
            else:
                list_name = f'{u_flag}_{site}_{extra}'
                dictionary[list_name] = []
    return dictionary

def esmira_scanner(data_root:str, common_dict:dict) ->pd.DataFrame:
    # common_list {'EAC':[ids], 'CSA':[ids], 'ATL':[ids]}
    default_site:list=['Wrist', 'MCP', 'Foot']
    default_dirc:list=['TRA', 'COR']
    default_reader:list=['Reader1', 'Reader2']
    default_biomarker:list=['ERO', 'BME', 'SYN', 'TSY']

    IMG_path:dict = _create_blank_list(default_site, default_dirc, img_flag=True)  # 'Wrist_TRA':[], 'Wrist_COR':[]
    RAMRIS_data:dict = _create_blank_list(default_site, default_biomarker, default_reader, img_flag=False)  # 'Wrist_Reader1':[], 'Wrist_Reader2':[]
    label_list:list = []  # one for one id
    id_list:list = []

    # merge the ids:
    id_list = []
    for key in list(common_dict.keys()):
        id_list.extend(common_dict[key])
    # id_list [ids for all category] get label from the img

    df_eac, df_csa, df_atl = ramris_reader('EAC'), ramris_reader('CSA'), ramris_reader('ATL')  # drop the useless information
    # simple ID-WRdata-MCPdata-Footdata
    length_id = len(id_list)
    output_id_list = []
    counter = 0
    # start the loop for all information
    for order, id in enumerate(id_list):
        output_id_list.append(id)
        counter += 1
        if counter % 20 == 0:
            print(f'progress: {counter}/{length_id}')
        # get the general category for both ramris and img path
        if 'Arth' in id:
            cate = 'EAC'
        elif 'Csa' in id:
            cate = 'CSA'
        elif 'Atlas' in id:
            cate = 'ATL'
        else:
            raise ValueError(f'id not valid:{id}')

        for site in default_site:
            # get the info for ramris
            for biomarker in default_biomarker:
                info_reader = {'reader1':[], 'reader2':[]}
                for i, reader in enumerate(default_reader):
                    ramris_column = f'{site}_{biomarker}_{reader}'

                    if cate == 'EAC':
                        simplified_id = id.replace('Arth', '')
                        simplified_id = simplified_id.replace('_EAC', '')
                        simplified_id = str(int(simplified_id))
                        ramris_data = ramris_search(df_eac, simplified_id, site, biomarker, reader)
                    elif cate == 'CSA':
                        simplified_id = id.replace('Csa', '')
                        simplified_id = simplified_id.replace('_CSA', '')
                        simplified_id = str(int(simplified_id))
                        ramris_data = ramris_search(df_csa, simplified_id, site, biomarker, reader)
                    elif cate == 'ATL':
                        simplified_id = id.replace('Atlas', '')
                        simplified_id = simplified_id.replace('_ATL', '')
                        simplified_id = str(int(simplified_id))
                        ramris_data = ramris_search(df_atl, simplified_id, site, biomarker, reader)  # list[scores of each sub-site in target site]
                    else:
                        raise ValueError('not valid category name')
                    RAMRIS_data[f'RAMRIS_{ramris_column}'].append(ramris_data)
                    info_reader[f'reader{i+1}'] = np.array(ramris_data, dtype=np.float16)
                
            # get the info for img
            for dirc in default_dirc:
                img_path_key = f'{site}_{dirc}'
                # get the abs_path:
                full_path_name = f'{cate}_{img_path_key}'
                abs_path_name = os.path.join(data_root, full_path_name)

                img_file_list = os.listdir(abs_path_name)  # list of names
                for img_name in img_file_list:
                    target_img_name = re.search(f'(.*){id}(.*)', img_name)  # find the img file names
                    if target_img_name:  # if find one, stop searching
                        break
                abs_file_path = os.path.join(abs_path_name, target_img_name.group())
                central_slice_range = central_selector(abs_file_path)  # str ':10to15'
                name_str = target_img_name.group() + central_slice_range  # 'Names_1.mha:10to15'
                name_str = os.path.join(full_path_name, name_str)
                IMG_path[f'IMG_{img_path_key}'].append(name_str)


        # add the label according to the name of img, outside the loop of other info
        id_wr_tra_path_cs = IMG_path['IMG_Wrist_TRA'][order]
        if 'Arth' in id_wr_tra_path_cs:
            if '_1.mha' in id_wr_tra_path_cs:
                label = 'EAC+'
            else:
                label = 'EAC-'
        elif 'Csa' in id_wr_tra_path_cs:
            if '_1.mha' in id_wr_tra_path_cs:
                label = 'CSA+'
            else:
                label = 'CSA-'
        elif 'Atlas' in id_wr_tra_path_cs:
            label = 'ATL'
        else:
            raise ValueError('not valid id wr tra path')
        label_list.append(label)

        # check the process
        for site in default_site:
            for dirc in default_dirc:
                img_path_key = f'IMG_{site}_{dirc}'
                assert len(label_list) == len(IMG_path[img_path_key])
            for biomarker in default_biomarker:
                for reader in default_reader:
                    ramris_column = f'RAMRIS_{site}_{biomarker}_{reader}'
                    assert len(label_list) == len(RAMRIS_data[ramris_column])


    # 完成所有部分的列表添加后，统一放入字典，并创建dataframe：
    label_dict = {'Abs_label':label_list}
    id_dict = {'Abs_ID':output_id_list}
    df_dict = {**id_dict, **label_dict, **IMG_path, **RAMRIS_data}  # merged dict for all info, with order of id
    df = pd.DataFrame(df_dict)
    df = df.set_index('Abs_ID', drop=False)
    return df