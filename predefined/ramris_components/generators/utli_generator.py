import os
import pandas as pd
# from scanner.id_scanner import img_id_scanner, ramris_id_filter
# from scanner.esmira_scanner import esmira_scanner
# from dataset.clipdataset import CLIPDataset
# from dataset.esmiradataset import ESMIRADataset
from predefined.ramris_components.generators.scanner.id_scanner import img_id_scanner, ramris_id_filter
from predefined.ramris_components.generators.scanner.esmira_scanner import esmira_scanner
from predefined.ramris_components.generators.dataset.clipdataset import CLIPDataset
from predefined.ramris_components.generators.dataset.esmiradataset import ESMIRADataset
from torch.utils.data import Dataset
from typing import Union, Tuple
import pickle
from sklearn.model_selection import KFold
import numpy as np
from predefined.ramris_components.utils.distri_calculator import distri_calculators, corr_calculator, sca_calculator
from predefined.ramris_components.generators.dataset.utils.resample import resampler


# 代码结构：

# -- init -- 部分，用于构建完整的id -- img_path -- ramris scores -- label 的csv file
# 创建 csv file：df = pd.DataFrame(columns=['Abs_ID', 'Abs_label',
#            'IMG_Wrist_TRA', 'IMG_Wrist_COR', 'IMG_MCP_TRA', 'IMG_MCP_COR', 'IMG_Foot_TRA', 'IMG_Foot_COR',
#            'RAMRIS_Wrist_Reader1', 'RAMRIS_Wrist_Reader2', 'RAMRIS_MCP_Reader1', 'RAMRIS_MCP_Reader2',
#            'RAMRIS_Foot_Reader1', 'RAMRIS_Foot_Reader2'])
# 采用： df.set_index('Abs_ID', drop=True) 
# 1. 获取common dict -- 基于img和ramris 并保存成一个id_list
#    loop: for order, id in enumerate(common_dict):
# 2. 根据common dict 依照id的顺序，依使用正则表达式，依次计算的Img的path和central slice，分别保存成一个顺序的列表。
#    IMG_path {'Wrist_TRA':[id_ordered str+10to15, ...], ...}
# 3. 同时也可以从dataframe里面读RAMRIS， 同样顺序保存进入列表。RAMRIS_data {'Wrist_Reader1':[id_ordered array[], ...], ...}
# 4. Label 通过IMG_path['Wrist_TRA'][order] 中搜索ID和1.mha来进行判断 -- 分成五类
# 4. 全部保存进入dataframe


class ESMIRA_generator:
    def __init__(self, data_root:str, target_category:list=['EAC', 'ATL'], target_site:list=['Wrist'], target_dirc:list=['TRA', 'COR'],
                 target_reader:list=['Reader1', 'Reader2'], target_biomarker:Union[list, None]=None, task_mode:str='class',
                 working_dir:str='.', print_flag:bool=False) ->None:
        self.working_dir = working_dir
        self.print_flag = print_flag
        # ----------------------------------------------------- id initialization ----------------------------------------------------- #
        # The common ids: common_list is the dict of EAC/CSA/ATL, that patients exist in all target_site
        # {'EAC':[LIST], 'CSA':[LIST], 'ATL':[LIST]}
        task_mode = task_mode.lower()
        assert task_mode in ['class', 'clip', 'multi']
        self.data_root = data_root
        self.target_category = target_category
        self.target_site = target_site
        if self.target_site is None:
            self.target_site = ['Wrist', 'MCP', 'Foot']
        self.target_biomarker = target_biomarker
        if self.target_biomarker:
            for item in self.target_biomarker:
                assert item in ['ERO', 'BME', 'SYN', 'TSY']
        else:
            self.target_biomarker = ['ERO', 'BME', 'SYN', 'TSY']
            target_biomarker = ['ERO', 'BME', 'SYN', 'TSY']
        self.len_ramris_site = 0
        output_matrix = [[15, 15, 3, 10],[8, 8, 4, 8],[10, 10, 5, 10]]
        site_order = {'Wrist':0, 'MCP':1, 'Foot':2}
        bio_order = {'ERO':0, 'BME':1, 'SYN':2, 'TSY':3}
        for site in target_site:
            for biomarker in target_biomarker:
                self.len_ramris_site += output_matrix[site_order[site]][bio_order[biomarker]]
        self.default_id_path = f'{self.working_dir}/generators/scanner/logs/id_list.pkl'
        self.default_df_path = f'{self.working_dir}/generators/scanner/logs/utli_df.pkl'
        if os.path.isfile(self.default_id_path):
            with open(self.default_id_path, "rb") as tf:
                self.common_dict = pickle.load(tf)
        else:
            self.common_dict = img_id_scanner(self.data_root)
            self.common_dict = ramris_id_filter(self.common_dict)  # find the common list of img and ramris
            with open(self.default_id_path, "wb") as tf:
                pickle.dump(self.common_dict, tf)  # dict of [list of ids] common in img and ramris
        # self.common_dict {'EAC':[list of ids], 'CSA':[list of ids], 'ATL':[list of ids]}

        # ---------------------------------------------------- path initialization ---------------------------------------------------- #
        if os.path.isfile(self.default_df_path):
            ulti_df = pd.read_pickle(self.default_df_path)
        else:
            ulti_df = esmira_scanner(data_root, self.common_dict)  # df = pd.DataFrame(columns=['Abs_ID', 'Abs_label',\
        #            'IMG_Wrist_TRA', 'IMG_Wrist_COR', 'IMG_MCP_TRA', 'IMG_MCP_COR', 'IMG_Foot_TRA', 'IMG_Foot_COR',
        #            'RAMRIS_Wrist_ERO_Reader1', 'RAMRIS_Wrist_ERO_Reader2', 'RAMRIS_MCP_ERO_Reader1', 'RAMRIS_MCP_ERO_Reader2',
        #            'RAMRIS_Foot_ERO_Reader1', 'RAMRIS_Foot_ERO_Reader2', ...])
            ulti_df.to_pickle(path=self.default_df_path)
            ulti_df.to_csv(path_or_buf=self.default_df_path.replace('.pkl', '.csv'))
        print('------------> Dataframe Done <------------')

        # ------------------------------------------------------- main generator ------------------------------------------------------- #      
        # 根据id_list, target_site, target_dirc, target_reader 展开为[split1 [id1[path1:cs, path2:cs, ...]]]
        if task_mode in ['class', 'multi']:
            target_img_save = self._img_split_saver(target_category, target_site, target_dirc, True)
            target_ramris_save = self._ramris_split_saver(target_category, target_site, target_reader, target_biomarker, True, keyword='_sc')
            atlas_img_save = self._img_split_saver(target_category, target_site, target_dirc, False)
            atlas_ramris_save = self._ramris_split_saver(target_category, target_site, target_reader, target_biomarker, False, keyword='_sc')
            if os.path.isfile(target_img_save) and os.path.isfile(target_ramris_save) \
                 and os.path.isfile(atlas_img_save) and os.path.isfile(atlas_ramris_save):
                with open(target_img_save, "rb") as tf:
                    self.target_img_split = pickle.load(tf)
                with open(target_ramris_save, "rb") as tf:
                    self.target_ramris_split = pickle.load(tf)
                with open(atlas_img_save, "rb") as tf:
                    self.atlas_img_split = pickle.load(tf)
                with open(atlas_ramris_save, "rb") as tf:
                    self.atlas_ramris_split = pickle.load(tf)
                print('--------found saved split--------')

            else:
                # column filtering - site/dirc/reader
                self.df = self._input_filter(ulti_df, target_category, target_site, target_dirc,
                                              target_biomarker, target_reader)
                print('Remained keys: ', self.df.columns)
                # row filtering - id filtering
                target_df, atlas_df = self._id_input_filter(self.df, target_category, func_mode=task_mode)
                # split generation
                target_id_list = target_df['Abs_ID']  # select id
                print(target_id_list)
                target_id_split = self._split_generator(target_id_list)  # from [ids] to [5*[ids]]
                # atlas split generation
                atlas_id_list = atlas_df['Abs_ID']  # select id
                atlas_id_split = self._split_generator(atlas_id_list)  # from [ids] to [5*[ids]]
                # balance for classification
                t_a_ratio = len(target_id_list) // len(atlas_id_list)
                a_t_ratio = len(atlas_id_list) // len(target_id_list)
                if t_a_ratio > 1:
                    # t_a_ratio = np.minimum(t_a_ratio, 3)
                    target_id_split_balanced = target_id_split
                    atlas_id_split_balanced = []
                    for split in atlas_id_split:
                        atlas_id_split_balanced.append(split*t_a_ratio)
                elif t_a_ratio <= 0:
                    # a_t_ratio = np.minimum(a_t_ratio, 3)
                    atlas_id_split_balanced = atlas_id_split
                    target_id_split_balanced = []
                    for split in target_id_split:
                        target_id_split_balanced.append(split*a_t_ratio)

                self.target_img_split = self._img_path_loader(target_df, target_id_split_balanced, target_site, target_dirc)
                self.target_ramris_split, _ = self._ramris_loader(target_df, target_id_split_balanced, target_site,
                                                                target_biomarker, target_reader, wmse_flag=True)
                self.atlas_img_split = self._img_path_loader(atlas_df, atlas_id_split_balanced, target_site, target_dirc)
                self.atlas_ramris_split, _ = self._ramris_loader(atlas_df, atlas_id_split_balanced, target_site,
                                                               target_biomarker, target_reader, wmse_flag=True)

                with open(target_img_save, "wb") as tf:
                    pickle.dump(self.target_img_split, tf)
                with open(target_ramris_save, "wb") as tf:
                    pickle.dump(self.target_ramris_split, tf)  
                with open(atlas_img_save, "wb") as tf:
                    pickle.dump(self.atlas_img_split, tf)
                with open(atlas_ramris_save, "wb") as tf:
                    pickle.dump(self.atlas_ramris_split, tf)

        else:
            if target_category is None:
                target_category = ['EAC', 'CSA', 'ATL']
            target_img_save = self._img_split_saver(target_category, target_site, target_dirc, True)
            target_ramris_save = self._ramris_split_saver(target_category, target_site, target_reader, target_biomarker, True, keyword='_sc')
            target_mse_save = self._ramris_split_saver(target_category, target_site, target_reader, target_biomarker, True, keyword='_mse')
            if os.path.isfile(target_img_save) and os.path.isfile(target_ramris_save):
                with open(target_img_save, "rb") as tf:
                    self.target_img_split = pickle.load(tf)
                with open(target_ramris_save, "rb") as tf:
                    self.target_ramris_split = pickle.load(tf)
                with open(target_mse_save, "rb") as tf:
                    self.target_mse = pickle.load(tf)
                print('--------found saved split--------')
            else:
                # column filtering - site/dirc/reader
                self.df = self._input_filter(ulti_df, target_category, target_site, target_dirc, target_biomarker, target_reader)
                print('Remained keys: ', self.df.columns)
                print('Length of IDs, Cols: ', self.df.shape)
                # row filtering - id filtering
                target_df, _ = self._id_input_filter(self.df, target_category=target_category, func_mode=task_mode)  # atlas_df = None
                # split generation
                target_id_list = target_df['Abs_ID']
                target_id_split = self._split_generator(target_id_list)  # from [ids] to [5*[ids]]
                # get the img_path and ramris
                self.target_img_split = self._img_path_loader(target_df, target_id_split, target_site, target_dirc)
                self.target_ramris_split, self.target_mse = self._ramris_loader(target_df, target_id_split, target_site, target_biomarker, target_reader, wmse_flag=True)
                with open(target_img_save, "wb") as tf:
                    pickle.dump(self.target_img_split, tf)
                with open(target_ramris_save, "wb") as tf:
                    pickle.dump(self.target_ramris_split, tf)
                with open(target_mse_save, "wb") as tf:
                    pickle.dump(self.target_mse, tf)
            if self.print_flag:
                self._reader_corr_printer(self.target_ramris_split, self.target_mse)
        print('-------------------------------> Dataset Initialization Finished <-------------------------------')


    def _input_filter(self, df:pd.DataFrame, target_category:Union[None, list]=None,
                     target_site:list=['Wrist'], target_dirc:Union[None, list]=None,
                     target_biomarker:Union[None, list]=None,
                     target_reader:Union[None, list]=None) -> pd.DataFrame:
        print('task define:')
        # site is necessary
        default_site = ['Wrist', 'MCP', 'Foot']
        default_dirc = ['TRA', 'COR']
        default_biomarker = ['ERO', 'BME', 'SYN', 'TSY']
        default_reader = ['Reader1', 'Reader2']
        drop_site = list(set(default_site) ^ set(target_site))  # return the values not belong to A & B
        print(f'clip/class mode: with site{str(target_site)}, dirc{str(target_dirc)}, BIO{str(target_biomarker)} reader{str(target_reader)}')
        assert ((target_reader is not None) or (target_dirc is not None))
        if target_biomarker == None:
            target_biomarker = ['ERO', 'BME', 'SYN', 'TSY']
        drop_biomarker = list(set(default_biomarker)^ set(target_biomarker))  # return the values not belong to A & B
        drop_dirc = list(set(default_dirc) ^ set(target_dirc))
        drop_icon = drop_dirc + drop_biomarker  # cause only one will be targeted
        drop_reader = list(set(default_reader) ^ set(target_reader))
        titles = [column for column in df]
        drop_column = []
        for title in titles:
            title_comp = title.split('_')  # IMG/RAMRIS _ SITE _ (BIOMARKER/DIRC) _ READER
            if len(title_comp) >= 3:
                if title_comp[1] in drop_site:
                    drop_column.append(title)  
                if title_comp[2] in drop_icon:
                    drop_column.append(title)
            if ((len(title_comp) >= 4) and (target_reader is not None)):
                if title_comp[3] in drop_reader:
                    drop_column.append(title)
        df = df.drop(labels=drop_column,axis=1)
        return df


    def _id_input_filter(self, df:pd.DataFrame, target_category:Union[None, list]=None, func_mode:str='clip') ->Tuple[pd.DataFrame, Union[pd.DataFrame, None]]:
        
        if func_mode == 'clip':
            if target_category == None:
                return df, None
            else:
                target_category_re = []
                if 'EAC' in target_category:
                    target_category_re.append('EAC+')
                    target_category_re.append('EAC-')
                if 'CSA' in target_category:
                    target_category_re.append('CSA+')
                    target_category_re.append('CSA-')
                if 'ATL' in target_category:
                    target_category_re.append('ATL')
                df = df[df['Abs_label'].isin(target_category_re)]
                return df, None
            
        elif func_mode in ['class', 'multi']:
            target_category_re = []
            atlas_category_re = []
            if len(target_category) == 2:
                if 'EAC' in target_category:
                    target_category_re.append('EAC+')
                    target_category_re.append('EAC-')
                elif 'CSA' in target_category:
                    target_category_re.append('CSA+')
                    target_category_re.append('CSA-')
                atlas_category_re.append('ATL')
            elif len(target_category) == 1:
                if 'EAC' in target_category:
                    target_category_re.append('EAC+')
                    atlas_category_re.append('EAC-')
                elif 'CSA' in target_category:
                    target_category_re.append('CSA+')
                    atlas_category_re.append('CSA-')
            else:
                raise ValueError('not valid category')
            target_df = df[df['Abs_label'].isin(target_category_re)]
            atlas_df = df[df['Abs_label'].isin(atlas_category_re)]

            return target_df, atlas_df
        else:
            raise ValueError('not valid func mode')


    def _split_generator(self, list_ids:list, shuffle:bool=True) ->list:
        # input: dict {'id':label -length==5N}
        # return: list [list[id, id,...], list[id, id,...], list[id, id,...], ...]
        list_ids  # [id, id, id]
        kf = KFold(n_splits=5, shuffle=shuffle)
        target_split_list = []
        for train_index, val_index in kf.split(range(len(list_ids))):
            # val_index -- the ids in val set
            split_sublist = []
            split_list = list(np.array(list_ids)[val_index])  # [ids, ...]
            for id in split_list:
                split_sublist.append(id)
            target_split_list.append(split_sublist)

        return target_split_list


    def _img_path_loader(self, df:pd.DataFrame, id_split:list, target_site:list, target_dirc:list) ->list:
        # id_split [5 * split[ids]]  -- return [5 * split[id[path1:cs, path2:cs, path3:cs, ...]
        remain_columns = ['Abs_ID']
        for site in target_site:
            for dirc in target_dirc:
                remain_columns.append(f'IMG_{site}_{dirc}')
        all_columns = [column for column in df]
        drop_columns = list(set(remain_columns) ^ set(all_columns))
        df = df.drop(labels=drop_columns, axis=1)
        return_5_split = []
        for split in id_split:
            return_split = []
            for id in split:
                paths = (df[df['Abs_ID']==id].values)[0][1:]  # [ ['IDs', path1, path2, ...] ]
                return_split.append(paths)
            return_5_split.append(return_split)
        return return_5_split  # return [5 * split[id[path1:cs, path2:cs, path3:cs, ...]
    

    def _ramris_filter(self, path_list:list, num_reader:int=2, merge_mode:str='avg', nan_rep:int=0, wmse_flag:bool=False):
        # pickle to make sure array/list in path_list, rather than str in path_list
        if num_reader == 1:
            paths = []
            num_sites = len(path_list)
            for i in range(num_sites):
                ramris = np.array(path_list[i])
                for j, item in enumerate(ramris):
                    if item >= 99:
                        ramris[j] = nan_rep
                    if item < 0:
                        ramris[j] = -item
                paths.append(ramris)
            return paths   # list [site1_array, site2_array]
        elif num_reader == 2:
            paths = []
            num_sites = len(path_list) // 2  # -- the number of sites that has been used
            if merge_mode == 'avg':
                if wmse_flag == True:
                    wmse = []
                    for i in range(num_sites):
                        reader1_score = np.absolute(np.array(path_list[2*i]))
                        reader2_score = np.absolute(np.array(path_list[2*i+1]))
                        for j in range(len(reader1_score)):
                            if reader1_score[j] >= 20:
                                reader1_score[j] = nan_rep
                            if reader2_score[j] >= 20:
                                reader2_score[j] = nan_rep
                        avg = (reader1_score + reader2_score)/2  # (np.array + np.array)/2 -> np.array 
                        mse_indiv = (avg - reader2_score)**2
                        paths.append(avg)
                        wmse.append(mse_indiv)
                    return paths, wmse  # list [site1_array, site2_array], list[mse_value[split, split, ...], mse_value, mse_value]
                else:
                    for i in range(num_sites):
                        avg = (np.array(path_list[2*i]) + np.array(path_list[2*i+1]))/2
                        for j, item in enumerate(avg):
                            if item >=48:
                                avg[j] = nan_rep
                        paths.append(avg)
                    return paths  # list [site1_array, site2_array]
            else:
                raise ValueError('not supported merge mode')


    def _ramris_loader(self, df:pd.DataFrame, id_split:list, target_site:list, target_bio:list, target_reader:list, wmse_flag:bool=False) ->Tuple[list, list]:
        # id_split [5 * split[ids]]  -- return [5 * split[id[path1:cs, path2:cs, path3:cs, ...]
        remain_columns = ['Abs_ID']
        for site in target_site:
            for biomarker in target_bio:
                for reader in target_reader:
                    remain_columns.append(f'RAMRIS_{site}_{biomarker}_{reader}')
        all_columns = [column for column in df]
        drop_columns = list(set(remain_columns) ^ set(all_columns))
        df = df.drop(labels=drop_columns, axis=1)
        return_5_split = []
        return_5_wmse = []
        # weight_5_split = []
        for split in id_split:
            return_split = []
            return_wmse = []
            for id in split:
                paths = (df[df['Abs_ID']==id].values)[0][1:]  # [- delete through 0 ['IDs' - delete through 1:, path1, path2, ...] ]
                scores = np.zeros((1, 0), dtype=np.float32)
                wmses =  np.zeros((1, 0), dtype=np.float32)
                for i in range(len(paths)//2):
                    score, wmse = self._ramris_filter(paths[2*i:2*i+2], num_reader=len(target_reader), wmse_flag=wmse_flag)  # list [site1'array[]', site2'array[]']
                    scores = np.append(scores, score, axis=1)
                    wmses = np.append(wmses, wmse, axis=1)
                return_split.append(scores)  # list [id[site1_array, site2_array], id[[site1_array, site2_array]], ...]
                return_wmse.append(wmses)  # list [id[site1_mse, site2_mse], id[site1_mse, site2_mse], ...]
                # weight_split.append(weight_evi)
            return_5_split.append(return_split)
            return_5_wmse.append(return_wmse)
            # weight_5_split.append(weight_split)
        return return_5_split, return_5_wmse  # return [5 * split[id[site1_array, site2_array], id[[site1_array, site2_array]] ...]
            # return2 [5 * split[id[site1_mse, site2_mse], id[[site1_mse, site2_mse]] ...]


    def _img_split_saver(self, target_category:list, target_site:list, target_dirc:list, target_flag:bool=True)->str:
        cate_name = ''
        if target_category is not None:
            for cate in target_category:
                cate_name = cate_name + cate + '_'
        else:
            cate_name = cate_name + 'All_'
        if len(target_site) > 1:
            site_name = str(len(target_site)) + 'site'
        else:
            site_name = target_site[0]
        
        if len(target_dirc)>1:
            reader_name = str(len(target_dirc)) + 'dirc'
        else:
            reader_name = target_dirc[0]

        if target_flag:
            target_name = 1
        else:
            target_name = 0
        output_name = "{}/generators/scanner/recover/{}_{}_{}_{}.pkl".format(self.working_dir, cate_name, site_name, reader_name, target_name)
        return output_name


    def _ramris_split_saver(self, target_category:list, target_site:list, target_reader:list,
                            target_biomarker:Union[list, None]=None, target_flag:bool=True, keyword:str='')->str:
        cate_name = ''
        if target_category is not None:
            for cate in target_category:
                cate_name = cate_name + cate + '_'
        else:
            cate_name = cate_name + 'All_'
        if len(target_site) > 1:
            site_name = str(len(target_site)) + 'site'
        else:
            site_name = target_site[0]
        
        if len(target_reader)>1:
            reader_name = str(len(target_reader)) + 'reader'
        else:
            reader_name = target_reader[0]

        if target_flag:
            target_name = 1
        else:
            target_name = 0
        if target_biomarker:
            if len(target_biomarker) > 1:
                t_b = 'allbio'
            else:
                t_b = target_biomarker[0]
            output_name = "{}/generators/scanner/recover/{}_{}_{}_{}_{}_{}.pkl".format(self.working_dir, cate_name, site_name, t_b, reader_name, target_name, keyword)
        else:
            output_name = "{}/generators/scanner/recover/{}_{}_{}_{}_{}.pkl".format(self.working_dir, cate_name, site_name, reader_name, target_name, keyword)
        return output_name


    def _split_definer(self, split_list:list, fold_order:int) ->Tuple[list, list]:
        # list [[id[], id[], ...] *5] 5 fold
        # -> train list [id[], id[], id[], ...], val list [id[], id[], id[], ...]
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


    def _ta_mixer(self, target_list:list, atlas_list:list) ->Tuple[list, list]:
        target_label = list(np.ones(len(target_list), dtype=np.int64))
        atlas_label = list(np.zeros(len(atlas_list), dtype=np.int64))
        mix_list = target_list + atlas_list
        label_list = target_label + atlas_label
        return mix_list, label_list


    def _mse_printer(self, train_mse_list:list, val_mse_list:list, len_ramris_site:int=43) ->Tuple[np.float32, np.float32, np.array, np.array]:
        # input: train list [id[site1_mse, site2_mse], id[site1_mse, site2_mse], id[mse], ...], val list [id[mse], id[mse], id[mse], ...]
        # -> output: float, float
        array_val = np.asarray(val_mse_list)  # [num_id, num_site, num_ramris_in_site (1)]
        array_all = np.asarray((train_mse_list + val_mse_list))
        print('the score shape:', array_val.shape)
        val_mse = np.mean(array_val).astype(np.float32)
        overall_mse = np.mean(array_all).astype(np.float32)

        ndarray_val = array_val.reshape((-1, len_ramris_site))
        ndarray_all = array_all.reshape((-1, len_ramris_site))
        split_val_mse = []
        split_all_mse = []
        for i in range(len_ramris_site):
            split_val_mse.append(np.mean(ndarray_val[:,i]))
            split_all_mse.append(np.mean(ndarray_all[:,i]))
        return overall_mse, val_mse, split_all_mse, split_val_mse
    

    def _corr_printer(self, score_array, mse_array, fig_path:str=f'./generators/background/dist.jpg', num_scores_per_site:int=43) -> Tuple[str, str, np.float32, np.float32]:
        # score_array -- [batch/num_patients, num_scores_per_site]
        if not os.path.exists(os.path.dirname(fig_path)):
            os.makedirs(os.path.dirname(fig_path))
        try:
            distri_calculators(score_array, num_scores_per_site, division=False, save_name=fig_path)  # create the distribution
            distri_path = fig_path
        except:
            raise ValueError('distribution calculation failed')
        
        # correlation
        try:
            corr_mse_score, p_value = corr_calculator(score_array, mse_array, num_scores_per_site, division=None)
        except:
            raise ValueError('correlation calculation failed')
        return distri_path, corr_mse_score, p_value


    def _reader_corr_printer(self, ramris_list:list, mse_list:list) ->None:
        # ramris_list [5 * split[id[site1_array, site2_array], id[[site1_array, site2_array]] ...]  (5, ids, num_scores)
        # mse_list [5 * split[id[site1_mse, site2_mse], id[[site1_mse, site2_mse]] ...] (5, ids, num_scores)
        extend_ramris = []
        extend_mse = []
        if len(ramris_list) == 5:
            for item in ramris_list:
                extend_ramris.extend(item)  # [ids, num_scores]
            for mse in mse_list:
                extend_mse.extend(mse)  # [ids, num_scores]
        else:
            extend_ramris = ramris_list
            extend_mse = mse_list
        array_ramris = np.asarray(extend_ramris)
        array_mae = np.sqrt(np.asarray(extend_mse))
        assert array_mae.shape == array_ramris.shape
        array_reader1 = array_ramris + array_mae  # [ids, num_scores]
        array_reader2 = array_ramris - array_mae  # [ids, num_scores]
        array_reader1 = np.sum(array_reader1, axis=1)
        array_reader2 = np.sum(array_reader2, axis=1)
        # get the correlation
        corr_mse_score, p_value = corr_calculator(array_reader1, array_reader2, array_ramris.shape[1], division=None)
        print('dataset readers correlation:', corr_mse_score)
        print('correlation p value:', p_value)
        sca_calculator(Garray=array_reader1, Parray=array_reader2, num_scores_per_site=array_ramris.shape[1],
                        division=False, save_path=f'{self.working_dir}/generators/background/scatter_gt_pr.jpg')


    def returner(self, task_mode:str='clip', phase:str='train', fold_order:int=0, material:Union[str, list]='img',
                 monai:bool=False, full_img:bool=False, dimension:int=2) ->Tuple[Union[None, Dataset], Dataset]:
        if monai:
            from monai import transforms
            transform = [transforms.Compose([
                                            # transforms.RandGaussianNoise(0.2, 0, 0.1),
                                            transforms.RandFlip(0.5, 0),
                                            transforms.RandRotate((10), prob=0.5),
                                            transforms.RandAffine(prob=1.0, translate_range=(20, 20)),
                                            # transforms.RandShiftIntensity(offsets=0.1, safe=True, prob=0.2),
                                            # transforms.RandStdShiftIntensity(factors=0.1, prob=0.2),
                                            # transforms.RandBiasField(degree=2, coeff_range=(0, 0.1), prob=0.2),
                                            # transforms.RandAdjustContrast(prob=0.5, gamma=(0.9, 1.1)),
                                            transforms.RandHistogramShift(num_control_points=10, prob=0.2),
                                            transforms.RandZoom(prob=0.3, min_zoom=0.9, max_zoom=1.0, keep_size=True)
                                            ]),
                        transforms.Compose([
                                            transforms.RandAffine(prob=0.0, translate_range=(20, 20)),
                                            ]),
                        ]           

        else:
            from torchvision import transforms
            transform = [transforms.Compose([transforms.RandomHorizontalFlip(0.5),
                                            transforms.RandomRotation((10),),
                                            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(1, 1), shear=None, fill=0),
                                        ]),  
                        transforms.Compose([
                                        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(1, 1), shear=None, fill=0),
                                        ])]
        
        if task_mode == 'clip':
            return self._clip_returner(phase, fold_order, transform, full_img, dimension)
        elif task_mode == 'class':
            return self._class_returner(phase, fold_order, material, transform, dimension)
        elif task_mode == 'multi':
            return self._multi_returner(phase, fold_order, material, transform, dimension)


    def _clip_returner(self, phase:str='train', fold_order:int=0, transform=[None, None],
                       full_img:bool=False, dimension:int=2) ->Tuple[Union[None, Dataset], Dataset]:
        if phase=='train':
            train_img_list, val_img_list = self._split_definer(self.target_img_split, fold_order) 
            train_ramris_list, val_ramris_list = self._split_definer(self.target_ramris_split, fold_order) 
            train_mse_list, val_mse_list = self._split_definer(self.target_mse, fold_order)
            
            # data balance
            train_img_list, train_ramris_list, train_mse_list =\
                  resampler(train_img_list, train_ramris_list, train_mse_list)
            # val_img_list, val_ramris_list, val_mse_list =\
            #       resampler(val_img_list, val_ramris_list, val_mse_list)

            # statistic
            if self.print_flag:
                overall_mse, val_mse, split_all_mse, split_val_mse = self._mse_printer(train_mse_list, val_mse_list, self.len_ramris_site)
                print(f'overall mse: {overall_mse}, val mse: {val_mse}')
                print(f'overall split mse: {split_all_mse}')
                print(f'val split mse: {split_val_mse}')
                distri_all_path, corr_mse_score, p_value = self._corr_printer(np.asarray((train_ramris_list)), 
                                                                          np.asarray((train_mse_list)),
                                                                          fig_path=f'{self.working_dir}/models/figs/fold_{fold_order}/train_{phase}_{fold_order}_dist.jpg',
                                                                          num_scores_per_site=self.len_ramris_site)
                distri_val_path, val_corr_mse_score, val_p_value = self._corr_printer(np.asarray(val_ramris_list), 
                                                                                  np.asarray(val_mse_list),
                                                                                  fig_path=f'{self.working_dir}/models/figs/fold_{fold_order}/val_{phase}_{fold_order}_dist.jpg',
                                                                                  num_scores_per_site=self.len_ramris_site)
                print(f'distribution of ramris in all set saved in {distri_all_path}, in val set saved in {distri_val_path}')
                print(f'corr of mse and rmaris in all is: {corr_mse_score}, p value is: {p_value}')
                print(f'for val the corr is {val_corr_mse_score}, val p value: {val_p_value}')

            # dataset
            train_dataset = CLIPDataset(self.data_root, train_img_list, train_ramris_list, transform[0], full_img, dimension=dimension)
            val_dataset = CLIPDataset(self.data_root, val_img_list, val_ramris_list, transform[1], full_img, dimension=dimension)
        else:
            val_img_list = self._val_split_definer(self.target_img_split)
            val_ramris_list = self._val_split_definer(self.target_ramris_split)
            val_dataset = CLIPDataset(self.data_root, val_img_list, val_ramris_list, transform[1], full_img, dimension=dimension)
            train_dataset = None
        return train_dataset, val_dataset  

    
    def _class_returner(self, phase:str='train', fold_order:int=0, material:Union[str, list]='img',
                        transform=[None, None], dimension:int=2) ->Tuple[Union[None, Dataset], Dataset]:
        assert material.lower() in ['all', 'img', 'ramris']
        if phase=='train':
            target_train_img_list, target_val_img_list = self._split_definer(self.target_img_split, fold_order) 
            atlas_train_img_list, atlas_val_img_list = self._split_definer(self.atlas_img_split, fold_order)
            train_img_list, train_label = self._ta_mixer(target_train_img_list, atlas_train_img_list)
            val_img_list, val_label = self._ta_mixer(target_val_img_list, atlas_val_img_list)
            target_train_ramris_list, target_val_ramris_list = self._split_definer(self.target_ramris_split, fold_order) 
            atlas_train_ramris_list, atlas_val_ramris_list = self._split_definer(self.atlas_ramris_split, fold_order)
            train_ramris_list, _ = self._ta_mixer(target_train_ramris_list, atlas_train_ramris_list)
            val_ramris_list, _ = self._ta_mixer(target_val_ramris_list, atlas_val_ramris_list)
            train_dataset = ESMIRADataset(self.data_root, train_img_list, train_ramris_list, train_label,
                                           material, transform[0], dimension=dimension)
            val_dataset = ESMIRADataset(self.data_root, val_img_list, val_ramris_list, val_label,
                                         material, transform[1], dimension=dimension)
        else:
            target_val_img_list = self._val_split_definer(self.target_img_split)
            atlas_val_img_list = self._val_split_definer(self.atlas_img_split)
            val_img_list, val_label = self._ta_mixer(target_val_img_list, atlas_val_img_list)
            target_val_ramris_list = self._val_split_definer(self.target_ramris_split)
            atlas_val_ramris_list = self._val_split_definer(self.atlas_ramris_split)
            val_ramris_list, val_label = self._ta_mixer(target_val_ramris_list, atlas_val_ramris_list)
            val_dataset = ESMIRADataset(self.data_root, val_img_list, val_ramris_list, val_label,
                                         material, transform[1], dimension=dimension)
            train_dataset = None
        return train_dataset, val_dataset 
    

    def _multi_returner(self, phase:str='train', fold_order:int=0, material:Union[str, list]='img',
                        transform=[None, None], dimension:int=2) ->Tuple[Union[None, Dataset], Dataset]:
        assert material in ['img', 'ramris'] or material==['img', 'ramris']
        if phase=='train':
            target_train_img_list, target_val_img_list = self._split_definer(self.target_img_split, fold_order) 
            atlas_train_img_list, atlas_val_img_list = self._split_definer(self.atlas_img_split, fold_order)
            train_img_list, train_label = self._ta_mixer(target_train_img_list, atlas_train_img_list)
            val_img_list, val_label = self._ta_mixer(target_val_img_list, atlas_val_img_list)
            target_train_ramris_list, target_val_ramris_list = self._split_definer(self.target_ramris_split, fold_order) 
            atlas_train_ramris_list, atlas_val_ramris_list = self._split_definer(self.atlas_ramris_split, fold_order)
            train_ramris_list, _ = self._ta_mixer(target_train_ramris_list, atlas_train_ramris_list)
            val_ramris_list, _ = self._ta_mixer(target_val_ramris_list, atlas_val_ramris_list)
            train_dataset = ESMIRADataset(self.data_root, train_img_list, train_ramris_list, train_label,
                                           material, transform[0], dimension=dimension)
            val_dataset = ESMIRADataset(self.data_root, val_img_list, val_ramris_list, val_label,
                                         material, transform[1], dimension=dimension)
        else:
            target_val_img_list = self._val_split_definer(self.target_img_split)
            atlas_val_img_list = self._val_split_definer(self.atlas_img_split)
            val_img_list, val_label = self._ta_mixer(target_val_img_list, atlas_val_img_list)
            target_val_ramris_list = self._val_split_definer(self.target_ramris_split)
            atlas_val_ramris_list = self._val_split_definer(self.atlas_ramris_split)
            val_ramris_list, val_label = self._ta_mixer(target_val_ramris_list, atlas_val_ramris_list)
            val_dataset = ESMIRADataset(self.data_root, val_img_list, val_ramris_list, val_label,
                                         material, transform[1], dimension=dimension)
            train_dataset = None
        return train_dataset, val_dataset 

if __name__ == '__main__':
    ESMIRA_generator('D:\\ESMIRA\\ESMIRA_common', target_category=None, 
                        target_site=['Wrist'], target_dirc=['TRA', 'COR'], target_reader=['Reader1', 'Reader2'])