import os
import SimpleITK as sitk
import numpy as np
from torch.utils import data
from skimage.transform import resize
import torch
from typing import Union


class CLIPDataset(data.Dataset):
    def __init__(self, data_root:str, img_list:list, ramris_list:list,
                 transform=None, full_img:Union[bool, int]=False, dimension:int=2,
                 score_sum:bool=False, path_flag:bool=False):
        self.root = data_root  # the root of data
        self.img_list = img_list  # [ID[PATHS OF IMG], ...]
        self.ramris_list = ramris_list  # [ID[SCORES], ...]
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

        self.dimension = dimension
        self.score_sum = score_sum
        if isinstance(full_img, int):
            self.slices = full_img
            self.full_img = False
        else:
            self.slices = 20
            self.full_img = full_img
        self.path_flag = path_flag

    
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if self.full_img:
            data, path = self._load_full(idx)  # data list [scan-tra, scan-cor]
        else:
            data, path = self._load_file(idx)  # data list [scan-tra, scan-cor]
        # data list[N, array[5/7/20, 512, 512]], path
        scores = self._load_ramris(idx)
        for i in range(len(data)):
            data[i] = torch.from_numpy(data[i])
            if self.transform is not None:
                data[i] = self.transform(data[i])
        # data list [tensors]
        if self.dimension==2:
            data = torch.vstack(data)  # [Site*TRA/COR*slice/depth, length, width]
        else:
            data = torch.stack(data)  # [Site*TRA/COR, slice/depth, length, width]
        if self.path_flag:
            return data, scores, path
        return data, scores 


    def _load_file(self, idx):
        data_matrix = []
        paths = self.img_list[idx]
        for indiv_path in paths:
            # indiv_path: 'subdir\names.mha:cs'
            # updated: 'subdir\names.mha:1to6plus1to11'
            path, cs = indiv_path.split(':')  # 'subdir\names.mha', 'cs'
            five, ten = cs.split('plus')
            fivelower, fiveupper = five.split('to')
            tenlower, tenupper = ten.split('to')
            if self.slices == 5:
                lower, upper = fivelower, fiveupper
            else:
                lower, upper = tenlower, tenupper
            lower, upper = int(lower), int(upper)
            abs_path = os.path.join(self.root, path)
            data_mha = sitk.ReadImage(abs_path)
            data_array = sitk.GetArrayFromImage(data_mha)

            # for COR，using lower and upper
            if 'CORT1f' in abs_path:
                data_array = self._itensity_normalize(data_array[lower:upper])
            # for TRA, using step
            elif 'TRAT1f' in abs_path:
                if data_array.shape[0]//2 >= self.slices:
                    s = slice(0, 2*self.slices, 2)
                    data_array = self._itensity_normalize(data_array[s])
                else:
                    data_array = self._itensity_normalize(data_array[lower:upper])
            # [5, 512, 512]/[10, 512, 512]
            if data_array.shape != (self.slices, 512, 512):
                if data_array.shape == (self.slices, 256, 256):
                    data_array = resize(data_array, (self.slices, 512, 512), preserve_range=True)  # preserve_range: no normalization
                else:
                    raise ValueError('the shape of input:{}, the id: {}, central_slice: {}'.format(data_array.shape, path, lower))
            data_matrix.append(data_array.astype(np.float32))
        return data_matrix, abs_path  # [N, 5, 512, 512]

    def _load_full(self, idx):
        data_matrix = []
        paths = self.img_list[idx]
        for indiv_path in paths:
            # indiv_path: 'subdir\names.mha:cs'
            path, _ = indiv_path.split(':')  # 'subdir\names.mha', 'cs'
            abs_path = os.path.join(self.root, path)
            data_mha = sitk.ReadImage(abs_path)
            data_array = sitk.GetArrayFromImage(data_mha)
            data_array = self._itensity_normalize(data_array)  # [20, 512, 512]
            if data_array.shape != (20, 512, 512):
                data_array = resize(data_array, (20, 512, 512), preserve_range=True)  # preserve_range: no normalization      
            data_matrix.append(data_array.astype(np.float32))
        return data_matrix, abs_path  # [N, 20, 512, 512]
            

    def _load_ramris(self, idx):
        data_collect = self.ramris_list[idx]  # [batch, site, scores] -> [site, scores]
        list_data = []
        for data in data_collect:
            list_data.extend(data)
        if self.score_sum:
            return np.sum(np.asarray([list_data], dtype=np.float32), axis=1)
        else:
            return np.asarray(list_data, dtype=np.float32)


    def _itensity_normalize(self, volume: np.array):
        """
        normalize the itensity of a volume based on the mean and std of nonzeor region
        inputs:
            volume: the input volume
        outputs:
            out: the normalized volume
        """
        min_value = volume.min()
        max_value = volume.max()
        if max_value > min_value:
            out = (volume - min_value) / (max_value - min_value)
        else:
            out = volume
        # out_random = np.random.normal(0, 1, size=volume.shape)
        # out[volume == 0] = out_random[volume == 0]
        return out