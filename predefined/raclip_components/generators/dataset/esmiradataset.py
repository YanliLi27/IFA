import os
import SimpleITK as sitk
import numpy as np
from torch.utils import data
from skimage.transform import resize
import torch
from typing import Union


class ESMIRADataset(data.Dataset):
    def __init__(self, data_root:str, img_list:list, ramris_list:list, label:list, material:Union[str, list]='img', 
                 full_img:Union[bool, int]=False,
                 transform=None, dimension:int=2, path_flag:bool=False):
        self.root = data_root
        self.img_list = img_list
        self.ramris_list = ramris_list
        self.label = label
        self.material = material
        if isinstance(full_img, int):
            self.slices = full_img
            self.full_img = False
            assert self.slices in [5, 7]
        else:
            self.full_img = full_img
        
        if 'img' in self.material:
            self.transform = transform
        else:
            self.transform = None
        
        self.dimension = dimension
        self.path_flag = path_flag


    def __len__(self):
        return len(self.label)


    def __getitem__(self, idx):
        label = self.label[idx]
        if type(self.material)==list and len(self.material)>1:
            # img load
            img, path = self._load_img_full(idx) if self.full_img else self._load_img_file(idx)
            for i in range(len(img)):
                img[i] = torch.from_numpy(img[i])
                if self.transform is not None:
                    img[i] = self.transform(img[i])
            if self.dimension==2:
                img = torch.vstack(img)
            else:
                img = torch.stack(img)
            # ramris load
            ramris = self._load_ramris_file(idx)
            if self.path_flag:
                return img, ramris, label.astype(np.int64), path
            return img, ramris, label.astype(np.int64)
        elif 'img' in self.material:
            img, path = self._load_img_full(idx) if self.full_img else self._load_img_file(idx)
            for i in range(len(img)):
                img[i] = torch.from_numpy(img[i])
                if self.transform is not None:
                    img[i] = self.transform(img[i])
            if self.dimension==2:
                img = torch.vstack(img)
            else:
                img = torch.stack(img)
            if self.path_flag:
                return img, label.astype(np.int64), path
            return img, label.astype(np.int64)
        elif 'ramris' in self.material:
            ramris = self._load_ramris_file(idx)
            if self.path_flag:
                return ramris, label.astype(np.int64), path  # [N*5, 512, 512], 1:int
            return ramris, label.astype(np.int64)  # [N*5, 512, 512], 1:int
        else:
            raise ValueError('materials not valid')


    def _load_img_file(self, idx):
        data_matrix = []
        paths = self.img_list[idx]
        for indiv_path in paths:
            # indiv_path: 'subdir\names.mha:cs'
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
                    
            if data_array.shape != (self.slices, 512, 512):
                if data_array.shape == (self.slices, 256, 256):
                    data_array = resize(data_array, (self.slices, 512, 512), preserve_range=True)  # preserve_range: no normalization
                else:
                    raise ValueError('the shape of input:{}, the id: {}, central_slice: {}'.format(data_array.shape, path, lower))
            data_matrix.append(data_array.astype(np.float32))
        return data_matrix, abs_path  # [N*5, 512, 512]
    
    def _load_img_full(self, idx):
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
        return data_matrix, abs_path  # [N*20, 512, 512]


    def _load_ramris_file(self, idx):
        data_collect = self.ramris_list[idx]
        list_data = []
        for data in data_collect:
            list_data.extend(data)
        return torch.from_numpy(np.asarray(list_data, dtype=np.float32))


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