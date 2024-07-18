import os
import SimpleITK as sitk
import numpy as np
from torch.utils import data
from typing import Tuple
from skimage.transform import resize
import torch
from predefined.synaug_components.generators.dataset.cleanup import clean_main
from predefined.synaug_components.generators.dataset.central_slice import central_selector


class SynaugReg(data.Dataset):
    def __init__(self, datalist:list[str], transform=None, mean_std:bool=False, path_flag:bool=False, cleanup:bool=False):
        # train_dict {'site_dirc':[LIST(Target+Atlas): subdir\names.mha:cs:label ], ...}
        self.data = datalist # [N* [origin, aug]]
        self.transform = transform # default to be nothing
        self.mean_std = mean_std
        self.path_flag = path_flag
        self.cleanup = cleanup

 
    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        data, aug = self._load_file(idx)
        data = torch.from_numpy(data)
        aug = torch.from_numpy(aug)
        path = self.data[idx][0]
        if self.path_flag:
            return data, aug, path
        return data, aug  # [N, 7, 512, 512], 1:int
    
    def _readimg(self, path):
        data_array = sitk.ReadImage(path)
        data_array = sitk.GetArrayFromImage(data_array)
        _, central_7_start = central_selector(data_array)
        data_array = self._itensity_normalize(data_array[central_7_start:central_7_start+7])  # [5, 512, 512]
        if data_array.shape != (7, 512, 512):
            if data_array.shape == (7, 256, 256):
                data_array = resize(data_array, (7, 512, 512), preserve_range=True)  # preserve_range: no normalization
            else:
                raise ValueError('the shape of input:{}, the id: {}'.format(data_array.shape, path))
        if self.cleanup:
            data_array = clean_main(data_array)
        return np.expand_dims(data_array, axis=0)  # [7, 512, 512] -> [1, 7, 512, 512]


    def _load_file(self, idx):  # item -- [5, 512, 512] * N
        data_array = self._readimg(self.data[idx][0])
        aug_array = self._readimg(self.data[idx][1])
        return np.array(data_array).astype(np.float32), np.array(aug_array).astype(np.float32)  # [N*5, 512, 512], 1:int


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
    

    def _itensity_normalize_ms(self, volume: np.array):
        """
        normalize the itensity of a volume based on the mean and std of nonzeor region
        inputs:
            volume: the input volume
        outputs:
            out: the normalized volume
        """
        mean = np.mean(volume[volume!=0])
        std = np.std(volume[volume!=0])
        out = (volume[volume!=0] - mean)/std
        return out