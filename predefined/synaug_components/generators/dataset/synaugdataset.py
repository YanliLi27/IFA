import os
import SimpleITK as sitk
import numpy as np
from torch.utils import data
from typing import Tuple
from skimage.transform import resize
import torch
from predefined.synaug_components.generators.dataset.cleanup import clean_main


class SynaugReg(data.Dataset):
    def __init__(self, datalist:list[str], transform=None, mean_std:bool=False, path_flag:bool=False, cleanup:bool=False):
        # train_dict {'site_dirc':[LIST(Target+Atlas): subdir\names.mha:cs:label ], ...}
        self.data = datalist
        self.transform = transform # default to be nothing
        self.mean_std = mean_std
        self.path_flag = path_flag
        self.cleanup = cleanup

 
    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        data, aug, path = self._load_file(idx)
        data = torch.from_numpy(data)
        aug = torch.from_numpy(aug)
        if self.transform is not None:
            pair = {'image':data, 'mask':mask}
            pair = self.transform(pair)
            data, mask = pair['image'], pair['mask']
        if self.path_flag:
            return data, aug, path
        return data, aug  # [N*5, 512, 512], 1:int
    
    def _readimg(self, path):
        data_array = sitk.ReadImage(path)
        data_array = sitk.GetArrayFromImage(data_array)
        data_array = self._itensity_normalize(data_array)  # [5, 512, 512]
        if data_array.shape != (5, 512, 512):
            if data_array.shape == (5, 256, 256):
                data_array = resize(data_array, (5, 512, 512), preserve_range=True)  # preserve_range: no normalization
            else:
                raise ValueError('the shape of input:{}, the id: {}, central_slice: {}'.format(data_array.shape, path))
        if self.cleanup:
            data_array = clean_main(data_array)
        return data_array


    def _load_file(self, idx):  # item -- [5, 512, 512] * N
        data_array = self._readimg(self.data[idx][0])
        aug_array = self._readimg(self.data[idx][1])
        return np.array(data_array).astype(np.float32), np.array(aug_array).astype(np.float32), self.data[idx]  # [N*5, 512, 512], 1:int


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