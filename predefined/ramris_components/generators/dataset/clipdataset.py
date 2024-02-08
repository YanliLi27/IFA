import os
import SimpleITK as sitk
import numpy as np
from torch.utils import data
from skimage.transform import resize
from predefined.ramris_components.generators.dataset.utils.resample import resampler
import torch


class CLIPDataset(data.Dataset):
    def __init__(self, data_root:str, img_list:list, ramris_list:list,
                 transform=None, full_img:bool=False, dimension:int=2):
        self.root = data_root  # the root of data
        self.img_list = img_list  # [ID[PATHS OF IMG], ...]
        self.ramris_list = ramris_list  # [ID[SCORES], ...]
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None
        self.full_img = full_img
        self.dimension = dimension
    
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if self.full_img:
            data = self._load_full(idx)  # data list [scan-tra, scan-cor]
        else:
            data = self._load_file(idx)  # data list [scan-tra, scan-cor]
        scores = self._load_ramris(idx)
        for i in range(len(data)):
            data[i] = torch.from_numpy(data[i])
            if self.transform is not None:
                data[i] = self.transform(data[i])
        # data list [tensors]
        if self.dimension==2:
            data = torch.vstack(data)
        else:
            data = torch.stack(data)  # [Site*TRA/COR, slice/depth, length, width]
        return data, scores 


    def _load_file(self, idx):
        data_matrix = []
        paths = self.img_list[idx]
        for indiv_path in paths:
            # indiv_path: 'subdir\names.mha:cs'
            path, cs = indiv_path.split(':')  # 'subdir\names.mha', 'cs'
            lower, upper = cs.split('to')
            lower, upper = int(lower), int(upper)
            abs_path = os.path.join(self.root, path)
            data_mha = sitk.ReadImage(abs_path)
            data_array = sitk.GetArrayFromImage(data_mha)
            data_array = self._itensity_normalize(data_array[lower:upper])  # [5, 512, 512]
            if data_array.shape != (5, 512, 512):
                if data_array.shape == (5, 256, 256):
                    data_array = resize(data_array, (5, 512, 512), preserve_range=True)  # preserve_range: no normalization
                else:
                    raise ValueError('the shape of input:{}, the id: {}, central_slice: {}'.format(data_array.shape, path, lower))
            data_matrix.append(data_array.astype(np.float32))
        return data_matrix  # [N*5, 512, 512]

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
        return data_matrix  # [N*20, 512, 512]
            

    def _load_ramris(self, idx):
        data_collect = self.ramris_list[idx]  # [batch, site, scores]
        list_data = []
        for data in data_collect:
            list_data.extend(data)
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