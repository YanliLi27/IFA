import os
import SimpleITK as sitk
import numpy as np
from torch.utils import data
from skimage.transform import resize
import torch
from typing import Union


class ESMIRADataset(data.Dataset):
    def __init__(self, data_root:str, img_list:list, ramris_list:list, label:list, material:Union[str, list]='img', transform=None,
                 dimension:int=2):
        self.root = data_root
        self.img_list = img_list
        self.ramris_list = ramris_list
        self.label = label
        self.material = material
        
        if 'img' in self.material:
            self.transform = transform
        else:
            self.transform = None
        
        self.dimension = dimension


    def __len__(self):
        return len(self.label)


    def __getitem__(self, idx):
        label = self.label[idx]
        if len(self.material)>1:
            # img load
            img = self._load_img_file(idx)
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
            return img, ramris, label.astype(np.int64)
        elif 'img' in self.material:
            img = self._load_img_file(idx)
            for i in range(len(img)):
                img[i] = torch.from_numpy(img[i])
                if self.transform is not None:
                    img[i] = self.transform(img[i])
            img = torch.vstack(img)
            return img, label.astype(np.int64)
        elif 'ramris' in self.material:
            ramris = self._load_ramris_file(idx)
            return ramris, label.astype(np.int64)  # [N*5, 512, 512], 1:int
        else:
            raise ValueError('materials not valid')


    def _load_img_file(self, idx):
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