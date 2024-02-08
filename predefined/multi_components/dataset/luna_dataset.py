import h5py
import numpy as np
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt
import pickle
from torch.utils.data import Dataset
import torch


def luna_cropped_initialization(train_dir:str='', load_save:bool=True):
    if load_save:
        with open(f'{train_dir}/luna_cropped/path_list.pkl', "rb") as tf:
            train_merge, val_merge = pickle.load(tf)
    else:
        f = h5py.File(f'{train_dir}/luna_cropped/all_patches.hdf5','r')
        print(list(f.keys()))

        ct_slices = f['ct_slices']
        slice_class = f['slice_class']

        ct_slices = np.array(ct_slices)
        slice_class = np.array(slice_class)
        print(ct_slices.shape)
        print(slice_class.shape)
        print(ct_slices[1].min())

        # preprocess
        ct_slices = np.clip(ct_slices, -1000,320)
        
        # merge and split
        merge_list = []
        for i in range(len(ct_slices)):
            merge_list.append([ct_slices[i], slice_class[i]])
        print(len(merge_list))

        train_merge, val_merge = train_test_split(merge_list, test_size=0.2)
        stack = [train_merge, val_merge]
        with open(f'{train_dir}/luna_cropped/path_list.pkl', "wb") as tf:
            pickle.dump(stack, tf)
    return train_merge, val_merge


class LunaDataset(Dataset):
    def __init__(self, stacked_list:list=None, transform=None, val_flag:bool=False, repeat:int=3):
        # define transformation
        if val_flag:
            self.transform = None
        else:
            if transform==None:
                # self.transform = None
                self.transform = transforms.Compose([# transforms.ToTensor(),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.RandomVerticalFlip(),
                                                # transforms.RandomRotation(10),
                                                #transforms.Resize((64, 64)),
                                                ])
            else:
                self.transform = transform
        # set dataset
        self.paths = stacked_list
        self.repeat = repeat

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        paths = self.paths[index]
        full_img_path = paths[0]
        
        # read image
        # print(full_img_path.shape)
        image = torch.from_numpy(full_img_path).unsqueeze(dim=0).repeat(self.repeat, 1, 1)
        #print(image.shape)
        if self.transform is not None:
            image = self.transform(image)
        image = self._itensity_normalize(image)
        #print(image.shape)
        # gt not included

        label = int(paths[1])
        # print(label)
        
        return image.float(), label
    

    def _itensity_normalize(self, volume: np.array):
        """
        normalize the itensity of a volume based on the mean and std of nonzeor region
        inputs:
            volume: the input volume
        outputs:
            out: the normalized volume
        """
        min_value = -1000
        max_value = 320
        out = (volume - min_value) / (max_value - min_value)
        return out


if __name__ == '__main__':
    a, b = luna_cropped_initialization(train_dir='D:/ImageNet', load_save=False)