from torch.utils.data import Dataset
import pandas as pd
import pickle
import os
import torchvision.transforms as transforms
import cv2
import numpy as np
from sklearn.model_selection import train_test_split


def _siim_balancer(list_merge:list)->list:
    # list of [[path, label], ...]
    array_merge = np.asarray(list_merge)[:, 1]
    array_merge = np.asarray(array_merge, dtype=int)
    target_ratio = np.sum(array_merge)/array_merge.shape[0]
    if target_ratio >= 0.5:
        print('no need for augmentation')
        return list_merge
    else:
        aug_ratio = int(np.minimum(1/target_ratio, 20))
        enhanced_list = []
        for item in list_merge:
            if item[1] == 1:
                for _ in range(aug_ratio):
                    enhanced_list.append(item)
            elif item[1] == 0:
                enhanced_list.append(item)
            else:
                raise ValueError(f'not valid label:{item[1]}, type:{type(item[1])}')
        return enhanced_list


def siim_initialization(train_dir:str='', load_save:bool=True):
    if load_save:
        with open(f'{train_dir}/siim-isic-melanoma-classification/path_list.pkl', "rb") as tf:
            train_merge, val_merge = pickle.load(tf)
    else: 
        meta_file = pd.read_csv(f'{train_dir}/siim-isic-melanoma-classification/train.csv')
        # val_meta_file = pd.read_csv(f'{train_dir}/siim-isic-melanoma-classification/test.csv')
        img_dir = f'{train_dir}/siim-isic-melanoma-classification/jpeg/train'
        # val_img_dir = f'{train_dir}/siim-isic-melanoma-classification/jpeg/test'
        meta_file['image_path'] = meta_file['image_name'].map(lambda x:  os.path.join(img_dir, x+'.jpg'))
        # val_meta_file['image_path'] = val_meta_file['image_name'].map(lambda x:  os.path.join(val_img_dir, x+'.jpg'))
        # show img:
        train_merge = []
        val_merge = []
        target_merge = []
        atlas_merge = []
        for i in range(len(meta_file['image_path'])):
            if meta_file['target'][i] == 1:
                target_merge.append([meta_file['image_path'][i], meta_file['target'][i]])
            elif meta_file['target'][i] == 0:
                atlas_merge.append([meta_file['image_path'][i], meta_file['target'][i]])
        
        train_target, val_target = train_test_split(target_merge, test_size=0.2)
        train_atlas, val_atlas = train_test_split(atlas_merge, test_size=0.2)
        train_merge.extend(train_target)
        train_merge.extend(train_atlas)
        val_merge.extend(val_target)
        val_merge.extend(val_atlas)
        # for j in range(len(val_meta_file['image_path'])):
            #val_merge.append([val_meta_file['image_path'][j], val_meta_file['target'][j]])
        print(f'length of the trainset: {len(train_merge)}')
        print(f'length of the valset: {len(val_merge)}')
        print('example:', train_merge[0])

        train_merge = _siim_balancer(train_merge)
        val_merge = _siim_balancer(val_merge)

        print(f'length of the augmented trainset: {len(train_merge)}')
        print(f'length of the augmented valset: {len(val_merge)}')
        print('example:', train_merge[0])

        stack = [train_merge, val_merge]
        with open(f'{train_dir}/siim-isic-melanoma-classification/path_list.pkl', "wb") as tf:
            pickle.dump(stack, tf)
    return train_merge, val_merge


class SIIMDataset(Dataset):
    def __init__(self, stacked_list:list=None, transform=None, val_flag:bool=False):
        # define transformation
        if val_flag:
            self.transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Resize((224, 224))])
        else:
            if transform==None:
                self.transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.RandomVerticalFlip(),
                                                transforms.RandomRotation(10),
                                                transforms.Resize((224, 224)),
                                                ])
            else:
                self.transform = transform
        # set dataset
        self.paths = stacked_list

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        paths = self.paths[index]
        full_img_path = paths[0]
        # gt_seg = paths[1]
        
        # read image
        image = cv2.imread(full_img_path)
        if self.transform is not None:
            image = self.transform(image)
        
        # gt not included

        label_str = paths[1]
        if label_str == 1:
            label = 1
        elif label_str == 0:
            label = 0
        else:
            raise ValueError(f'Wrong label: {label_str}, type:{type(label_str)}')
        
        return image.float(), label


if __name__ == '__main__':
    a, b = siim_initialization(train_dir='D:/ImageNet', load_save=False)