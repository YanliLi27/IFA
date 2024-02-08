from torch.utils.data import Dataset
import pickle
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt
import torch


def _import_img(folder, target) ->list:
    images = []
    for item in os.listdir(folder):
        if '_mask.png' in item:
            continue
        else:
            img = os.path.join(folder,item)
            gt_path = img.replace('.png', '_mask.png')
            if (os.path.isfile(img) and os.path.isfile(gt_path)):
                images.append([img,gt_path,target])
    return images

def us_initialization(train_dir:str='', classes:int=3, load_save:bool=True):
    if load_save:
        with open(f'{train_dir}/Dataset_BUSI_with_GT/path_list076.pkl', "rb") as tf:
            train_merge, val_merge = pickle.load(tf)
    else: 
        benign = _import_img(f'{train_dir}/Dataset_BUSI_with_GT/benign/', 0)  # 437
        train_benign, val_benign = train_test_split(benign, test_size=0.2)
        malignant = _import_img(f'{train_dir}/Dataset_BUSI_with_GT/malignant/', 1)  # 231
        train_malignant, val_malignant = train_test_split(malignant, test_size=0.2)
        if classes == 3:
            normal = _import_img(f'{train_dir}/Dataset_BUSI_with_GT/normal/', 2)  # 133
            train_normal, val_normal = train_test_split(normal, test_size=0.2)
        
        train_merge = []
        train_merge.extend(train_benign)
        train_merge.extend(train_malignant)
        if classes == 3:
            train_merge.extend(train_normal)

        val_merge = []
        val_merge.extend(val_benign)
        val_merge.extend(val_malignant)
        if classes == 3:
            val_merge.extend(val_normal)
        # show img:
        # k = 16
        # plt.imshow(cv2.imread(train_merge[k][0]))
        # print(train_merge[k][2])
        # plt.show()
        # plt.imshow(cv2.imread(val_merge[k][0]))
        # print(val_merge[k][2])
        # plt.show()

        stack = [train_merge, val_merge]
        with open(f'{train_dir}/Dataset_BUSI_with_GT/path_list.pkl', "wb") as tf:
            pickle.dump(stack, tf)
    return train_merge, val_merge


class USDataset(Dataset):
    def __init__(self, stacked_list:list=None, transform=None, val_flag:bool=False, repeat:int=3):
        # define transformation
        if val_flag:
            self.transform = transforms.Compose([#transforms.ToTensor(),
                                            transforms.Resize((128, 128))])
        else:
            if transform==None:
                self.transform = transforms.Compose([#transforms.ToTensor(),
                                                transforms.RandomHorizontalFlip(),
                                                # transforms.RandomRotation(10),
                                                transforms.Resize((128, 128)),
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
        # gt_seg = paths[1]
        
        # read image
        image = cv2.imread(full_img_path, 0)/255
        image = torch.from_numpy(image).unsqueeze(dim=0).repeat(self.repeat, 1, 1)
        if self.transform is not None:
            image = self.transform(image)
        
        # gt not included

        label_str = paths[2]
        if label_str == 1:
            label = 1
        elif label_str == 0:
            label = 0
        elif label_str == 2:
            label = 2
        else:
            raise ValueError(f'Wrong label: {label_str}, type:{type(label_str)}')
        
        return image.float(), label


if __name__ == '__main__':
    a, b = us_initialization(train_dir='D:/ImageNet', load_save=False)