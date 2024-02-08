from torch.utils.data import Dataset, DataLoader
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


class CatDogDataset3D(Dataset):

    def __init__(self, train_dir, img_list, transform=None, random_label:bool=False):
        
        self.train_dir = train_dir
        self.transform = transform
        self.images = img_list
        self.random_label = random_label
        self.depth:int=5
        self.block_ratio:float=0.2
        

    def __len__(self):
        return len(self.images)
    
    def _dimension_increase(self, img:np.array):
        # [3, 224, 224]
        assert len(img.shape)==3
        img = np.expand_dims(img, axis=1)  # [3, 1, 224, 224]
        block_depth = int(self.depth*self.block_ratio)
        block_range = self.depth-block_depth-1  # 20 -5 -1 -- range in [0,15]
        # assert (block_range < depth)
        img = np.repeat(img, repeats=self.depth, axis=1)  # [3, depth, 224, 224]
        # print(img.shape)
        block_index = np.random.randint(0, block_range)  # [0, 15]
        # print('block_range:', block_range, ', index: ',block_index, ' to ', (block_index+block_depth))
        img[:, block_index:(block_index+block_depth), :] = 0  # [3, depth, 224, 224] --> block some slices

        # self._show_img(img)

        return img
    
    def _show_img(self, img:np.array) ->None:
        for i in range(img.shape[1]):
            show_img = np.transpose(img[:, i, :], (1, 2, 0))
            plt.figure(f'Image {i}')
            plt.imshow(show_img)
            plt.show()
    
    def __getitem__(self, index):
        image_path = os.path.join(self.train_dir, self.images[index])
        label = self.images[index].split(".")[0]

        if self.random_label:
            label = np.random.randint(0, 2)
        else:
            label = 0 if label == 'cat' else 1
        img = cv2.imread(image_path)
        if self.transform:
            img = self.transform(img)
        img = img.numpy()  # [channel=3, 224, 224]
        img_3d = self._dimension_increase(img)  # [channel=3, depth=20, 224, 224]
        # print(img_3d.shape)
        return img_3d.astype('float32'), label
