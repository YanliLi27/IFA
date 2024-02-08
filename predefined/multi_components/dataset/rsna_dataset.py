from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pydicom import dcmread
from PIL import Image
import torchvision.transforms as transforms
import pickle



def rsna_initialization(train_dir:str='', load_save:bool=True):
    if load_save:
        with open(f'{train_dir}/rsna-pneumonia-detection-challenge/path_list083.pkl', "rb") as tf:
            train_paths, train_labels, val_paths, val_labels = pickle.load(tf)
    else: 
        label_data = pd.read_csv(f'{train_dir}/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv')
        columns = ['patientId', 'Target']
        label_data = label_data.filter(columns)
        print(label_data.head(5))
        # split the train and test
        train_labels, val_labels = train_test_split(label_data.values, test_size=0.1)  # 10 percent for test  
        train_f = f'{train_dir}/rsna-pneumonia-detection-challenge/stage_2_train_images'
        train_f = f'{train_dir}/rsna-pneumonia-detection-challenge/stage_2_train_images'
        # test_f = f'{train_dir}/rsna-pneumonia-detection-challenge/stage_2_test_images'
        train_paths = [os.path.join(train_f, image[0]) for image in train_labels]
        val_paths = [os.path.join(train_f, image[0]) for image in val_labels]
        stack = [train_paths, train_labels, val_paths, val_labels]
        with open(f'{train_dir}/rsna-pneumonia-detection-challenge/path_list.pkl', "wb") as tf:
            pickle.dump(stack, tf)
    return train_paths, train_labels, val_paths, val_labels


class RSNADataset(Dataset):
    def __init__(self, paths:list, labels:list, transform=None, val_flag:bool=False):
        # define transformation
        if val_flag:
            self.transform = transforms.Compose([transforms.Resize(224),
                                            transforms.ToTensor()])
        else:
            if transform==None:
                self.transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                transforms.RandomRotation(10),
                                                transforms.Resize(224),
                                                transforms.ToTensor()])
            else:
                self.transform = transform
        # set dataset
        self.paths = paths
        self.labels = labels

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        image = dcmread(f'{self.paths[index]}.dcm')
        image = image.pixel_array
        image = image / 255.0
        # image = self._itensity_normalize(image)

        image = (255*image).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(image).convert('RGB')

        label = self.labels[index][1]

        image = self.transform(image)
            
        return image, label
    
    
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