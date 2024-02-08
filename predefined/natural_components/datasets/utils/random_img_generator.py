import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import time


class RandomDataset(Dataset):

    def __init__(self, target_size=(28,28), target_label:int=0, data_capacity:int=1000, ):
        self.target_size = target_size
        self.target_label = target_label
        self.images = data_capacity

    def __len__(self):
        return len(self.images)

    def __getitem__(self):
        img= np.random.uniform(low=0, high=1, size=self.target_size)
        label = self.target_label
        
        return img.astype('float32'), label

    