from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import pickle
import pandas as pd
import pickle
import cv2


def mura_initialization(train_dir:str='D:/ImageNet', category:str='XR_WRIST', load_save:bool=True):
    assert category in ['XR_ELBOW', 'XR_FINGER', 'XR_FOREARM', 'XR_HAND', 'XR_HUMERUS', 'XR_SHOULDER', 'XR_WRIST', None]
    if load_save:
        with open(f'{train_dir}/mura/path_list.pkl', "rb") as tf:
            train_paths, train_labels, val_paths, val_labels = pickle.load(tf)
    else: 
        train_label = pd.read_csv(f'{train_dir}/mura/MURA-v1.1/train_image_paths.csv',dtype=str,header=None)
        val_label = pd.read_csv(f'{train_dir}/mura/MURA-v1.1/valid_image_paths.csv')

        train_label.columns = ['image_path']
        val_label.columns = ['image_path']

        train_label['label'] = train_label['image_path'].map(lambda x:'positive' if 'positive' in x else 'negative')
        train_label['category']  = train_label['image_path'].apply(lambda x: x.split('/')[2])  
        train_label['patientId']  = train_label['image_path'].apply(lambda x: x.split('/')[3].replace('patient',''))
        # change the path at last one 
        train_label['image_path'] = train_label['image_path'].map(lambda x: x.replace('MURA-v1.1/', f'{train_dir}/mura/MURA-v1.1/'))

        val_label['label'] = val_label['image_path'].map(lambda x:'positive' if 'positive' in x else 'negative')
        val_label['category']  = val_label['image_path'].apply(lambda x: x.split('/')[2])  
        val_label['patientId']  = val_label['image_path'].apply(lambda x: x.split('/')[3].replace('patient',''))
        # change the path at last one
        val_label['image_path'] = val_label['image_path'].map(lambda x: x.replace('MURA-v1.1/', f'{train_dir}/mura/MURA-v1.1/'))

        # raise ValueError('testing')

        total_number_of_training_images = np.shape(train_label)[0]
        print("total number of images:",total_number_of_training_images )
        print("number of training images:",np.shape(train_label['image_path'])[0])

        categories_counts = pd.DataFrame(train_label['category'].value_counts())
        print ('categories:',categories_counts )
        print('number of patients:',train_label['patientId'].nunique())
        print('number of labels:',train_label['label'].nunique())
        print ('positive casses:',len(train_label[train_label['label']=='positive']))
        print ('negative casses:',len(train_label[train_label['label']=='negative']))

        
        print("data_shape:",np.shape(val_label))
        print("number of validation images:",np.shape(val_label['image_path']))

        validaton_categories_counts = pd.DataFrame(val_label['category'].value_counts())
        print ('categories:',validaton_categories_counts)
        print('number of patients:',val_label['patientId'].nunique())
        print('number of labels:',val_label['label'].nunique())
        print ('positive casses:',len(val_label[val_label['label']=='positive']))
        print ('negative casses:',len(val_label[val_label['label']=='negative']))

        if category:
            train_label = train_label[train_label['category']==category]  
            # {'image_path':[list of absolute paths], 'label':[], 'category':[], 'patientID':[]}
            val_label = val_label[val_label['category']==category]
        print('category-filtered cases in train:', len(train_label['label']))
        print('category-filtered cases in val:', len(val_label['label']))

        train_paths, train_labels, val_paths, val_labels =\
            train_label['image_path'].values, train_label['label'].values, val_label['image_path'].values, val_label['label'].values
    
        # print(val_paths[:10])
        # print(val_labels[:10])

        stack = [train_paths, train_labels, val_paths, val_labels]
        with open(f'{train_dir}/mura/MURA-v1.1/path_list_{category}.pkl', "wb") as tf:
            pickle.dump(stack, tf)
    return train_paths, train_labels, val_paths, val_labels


class MURADataset(Dataset):
    def __init__(self, paths:list, labels:list, transform=None, val_flag:bool=False):
        # define transformation
        if val_flag:
            import torchvision.transforms as transforms
            self.transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Resize((224, 224)),
                                            ])
        else:
            if transform==None:
                # from monai import transforms
                # self.transform = transforms.Compose([transforms.ToTensor(),
                #                                 transforms.RandGaussianNoise(0.2, 0, 0.1),
                #                                 transforms.RandFlip(0.5, 0),
                #                                 transforms.RandRotate((45), prob=0.5),
                #                                 transforms.RandAffine(prob=1.0, translate_range=(20, 20)),
                #                                 transforms.RandShiftIntensity(offsets=0.1, safe=True, prob=0.2),
                #                                 transforms.RandStdShiftIntensity(factors=0.1, prob=0.2),
                #                                 # transforms.RandBiasField(degree=2, coeff_range=(0, 0.1), prob=0.2),
                #                                 transforms.RandAdjustContrast(prob=0.5, gamma=(0.9, 1.1)),
                #                                 transforms.RandHistogramShift(num_control_points=10, prob=0.2),
                #                                 transforms.RandZoom(prob=0.3, min_zoom=0.9, max_zoom=1.0, keep_size=True),
                #                                 transforms.Resize(224),
                #                                 ])
                import torchvision.transforms as transforms
                self.transform = transforms.Compose([transforms.ToTensor(),
                                                    transforms.RandomHorizontalFlip(),
                                                    # transforms.RandomRotation(45, fill=0),
                                                    transforms.RandomAffine(degrees=10,translate=(0.1, 0.1), scale=(0.9, 1.1)),
                                                    transforms.Resize((224, 224)),
                                                    ])
            else:
                self.transform = transform
        # set dataset
        self.paths = paths
        self.labels = labels
        print('-----------------------------------initialization-----------------------------------')

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        image = cv2.imread(self.paths[index])
        # image = self._itensity_normalize(image)
        if self.transform is not None:
            image = self.transform(image)

        label_str = self.labels[index]
        if label_str == 'positive':
            label = 1
        elif label_str == 'negative':
            label = 0
        else:
            raise ValueError(f'Wrong label: {label_str}, type:{type(label_str)}')
        return image.float(), label
    
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




if __name__ == '__main__':
    mura_initialization(train_dir='D:/ImageNet', category='XR_WRIST', load_save=False)
