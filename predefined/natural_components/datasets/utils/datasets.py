from torch.utils.data import Dataset, DataLoader
import cv2
import os
import numpy as np


class CatDogDataset(Dataset):

    def __init__(self, train_dir, img_list, transform=None, random_label:bool=False):
        
        self.train_dir = train_dir
        self.transform = transform
        self.images = img_list
        self.random_label = random_label
        

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image_path = os.path.join(self.train_dir, self.images[index])
        label = self.images[index].split(".")[0]

        if self.random_label:
            label = np.random.randint(0, 2)
        else:
            if label == 'cat':
                label = 0
            elif label == 'dog':
                label = 1
            else:
                raise ValueError('Not valid label')
        img = cv2.imread(image_path)
        if self.transform:
            img = self.transform(img)
        img = img.numpy()
        return img.astype('float32'), label




class ClickMeDataset(Dataset):

    def __init__(self, root:str='', transform=None) ->None:
        self.transform = transform
        self.image_list = self._clickme_explainer(root)
        

    def __len__(self):
        return len(self.image_list)


    def _clickme_explainer(self, root:str) ->list:
        root = os.path.join(root, 'data')
        if not os.path.exists(root):
            raise ValueError(f'not valid root path for clickme:{root}')
        else:
            img_list = os.listdir(root)
        re_list = []
        for img in img_list:
            rename = os.path.join(root, img)
            re_list.append(rename)
        return re_list
        

    def __getitem__(self, index):
        vox = self.image_list[index]
        data = np.load(vox)  # ['label', 'clicks', 'heatmaps', 'image']
        img = data['image']
        if self.transform:
            img = self.transform(img)
        img = img.numpy()
        label = data['label']
        return img.astype('float32'), label



# import numpy as np
# import cv2

# for order in range(800):
#     path = f'D:\\ImageNet\\clickme_val\\{order}.npz'


#     data = np.load(path)
#     print(data.files)

#     print(data['label'])
#     print(data['clicks'])
#     print(data['heatmap'].shape)
#     print(data['image'].shape)

#     img = data['image']
#     img2 = np.zeros([256, 256, 3], dtype=np.uint8)
#     img2[:] = img

#     sm = np.zeros([256, 256, 3], dtype=np.uint8)
#     for i in range(3):
#         sm[:, :, i] = data['heatmap'] * 255

#     img3 = np.hstack([img2, sm])
#     cv2.imshow('image', img3)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
