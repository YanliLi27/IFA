from torch.utils.data import Dataset
import monai.transforms as transforms
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import SimpleITK as sitk
from matplotlib import pyplot as plt
from typing import Literal
from skimage.transform import resize


def visual(img:np.array):
    plt.imshow(img, cmap='grey')
    plt.show()
    plt.waitforbuttonpress()


def kits_intialization(datacsv:str=f'./dataprepare/splitdatapath.npy') ->list:
    datakits = np.load(datacsv, allow_pickle=True)
    data1 = datakits.item().get('label1')  # [crit[p1[imgpath, maskpath],p2,...],minor[p1,p2,...], depend[p1,p2,..], none[p1,p2]]
    data0 = datakits.item().get('label0')
    collect1 = []
    for d in data1:
        collect1.extend(list(d))
    collect0 = []
    for d in data0:
        collect0.extend(list(d))
    train1, val1 = train_test_split(collect1, test_size=0.2)
    train0, val0 = train_test_split(collect0, test_size=0.2)

    trainmerge = []
    trainmerge.extend(train1)  # [p1[img, mask, 1], p2, p3, ...]
    trainmerge.extend(train0)  # [p1[img, mask, 0], p2, p3, ...]
    valmerge = []
    valmerge.extend(val1)
    valmerge.extend(val0)
    return trainmerge, valmerge
    

class KitsDataset(Dataset):
    def __init__(self, stacked_list:list, transform=None, val_flag:bool=False, repeat:int=3, maskout:Literal[True, False, 'mask']=False):
        # define transformation
        if val_flag:
            self.transform = None
        else:
            if transform==None:
                self.transform = transforms.Compose([
                                        # transforms.Resized(keys=['image', 'mask'], spatial_size=[repeat, 512, 512], mode='nearest'),
                                        transforms.RandGaussianNoised(keys=['image'], prob=0.8, mean=0, std=0.1, allow_missing_keys=True),
                                        # transforms.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=(1,2)),
                                        # transforms.RandRotated(keys=['image', 'mask'], range_y=(10), range_z=(10), prob=0.5),
                                        transforms.RandAffined(keys=['image', 'mask'], prob=0.5, translate_range=(50, 50)),
                                        # transforms.RandShiftIntensityd(keys=['image'], offsets=0.1, safe=True, prob=0.2, allow_missing_keys=True),
                                        # transforms.RandStdShiftIntensityd(keys=['image'], factors=0.1, prob=0.2, allow_missing_keys=True),
                                        # transforms.RandBiasFieldd(keys=['image'], degree=2, coeff_range=(0, 0.1), prob=0.2, allow_missing_keys=True),
                                        # transforms.RandAdjustContrastd(keys=['image'], prob=0.5, gamma=(0.9, 1.1), allow_missing_keys=True),
                                        # transforms.RandHistogramShiftd(keys=['image'], num_control_points=10, prob=0.2, allow_missing_keys=True),
                                        # transforms.RandZoomd(keys=['image', 'mask'], prob=0.7, min_zoom=0.9, max_zoom=1.0, keep_size=True),
                                        ])
            else:
                self.transform = transform
        # set dataset
        self.paths = stacked_list
        self.repeat = repeat
        self.maskout = maskout

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        paths = self.paths[index]  # [p1[img, mask, 1], p2, p3, ...]
        img_path, mask_path, label = paths[0], paths[1], paths[2]
        # read image
        imgpath, sliceimg = img_path.split(';')
        maskpath, slicemask = mask_path.split(';')
        sliceimg = int(sliceimg)
        slicemask = int(slicemask)
        image = sitk.ReadImage(imgpath)
        image = sitk.GetArrayFromImage(image)[:, :, sliceimg]  # [512, 512]
        image[image<-500.0]= -500.0
        image[image>500.0]= 500
        # visual(self._itensity_normalize(image))
        image = self._itensity_normalize(image)
        image = np.expand_dims(image, axis=0) # [self.repeat, 512, 512]

        mask = sitk.ReadImage(maskpath)
        mask = sitk.GetArrayFromImage(mask)[:, :, slicemask]
        # visual(self._itensity_normalize(mask))
        mask = self._itensity_normalize(mask)
        mask = np.expand_dims(mask, axis=0) # [self.repeat, 512, 512]
        if image.shape!=(self.repeat, 512, 512) or mask.shape!=(self.repeat, 512, 512):
            try:
                image = image[:, :512, :512]
                mask = mask[:, :512, :512]
            except:
                raise ValueError(f'not supported shape: {image.shape}, {mask.shape}, path:{imgpath}, slice{sliceimg}')
            
        if self.transform is not None:
            pair = {'image':image, 'mask':mask}
            pair = self.transform(pair)
            image, mask = pair['image'], pair['mask']
        else:
            image, mask = torch.from_numpy(np.asarray(image, dtype=np.float32)), torch.from_numpy(np.asarray(mask, dtype=np.float32))
        # image, mask = torch.squeeze(image, dim=0), torch.squeeze(mask, dim=0)
        # visual(image.get_array()[0])
        # visual(mask.get_array()[0])
        if self.maskout==True:
            return  image, mask, np.int64(label)
        elif self.maskout=='mask':
            return mask, np.int64(label)
        return image, np.int64(label)


    def _itensity_normalize(self, volume: np.array):
        """
        normalize the itensity of a volume based on the mean and std of nonzeor region
        inputs:
            volume: the input volume
        outputs:
            out: the normalized volume
        """
        return (volume - volume.min())/(volume.max() - volume.min()+1e-7)
        # return (volume + 500) / 1000


    def _itensity_normalize_hpyer(self, volume: np.array):
        """
        normalize the itensity of a volume based on the mean and std of nonzeor region
        inputs:
            volume: the input volume
        outputs:
            out: the normalized volume
        """
        self.para_k = (np.arctanh(0.9) - np.arctanh(0.1))/1000
        self.para_b = (np.arctanh(0.9)*(-500)-np.arctanh(0.1)*500)/(-1000)
        return np.tanh(self.para_k*volume+self.para_b)