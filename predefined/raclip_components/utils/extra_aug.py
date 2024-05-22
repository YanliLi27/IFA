import torch
import torchvision.transforms as transforms
import random
import numpy as np


def extra_aug(original_img): # [batch, Z, 512, 512]
    data_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation((10),),
        transforms.RandomAffine(0, translate=(0.05, 0), scale=(1, 1), shear=None, fill=0),
    ])  
    return data_transform(original_img) 


if __name__ == "__main__":
    import SimpleITK as sitk
    import matplotlib.pyplot as plt
    data_nii = sitk.ReadImage(r'D:\\MedicalData\\Train_Sets\\MR\\1\\T1DUAL\\DICOM_anon\\3d.nii')
    data_array = sitk.GetArrayFromImage(data_nii).astype(float) # [35, 256, 256]
    batch_data = np.zeros([2, 35, 256, 256])
    batch_data[0] = data_array
    batch_data[1] = data_array
    data_tensor = torch.from_numpy(batch_data)
    print(data_tensor.shape)
    data_aug = extra_aug(data_tensor)
    data_aug_array = data_aug.numpy()
    print(data_aug_array[0][0].shape)
    plt.figure(figsize=(20, 20))
    plt.subplot(2, 2, 1)
    plt.imshow(batch_data[0][0], cmap='gray')
    plt.subplot(2, 2, 2)
    plt.imshow(batch_data[0][1], cmap='gray')
    plt.subplot(2, 2, 3)
    plt.imshow(data_aug_array[0][0], cmap='gray')
    plt.subplot(2, 2, 4)
    plt.imshow(data_aug_array[0][1], cmap='gray')
    plt.show()

