import os
import torchvision
from torchvision import transforms


def transform_generator(model_flag:str='vgg', task_flag:str='CatsDogs', phase:str='val'):
    transform_list = [transforms.ToTensor()]
    if task_flag == 'MNIST':
        transform_list.append(transforms.Normalize([0.5], [0.5]))
    else:
        if model_flag=='vgg' or model_flag=='resnet':
            transform_list.append(transforms.Resize((256, 256)))
        else:
            transform_list.append(transforms.Resize((224, 224)))
        
        if phase=='val' or phase=='test':
            transform_list.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
            
        else:
            transform_list.append(transforms.RandomCrop(224))
            transform_list.append(transforms.ColorJitter())
            transform_list.append(transforms.RandomHorizontalFlip())
            transform_list.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    data_transform = transforms.Compose(transform_list)
    return data_transform


def dataset_generator(model_flag:str='vgg', task:str='CatsDogs', dataset_split:str='val', pre_root=None):
    data_transform = transform_generator(model_flag, task, dataset_split)
    if task == 'CatsDogs':
        # define root:
        if pre_root == None:
            root = 'D:\\CatsDogs\\kaggle\\working\\extracted\\train' 
        else: 
            root = pre_root
        train_img_list = os.listdir(root)[:9999] + os.listdir(root)[12500:22499] # 0-12499 cats, 12500-24999 dogs
        val_img_list = os.listdir(root)[10000:12499] + os.listdir(root)[22499:]
        test_data_root = root.replace('train', 'test1')
        test_img_list = os.listdir(test_data_root)
        # set the dataset
        from predefined.natural_components.datasets.utils.datasets import CatDogDataset
        if dataset_split=='val':
            target_dataset = CatDogDataset(root, val_img_list, transform = data_transform)
        elif dataset_split=='train':
            target_dataset = CatDogDataset(root, train_img_list, transform = data_transform)
        elif dataset_split=='test':
            target_dataset = CatDogDataset(test_data_root, test_img_list, transform = data_transform)
        else:
            raise AttributeError('Not valid set split')
        in_channel = 3
        out_channel = 512
        num_classes = 2

    elif task == 'MNIST':  # root:str='D:\\CatsDogs\\MNIST'
        if pre_root == None:
            root = 'D:\\CatsDogs\\MNIST'
        else: 
            root = pre_root
        if dataset_split=='train':
            target_dataset = torchvision.datasets.MNIST(root=root, train=True, transform=data_transform, download=False)
        elif dataset_split=='val':
            target_dataset = torchvision.datasets.MNIST(root=root, train=False, transform=data_transform, download=False)
        else:
             raise AttributeError('Not valid set split')
        in_channel = 1
        out_channel = 200
        num_classes = 10

    elif task == 'Imagenet':
        if pre_root == None:
            root = 'D:\\CatsDogs\\Imagenet\\ILSVRC2012'
        else: 
            root = pre_root
        if dataset_split!='val':
            raise ValueError('imagenet only support validation set')
        target_dataset = torchvision.datasets.ImageFolder(root=root, transform=data_transform)
        in_channel = 3
        out_channel = 512
        num_classes = 1000
    
    elif task == 'ClickMe':
        if pre_root == None:
            root = 'D:\\CatsDogs\\ClickMe'
        else:
            root = pre_root
        if dataset_split!='val':
            raise ValueError('clickme only support validation set')
        from predefined.natural_components.datasets.utils.datasets import ClickMeDataset
        target_dataset = ClickMeDataset(root=root, transform=data_transform)
        in_channel = 3
        out_channel = 512
        num_classes = 1000

    else:
        raise ValueError('Not avaliable task')

    return target_dataset, in_channel, out_channel, num_classes