import torch
import torch.nn as nn
import torchvision
import os
import torchvision.transforms as transforms


def _get_model(task_name:str=''):
    if task_name == 'catsdogs3d':
        from predefined.multi_components.models.catsdogs_3d_model import catsdogs_3d_nn
        model = catsdogs_3d_nn(in_channel=3, num_classes=2)
        target_layer = [model.cnn[-2]]
        out_channel = 512
    else:
        raise ValueError(f'task name not supported {task_name}, type:{type(task_name)}')
    return model, target_layer, out_channel


def _get_weight(task_name:str=''):
    default_dir = 'D:/CatsDogs'
    if task_name == 'catsdogs3d':
        weight_path = f'{default_dir}/models/catsdogs3d/best_model.pth.tar'
    else:
        raise ValueError(f'task name not supported {task_name}, type:{type(task_name)}')
    if os.path.isfile(weight_path):
        weight = torch.load(weight_path)
        print('weights loaded')
    else:
         raise ValueError(f'weight path not exist {weight_path}')
    return weight


def _get_dataset(task_name:str=''):
    if task_name == 'catsdogs3d':
        from predefined.multi_components.dataset.catsdogs_3d import CatDogDataset3D
        train_dir = 'D:/CatsDogs/kaggle/working/extracted/train'
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.ColorJitter(),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), # MEAN=[0.485, 0.456, 0.406], STD=[0.229, 0.224, 0.225]
        ])
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), # MEAN=[0.485, 0.456, 0.406], STD=[0.229, 0.224, 0.225]
        ])
        train_dir = train_dir
        train_img_list = os.listdir(train_dir)[:9999] + os.listdir(train_dir)[12500:22499] # 0-12499 cats, 12500-24999 dogs
        val_img_list = os.listdir(train_dir)[10000:12499] + os.listdir(train_dir)[22499:]
        train_dataset = CatDogDataset3D(train_dir, train_img_list, transform = data_transform, random_label=False)
        val_dataset = CatDogDataset3D(train_dir, val_img_list, transform = val_transform, random_label=False)
        num_classes = 2
    else:
        raise ValueError(f'task name not supported {task_name}, type:{type(task_name)}')
    print(f'{task_name} dataset ready')
    return train_dataset, val_dataset, num_classes


def _get_paths(task_name:str=''):
    if task_name == 'catsdogs3d':
        im_path = './output/im/catsdogs3d'
        output_path = './output/cam/catsdogs3d'
    else:
        raise ValueError(f'task name not supported {task_name}, type:{type(task_name)}')
    if not os.path.exists(im_path):
            os.makedirs(im_path)
    if not os.path.exists(output_path):
            os.makedirs(output_path)
    return im_path, output_path


def get_data_weight_output_path(task_name:str=''):
    model, target_layer, out_channel = _get_model(task_name=task_name)
    weight = _get_weight(task_name=task_name)
    model.load_state_dict(weight)
    train_dataset, val_dataset, num_classes = _get_dataset(task_name=task_name)
    im_dir, output_dir = _get_paths(task_name=task_name)
    return model, train_dataset, val_dataset, im_dir, output_dir, target_layer, out_channel, num_classes # weight-loaded model, dataset, im_path, output_path