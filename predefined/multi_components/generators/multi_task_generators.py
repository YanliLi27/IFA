import torch
import torch.nn as nn
import torchvision
import os


def _get_model(task_name:str=''):
    if task_name == 'ddsm':
        model = torchvision.models.resnet34(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        target_layer = [model.layer4[-1]]
        out_channel = 512
    elif task_name == 'luna':
        model = torchvision.models.vgg11(weights=None)#torchvision.models.VGG11_Weights)
        model.classifier._modules['6'] = nn.Linear(4096, 2)
        target_layer = [model.features[-1]]
        # print(model.features)
        out_channel = 512
    elif task_name == 'rsna':
        model = torchvision.models.vgg16(weights=None)
        model.classifier._modules['6'] = nn.Linear(4096, 2)
        target_layer = [model.features[-1]]
        out_channel = 512
    elif task_name == 'us':
        from predefined.multi_components.models.us_model import us_nn
        model = us_nn(1, 2)
        target_layer = [model.cnn[-8]]
        out_channel = 128
    elif task_name == 'siim':
        model = torchvision.models.resnet34(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        target_layer = [model.layer4[-1]]
        out_channel = 512
    else:
        raise ValueError(f'task name not supported {task_name}, type:{type(task_name)}')
    return model, target_layer, out_channel


def _get_weight(task_name:str=''):
    default_dir = 'D:/CatsDogs'
    if task_name == 'ddsm':
        weight_path = f'{default_dir}/models/ddsm/best_model.pth.tar'
    elif task_name == 'luna':
        weight_path = f'{default_dir}/models/luna/best_model.pth.tar'
    elif task_name == 'rsna':
        weight_path = f'{default_dir}/models/rsna/best_model083.pth.tar'
    elif task_name == 'us':
        weight_path = f'{default_dir}/models/usbc/best_model076.pth.tar'
    elif task_name == 'siim':
        weight_path = f'{default_dir}/models/siim/best_model.pth.tar'
    else:
        raise ValueError(f'task name not supported {task_name}, type:{type(task_name)}')
    if os.path.isfile(weight_path):
        weight = torch.load(weight_path)
        print('weights loaded')
    else:
         raise ValueError(f'weight path not exist {weight_path}')
    return weight


def _get_dataset(task_name:str=''):
    if task_name == 'ddsm':
        from predefined.multi_components.dataset.ddsm_dataset import ddsm_initialization, DDSMDataset
        train_list, val_list = ddsm_initialization(train_dir='D:/ImageNet', load_save=True)
        train_dataset = DDSMDataset(train_list)
        val_dataset = DDSMDataset(val_list, val_flag=True)
        num_classes = 2
    elif task_name == 'luna':
        from predefined.multi_components.dataset.luna_dataset import luna_cropped_initialization, LunaDataset
        train_list, val_list = luna_cropped_initialization(train_dir='D:/ImageNet', load_save=True)
        train_dataset = LunaDataset(train_list, repeat=3)
        val_dataset = LunaDataset(val_list, val_flag=True, repeat=3)
        num_classes = 2
    elif task_name == 'rsna':
        from predefined.multi_components.dataset.rsna_dataset import rsna_initialization, RSNADataset
        train_paths, train_labels, val_paths, val_labels = rsna_initialization(train_dir='D:/ImageNet', load_save=True)
        train_dataset = RSNADataset(paths=train_paths, labels=train_labels)
        val_dataset = RSNADataset(paths=val_paths, labels=val_labels, val_flag=True)
        num_classes = 2
    elif task_name == 'us':
        from predefined.multi_components.dataset.us_dataset import us_initialization, USDataset
        train_list, val_list = us_initialization(train_dir='D:/ImageNet', classes=2, load_save=True)
        train_dataset = USDataset(train_list, repeat=1)
        val_dataset = USDataset(val_list, val_flag=True, repeat=1)
        num_classes = 2
    elif task_name == 'siim':
        from predefined.multi_components.dataset.siim_dataset import siim_initialization, SIIMDataset
        train_list, val_list = siim_initialization(train_dir='D:/ImageNet', load_save=True)
        train_dataset = SIIMDataset(train_list)
        val_dataset = SIIMDataset(val_list, val_flag=True)
        num_classes = 2
    else:
        raise ValueError(f'task name not supported {task_name}, type:{type(task_name)}')
    print(f'{task_name} dataset ready')
    return train_dataset, val_dataset, num_classes


def _get_paths(task_name:str=''):
    if task_name == 'ddsm':
        im_path = './output/im/ddsm'
        output_path = './output/cam/ddsm'
    elif task_name == 'luna':
        im_path = './output/im/luna'
        output_path = './output/cam/luna'
    elif task_name == 'rsna':
        im_path = './output/im/rsna'
        output_path = './output/cam/rsna'
    elif task_name == 'us':
        im_path = './output/im/usbc/'
        output_path = './output/cam/usbc'
    elif task_name == 'siim':
        im_path = './output/im/siim/'
        output_path = './output/cam/siim'
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