import os
import torchvision
from torch import nn
from predefined.natural_components.models.scratch_model import scratch_mnist, scratch_nn
from predefined.natural_components.models.utils.find_layer import locate_candidate_layer
import torch


def model_zoo(model_flag:str='vgg', in_channel:int=3, num_classes:int=2, task_flag:str='CatsDogs'):
    if model_flag=='vgg':
        if task_flag=='Imagenet' or task_flag=='ClickMe':
            model_vgg = torchvision.models.vgg16(pretrained=True)
        else:
            model_vgg = torchvision.models.vgg11(pretrained=True)
            fc_features = model_vgg.classifier[6].in_features
            model_vgg.classifier[6] = nn.Linear(fc_features, num_classes)
        model = model_vgg
    elif model_flag=='resnet':
        if task_flag=='Imagenet':
            model_res = torchvision.models.resnet18(pretrained=True)
        else:
            model_res = torchvision.models.resnet18(pretrained=True)
            fc_features = model_res.fc.in_features
            model_res.fc = nn.Linear(fc_features, num_classes)
        model = model_res
    elif model_flag=='scratch_mnist':
        if task_flag=='Imagenet':
            raise AttributeError('not valid combination of task and model')
        model = scratch_mnist(in_channel=in_channel, num_classes=num_classes)
    else:
        if task_flag=='Imagenet':
            raise AttributeError('not valid combination of task and model')
        model = scratch_nn(in_channel=in_channel, num_classes=num_classes)
    return model, model_flag


def model_generator(model_flag, weights_path, task, randomization, random_severity, in_channel, num_classes, layer_preset=True):
    # model setting ------------------------------------------------------------------------#
    model, model_flag = model_zoo(model_flag=model_flag, in_channel=in_channel, num_classes=num_classes, task_flag=task)
    
    # load the weights
    if os.path.isfile(weights_path):
        checkpoint = torch.load(weights_path)
        model.load_state_dict(checkpoint)
        print('model_load_successfully')
    else:
        raise ValueError(f'Not loading weights {weights_path}')

    # model randomization -----------------------------------------------------------------#
    if randomization:
        from predefined.natural_components.models.utils.model_destroyer import model_destroyer
        model = model_destroyer(model, model_flag, random_severity)

    # Target layer ------------------------------------------------------------------------#
    if model_flag=='vgg':
        if layer_preset:
            target_layers = [model.features[-1]]
        else:
            target_layers = locate_candidate_layer(model, [3, 224, 224])
    elif model_flag=='resnet':
        if layer_preset:
            target_layers = [model.layer4[-1]]
        else:
            target_layers = locate_candidate_layer(model, [3, 224, 224])
    elif model_flag=='scratch' or model_flag=='scratch_mnist':
        try:
            target_layers = [model.conv3]
        except:
            target_layers = [model.conv2]
    else:
        raise ValueError(f'not valid model flag {model_flag}')

    return model, target_layers