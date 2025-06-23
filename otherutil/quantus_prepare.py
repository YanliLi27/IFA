from ..predefined.natural_components.main_generator import main_generator
from ..predefined.multi_components.generators.multi_task_generators import get_data_weight_output_path

from ..predefined.esmira_components.generators.dataset_class import ESMIRA_generator
from ..predefined.esmira_components.model import ModelClass
from ..predefined.esmira_components.weight_path import output_finder
import os
import torch

    
def imagenet_generator():
    model, target_layer, dataset, im_dir, cam_dir, num_out_channel, num_classes = \
                                                    main_generator(model_flag='vgg',
                                                                    task='Imagenet',
                                                                    dataset_split='val',
                                                                    fold_order=0, 
                                                                    randomization=False, random_severity=0
                                                                    )
    # return model, target_layer, dataset, groups, ram, cam_type
    return model, target_layer, dataset, 1, False, '2D'


def catsdogs_generator():
    model, target_layer, dataset, im_dir, cam_dir, num_out_channel, num_classes = \
                                                    main_generator(model_flag='vgg',
                                                                    task='CatsDogs',
                                                                    dataset_split='val',
                                                                    fold_order=0, 
                                                                    randomization=False, random_severity=0
                                                                    )
    # return model, target_layer, dataset, groups, ram, cam_type
    return model, target_layer, dataset, 1, False, '2D'


def mnist_generator():
    model, target_layer, dataset, im_dir, cam_dir, num_out_channel, num_classes = \
                                                    main_generator(model_flag='scratch_mnist',
                                                                    task='MNIST',
                                                                    dataset_split='val',
                                                                    fold_order=0, 
                                                                    randomization=False, random_severity=0
                                                                    )
    # return model, target_layer, dataset, groups, ram, cam_type
    return model, target_layer, dataset, 1, False, '2D'


def luna_generator():
    model, train_dataset, val_dataset, im_dir, cam_dir, target_layer, num_out_channel, num_classes =\
              get_data_weight_output_path(task_name='luna')
    
    return model, target_layer, train_dataset, 1, False, '2D'


def rsna_generator():
    model, train_dataset, val_dataset, im_dir, cam_dir, target_layer, num_out_channel, num_classes =\
              get_data_weight_output_path(task_name='rsna')
    
    return model, target_layer, train_dataset, 1, False, '2D'


def siim_generator():
    model, train_dataset, val_dataset, im_dir, cam_dir, target_layer, num_out_channel, num_classes =\
              get_data_weight_output_path(task_name='siim')
    
    return model, target_layer, train_dataset, 1, False, '2D'


def us_generator():
    model, train_dataset, val_dataset, im_dir, cam_dir, target_layer, num_out_channel, num_classes =\
              get_data_weight_output_path(task_name='us')
    
    return model, target_layer, train_dataset, 1, False, '2D'


def ddsm_generator():
    model, train_dataset, val_dataset, im_dir, cam_dir, target_layer, num_out_channel, num_classes =\
              get_data_weight_output_path(task_name='ddsm')
    
    return model, target_layer, train_dataset, 1, False, '2D'


def esmira_generator():
    dataset_generator = ESMIRA_generator('D:\\ESMIRA\\ESMIRA_common', ['EAC','ATL'], ['Wrist'], ['TRA', 'COR'])
    _, target_dataset = dataset_generator.returner(phase='train', fold_order=2, mean_std=False)

    in_channel = 2 * 5
    model = ModelClass(in_channel, num_classes=2)
    weight_path = output_finder(['EAC','ATL'], ['Wrist'], ['TRA', 'COR'], 2)
    weight_abs_path = os.path.join('D:\\ESMIRAcode\\RA_Class\\models\\weights\\modelclass_save', weight_path)
    if os.path.isfile(weight_abs_path):
        checkpoint = torch.load(weight_abs_path)
        model.load_state_dict(checkpoint)
    else:
        raise ValueError('weights not exisst')
    target_layer = [model.encoder_class.Conv4]

    return model, target_layer, target_dataset, 2, False, '2D'



