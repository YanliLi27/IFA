from predefined.natural_components.datasets.dataset_generator import dataset_generator
from predefined.natural_components.models.model_generator import model_generator
from predefined.natural_components.path_creator.path_generator import path_generator


def main_generator(model_flag:str='resnet',
                   task:str='CatsDogs', dataset_split:str='val',
                   fold_order:int=0, randomization:bool=False, random_severity:int=0
                   ):

    # use path_generator to create the path and flags for dataset and model
    weights_path, im_path, cam_dir = path_generator(model_flag, task, dataset_split)

    # import the dataloader for different inputs -------------------------------------------#
    target_dataset, in_channel, out_channel, num_classes = dataset_generator(model_flag=model_flag, task=task, dataset_split=dataset_split,
                                                                             pre_root=None)

    # set up the model and find target_layers
    model, target_layer = model_generator(model_flag, weights_path, task, randomization, random_severity, in_channel, num_classes, layer_preset=True)

    return model, target_layer, target_dataset, im_path, cam_dir, out_channel, num_classes
