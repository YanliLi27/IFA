import os


def weights_path_creator(model_flag, task):
    '''
    can be replaced with any reasonable path creator
    '''
    if task=='CatsDogs':
        model_dir = f"D:\\CatsDogs\\kaggle\\working"
    elif task=='MNIST':
        model_dir = f"D:\\CatsDogs\\MNIST"
    elif task=='Imagenet':
        model_dir = f"D:\\CatsDogs\\Imagenet"
    elif task=='ClickMe':
        model_dir = f"D:\\CatsDogs\\ClickMe"
    else:
        raise AttributeError('task name not valid')
    resume_path = os.path.join(model_dir, f"{task}_{model_flag}_best_model.model")
    return resume_path


def im_path_creator(model_flag, task, dataset_split):
    im_dir = os.path.join('./output/im/', '{}_{}_{}'.format(task, dataset_split, model_flag))
    if not os.path.exists(im_dir):
        os.makedirs(im_dir)
    return im_dir
    # ./output/im/MNIST_val_resnet/All_fold0_im_cateNone_gradmean.csv


def cam_dir_creator(model_flag, task, dataset_split):
    cam_dir = os.path.join('./output/cam/', '{}_{}_{}'.format(task, dataset_split, model_flag))
    if not os.path.exists(cam_dir):
        os.makedirs(cam_dir)
    return cam_dir

# {stat_maxmin_name}_{remove_minus_name}_{im_selection_mode}{im_selection_extra}


def path_generator(model_flag:str='resnet',
                   task:str='CatsDogs', dataset_split:str='val'):
    weights_path = weights_path_creator(model_flag, task)  # absolute path of the weights file

    im_path = im_path_creator(model_flag, task, dataset_split)

    cam_dir = cam_dir_creator(model_flag, task, dataset_split)

    return weights_path, im_path, cam_dir
