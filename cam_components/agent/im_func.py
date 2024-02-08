import pandas as pd
import numpy as np


def im_save(im_overall, im_target, im_diff, 
            cam_grad_max_matrix:np.array, cam_grad_min_matrix:np.array,
            cam_output_path:str
            ):
    # cam_output_path: ./output/im/MNIST_val_resnet/All_im_cateNone_GradCAM.csv
    # im_overall [num_out_channel]
    # im_target/im_diff [num_classes, num_out_channel]
    all_save_name = cam_output_path
    im_save = pd.DataFrame(im_overall)
    im_save.to_csv(all_save_name)
    print(f'save csv file:{all_save_name}, succeed')

    target_save_name = cam_output_path.replace('All_', 'Target_')
    target_im_save = pd.DataFrame(im_target)
    target_im_save.to_csv(target_save_name)
    print(f'save csv file:{target_save_name}, succeed')

    diff_save_name = cam_output_path.replace('All_', 'Diff_')
    diff_im_save = pd.DataFrame(im_diff)
    diff_im_save.to_csv(diff_save_name)
    print(f'save csv file:{diff_save_name}, succeed')

    percentile = np.array([25, 50, 75, 90, 95])
    mm_save_name = cam_output_path.replace('.csv', '_maxmin.csv')
    mm_save = pd.DataFrame({
    'Grad_max_value':cam_grad_max_matrix, 'Grad_min_value':cam_grad_min_matrix,
    'Percentile':percentile
    })
    mm_save.to_csv(mm_save_name)
    print(f'save maxmin csv file:{mm_save_name}, succeed')


def im_reader(IM_save_name:str, cam_mode:str):
    if 'abs' in cam_mode:
        im_save_name = IM_save_name.replace('All_', 'Target_')
    elif 'diff' in cam_mode:
        im_save_name = IM_save_name.replace('All_', 'Diff_')
    elif 'all' in cam_mode:
        return None
    else:
        raise ValueError(f'not save name: {IM_save_name}')
    im = pd.read_csv(im_save_name)
    im = im.values[:, 1:]
    return im  # im [num_classes, num_features]


def maxmin_reader(IM_save_name:str, target_category=None):
    mm_save_name = IM_save_name.replace('.csv', '_maxmin.csv')
    try:
        mm_file = pd.read_csv(mm_save_name)
    except:
        mm_save_name = mm_save_name.replace(f'cate{target_category}', 'cateNone')
        mm_file = pd.read_csv(mm_save_name)
    data_max_value = mm_file['Grad_max_value'][3]
    data_max_value = np.max(data_max_value)
    data_min_value = mm_file['Grad_min_value'][1]
    data_min_value = np.min(data_min_value)
    return data_max_value, data_min_value