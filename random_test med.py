from typing import Union
# from cam_components.agent_main import CAMAgent
from cam_components.camagent import CAMAgent
from torch.utils.data import DataLoader
import os


def medical_runner(target_category:Union[None, int, str]=1, task:str='luna', dataset_split:str='val', max_iter=None, 
                   cam_save:bool=True,
                   eval_flag:str='basic', tan_flag:bool=False, cam_method=None, fold_idx:int=7):
    # -------------------------------- optional: -------------------------------- #
    batch_size:int=1
    target_category:Union[None, int, str]=target_category  # info of the running process
    # more functions
    im_selection_extra:float=0.3 if task in ['Imagenet', 'MNIST']  else 0.1  # importance matrices attributes
    groups:int=1
    ram:bool=False  # if it's a regression task
    use_pred:bool=False
    # -------------------------------- optional end -------------------------------- #
    assert task in ['luna', 'rsna', 'siim', 'us', 'ddsm' ]

    # information needed:
    from predefined.multi_components.generators.multi_task_generators import get_data_weight_output_path
    for i in range(0, 1):
        fold_order = i
        model, train_dataset, val_dataset, im_dir, cam_dir, target_layer, num_out_channel, num_classes =\
              get_data_weight_output_path(task_name=task)
        if dataset_split == 'val':
            dataset = val_dataset
        else:
            dataset = train_dataset
        dataset = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)

    # -------------------------------- start loop -------------------------------- #
    if cam_method==None:
        cam_method_zoo = ['fullcam', 'gradcam', 'gradcampp', 'xgradcam']
    else:
        cam_method_zoo = ['fullcam']
    # maxmin_flag_zoo = [True, False]  # intensity scaling
    # remove_minus_flag_zoo = [False, True]  # remove the part below zero, default: True in the original Grad CAM
    mm_rm_zoo =  [[False, True]] # [True, False], 
    # im_selection_mode_zoo = ['all', 'diff_top']#, 'diff_top', 'reverse_diff_top']  # use feature selection or not -- relied on the importance matrices

    for method in cam_method_zoo:
        for um in mm_rm_zoo:
            mm, rm = um
            if mm:
                mm = 'tanh' if tan_flag else 'norm'
            else:
                mm = False

            Agent = CAMAgent(model, target_layer, dataset,  
                            groups, ram,
                            # optional:
                            cam_method=method, name_str=f'{task}_{fold_idx}',# cam method and im paths and cam output
                            batch_size=batch_size, select_category=target_category,  # info of the running process
                            rescale=mm,  remove_minus_flag=rm, scale_ratio=1,
                            feature_selection='all', feature_selection_ratio=im_selection_extra,  # feature selection
                            randomization=None,  # model randomization for sanity check
                            use_pred=use_pred,
                            rescaler=None,  # outer scaler
                            cam_type=None  # output 2D or 3D
                            )
            Agent.creator_main(cr_dataset=None, creator_target_category=target_category, eval_act=eval_flag, cam_save=cam_save,
                                cluster=None, use_origin=True, max_iter=max_iter, random_im_mask=0.05)
                


if __name__ == '__main__':
    # for natural images test 
    task_zoo = ['luna', 'rsna', 'siim', 'us', 'ddsm' ]
    for task in task_zoo:
        for i in range(10):
            medical_runner(target_category=0, task=task, dataset_split='val',
                            max_iter=20,
                            cam_save=True, eval_flag=False, tan_flag=False, cam_method=['fullcam'], fold_idx=i)
