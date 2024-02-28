from typing import Union
from cam_components.camagent import CAMAgent
from torch.utils.data import DataLoader
import os
from torchsummary import summary


def ramris_pred_runner(data_dir='', target_category:Union[None, int, str, list]=None, 
                 target_site=['Wrist'], target_dirc=['TRA', 'COR'],
                 target_biomarker=['SYN'],
                 target_reader=['Reader1', 'Reader2'], task_mode='clip', phase='train',
                 full_img:bool=True, dimension:int=2,
                 target_output:Union[None, int, str, list]=[0],
                 cluster:Union[None, list]=[15, 15, 3, 10],
                 tanh:bool=True):
    # -------------------------------- optional: -------------------------------- #
    batch_size:int=2
    target_category:Union[None, int, str, list]=target_category  # info of the running process
    # more functions
    im_selection_extra:float=0.05  # importance matrices attributes
    max_iter=None  # early stop
    groups:int=len(target_dirc) * len(target_site)
    ram:bool=True  # if it's a regression task
    use_pred:bool=False
    # -------------------------------- optional end -------------------------------- #

    # information needed:
    from predefined.ramris_components.models.clip_model import ModelClip
    from predefined.ramris_components.generators.utli_generator import ESMIRA_generator
    from predefined.ramris_components.utils.output_finder import output_finder
    import torch

    if target_biomarker:
        for item in target_biomarker:
            assert (item in ['ERO', 'BME', 'SYN', 'TSY'])
    dataset_generator = ESMIRA_generator(data_dir, target_category, target_site, target_dirc, target_reader, 
                                         target_biomarker, task_mode, working_dir='D:\\ESMIRAcode\\RA_CLIP\\')

    for fold_order in range(0, 1):
        _, val_dataset = dataset_generator.returner(task_mode=task_mode, phase=phase, fold_order=fold_order,
                                                                material='img', monai=True, full_img=full_img, dimension=dimension)
        dataset = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True)
        # input: [N*5, 512, 512] + int(label)

        # Step. 2 get the model: (can be any nn.Module, make sure it fit your input size and output size)
        if full_img:
            depth = 20
        else:
            depth = 5
        group_num = len(target_site) * len(target_dirc)   # input is a (5*site*dirc) * 512 * 512 img
        out_ch = 0
        output_matrix = [[15, 15, 3, 10],[8, 8, 4, 8],[10, 10, 5, 10]]
        site_order = {'Wrist':0, 'MCP':1, 'Foot':2}
        bio_order = {'ERO':0, 'BME':1, 'SYN':2, 'TSY':3}
        for site in target_site:
            if target_biomarker is None:
                target_biomarker = ['ERO', 'BME', 'SYN', 'TSY']
            for biomarker in target_biomarker:
                out_ch += output_matrix[site_order[site]][bio_order[biomarker]]
        if dimension==2:
            width = 2
            model = ModelClip(group_num=group_num, group_cap=depth, out_ch=out_ch, width=width, dimension=dimension)  
            summary(model, (40, 512, 512))
        elif dimension==3:
            width = 1
            model = ModelClip(group_num=group_num, group_cap=depth, out_ch=out_ch, width=width, dimension=dimension)  
            summary(model, (2, 20, 512, 512))

        weight_path = output_finder(target_biomarker, target_site, target_dirc, fold_order)
        mid_path = 'ALLBIO' if (target_biomarker is None or len(target_biomarker)>1) else f'ALL{target_biomarker[0]}'
        weight_abs_path = os.path.join(f'D:\\ESMIRAcode\\RA_CLIP\\models\\weights\\{mid_path}', weight_path)
        if os.path.isfile(weight_abs_path):
            checkpoint = torch.load(weight_abs_path)
            model.load_state_dict(checkpoint)
        else:
            raise ValueError(f'weights not exist: {weight_abs_path}')
 
        target_layer = [model.encoder_class.Conv4]
        # --------------------------------------- model --------------------------------------- #

        # -------------------------------- start loop -------------------------------- #
        cam_method_zoo = ['gradcam']#, 'fullcam', 'gradcampp', 'xgradcam']
        maxmin_flag_zoo = [True, False]  # intensity scaling
        remove_minus_flag_zoo = [False, True]  # remove the part below zero, default: True in the original Grad CAM
        im_selection_mode_zoo = ['all']#, 'diff_top']  # use feature selection or not -- relied on the importance matrices

        for method in cam_method_zoo:
            for im in im_selection_mode_zoo:
                for mm in maxmin_flag_zoo:
                    for rm in remove_minus_flag_zoo:
                        if mm and tanh:
                            mm = 'tanh'
                        else:
                            mm = 'norm'
                        Agent = CAMAgent(model, target_layer, dataset,  
                                groups, ram,
                                # optional:
                                cam_method=method, name_str=f'esmira_{fold_order}',# cam method and im paths and cam output
                                batch_size=batch_size, select_category=target_category,  # info of the running process
                                rescale=mm,  remove_minus_flag=rm, scale_ratio=2,
                                feature_selection=im, feature_selection_ratio=im_selection_extra,  # feature selection
                                randomization=None,  # model randomization for sanity check
                                use_pred=use_pred,
                                rescaler=None,  # outer scaler
                                cam_type=None  # output 2D or 3D
                                )
                        Agent.creator_main(cr_dataset=None, creator_target_category=target_output, eval_act='corr', cam_save=True,
                                    cluster=cluster, use_origin=True, max_iter=max_iter)