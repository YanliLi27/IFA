from typing import Union
from cam_components.camagent import CAMAgent
from torch.utils.data import DataLoader
import os
from torchsummary import summary


def ramris3d_pred_runner(data_dir='', target_category:Union[None, int, str, list]=None, 
                        target_site=['Wrist'], target_dirc=['TRA', 'COR'],
                        target_biomarker=['SYN'],
                        target_reader=['Reader1', 'Reader2'], task_mode='clip', phase='train',
                        full_img:Union[bool, int]=True, dimension:int=2,
                        target_output:Union[None, int, str, list]=[0],
                        cluster:Union[None, list]=[15, 15, 3, 10],
                        tanh:bool=True,  
                        model_csv:bool=False, extension:int=0, score_sum:bool=False,
                        maxfold:int=5):
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
    from predefined.raclip_components.models.clip_model import ModelClip
    from predefined.raclip_components.models.csv3d import make_csv3dmodel
    from predefined.raclip_components.models.convsharevit import make_csvmodel
    from predefined.raclip_components.generators.utli_generator import ESMIRA_generator
    from predefined.raclip_components.utils.output_finder import output_finder
    import torch

    # TODO prepare for the dataloader for synthetic segmentation
    if target_biomarker:
        for item in target_biomarker:
            assert (item in ['ERO', 'BME', 'SYN', 'TSY'])
    dataset_generator = ESMIRA_generator(data_dir, None, target_category, target_site, target_dirc, 
                                         target_reader, target_biomarker, task_mode, print_flag=True, maxfold=5, score_sum=score_sum)

    for fold_order in range(0, maxfold):
        train_dataset, val_dataset = dataset_generator.returner(task_mode=task_mode, phase=phase, fold_order=fold_order,
                                                                material='img', monai=True, full_img=full_img,
                                                                dimension=dimension, data_balance=False,
                                                                path_flag=False)
        
        dataset = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True)
        # input: [N*5, 512, 512] + int(label)

        # Step. 2 get the model: (can be any nn.Module, make sure it fit your input size and output size)
        if full_img is True:
            depth = 20
        elif isinstance(full_img, int):
            depth = full_img
        else:
            depth = 5
        group_num = len(target_site) * len(target_dirc)   # input is a (5*site*dirc) * 512 * 512 img
        out_ch = 0
        if score_sum:
            out_ch = 1
        else:
            output_matrix = [[15, 15, 3, 10],[8, 8, 4, 8],[10, 10, 5, 10]]
            site_order = {'Wrist':0, 'MCP':1, 'Foot':2}
            bio_order = {'ERO':0, 'BME':1, 'SYN':2, 'TSY':3}
            for site in target_site:
                if target_biomarker is None:
                    target_biomarker = ['ERO', 'BME', 'SYN', 'TSY']
                for biomarker in target_biomarker:
                    out_ch += output_matrix[site_order[site]][bio_order[biomarker]]

        if dimension==2:
            if model_csv:
                model = make_csvmodel(img_2dsize=(512, 512), inch=depth*group_num, num_classes=2, num_features=out_ch, extension=extension, 
                  groups=group_num, width=1, dsconv=False, attn_type='normal', patch_size=(2,2), mode_feature=True, dropout=False, init=False)
                target_layer = [model.features[-1]]  #  [model.features[-1]] 
            else:
                model = ModelClip(group_num=group_num, group_cap=depth, out_ch=out_ch, width=2, dimension=dimension) 
                target_layer = [model.encoder_class.Conv4]
            batch_size = 6
            # summary(model, (depth*group_num, 512, 512))
        elif dimension==3: 
            if model_csv:
                model = make_csv3dmodel(img_2dsize=(depth, 512, 512), inch=group_num, num_classes=2, num_features=out_ch, extension=extension, 
                    groups=(len(target_site) * len(target_dirc)), width=2, dsconv=False, attn_type='normal', patch_size=(2,2), 
                    mode_feature=True, dropout=False, init=False)
                target_layer = [model.features[-2]]
                # model = make_csvmodel(num_features=out_ch, mode_feature=True)
            else:
                model = ModelClip(group_num=group_num, group_cap=depth, out_ch=out_ch, width=2, dimension=dimension) 
                target_layer = [model.encoder_class.Conv4]
            batch_size = 8//group_num if group_num<=2 else 2
            summary(model, (group_num, depth, 512, 512))


        weight_path = output_finder(target_biomarker, target_site, target_dirc, None, fold_order, sumscore=score_sum)
        weight_path = weight_path.replace('./models/weights/', '')
        mid_path = 'un22_csv3d_sumscore_splitsites_20240514'
        weight_abs_path = os.path.join(f'D:\\ESMIRAcode\\RA_CLIP\\models\\weights\\{mid_path}', weight_path)
        if os.path.isfile(weight_abs_path):
            checkpoint = torch.load(weight_abs_path)
            model.load_state_dict(checkpoint)
        else:
            raise ValueError(f'weights not exist: {weight_abs_path}')

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
                                cam_type='3D'  # output 2D or 3D
                                )
                        Agent.creator_main(cr_dataset=None, creator_target_category=target_output, eval_act='corr', cam_save=True,
                                    cluster=cluster, use_origin=False, max_iter=max_iter)