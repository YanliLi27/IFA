from typing import Union
# from cam_components.agent_main import CAMAgent
from cam_components.camagent import CAMAgent
from torch.utils.data import DataLoader
import os


def naturalimage_runner(target_category:Union[None, int, str]=None, model_flag:str='resnet',
                        task:str='CatsDogs', dataset_split:str='val',
                        max_iter=None, randomization=False, random_severity=0,
                        eval_flag:str='basic', tan_flag:bool=False,
                        cam_method:Union[list, None]=None,
                        cam_save:bool=True):
    # -------------------------------- optional: -------------------------------- #
    batch_size:int=16
    target_category:Union[None, int, str]=target_category  # info of the running process
    # more functions
    im_selection_extra:float=0.3 if task in ['Imagenet', 'MNIST']  else 0.05  # importance matrices attributes
    max_iter=max_iter  # early stop
    groups:int=1
    ram:bool=False  # if it's a regression task
    use_pred:bool=False
    # -------------------------------- optional end -------------------------------- #
    assert task in ['CatsDogs', 'MNIST', 'Imagenet']
    assert model_flag in ['vgg', 'resnet', 'scratch', 'scratch_mnist']


    # information needed:
    from predefined.natural_components.main_generator import main_generator
    model, target_layer, dataset, im_dir, cam_dir, num_out_channel, num_classes = \
                                                    main_generator(model_flag=model_flag,
                                                                    task=task,
                                                                    dataset_split=dataset_split,
                                                                    fold_order=0, 
                                                                    randomization=randomization, random_severity=random_severity
                                                                    )
    dataset = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    
    use_origin = False if task=='MNIST' else True
    fold_order:int=0

    # -------------------------------- start loop -------------------------------- #
    if cam_method==None:
        cam_method_zoo = ['fullcam', 'gradcam', 'gradcampp', 'xgradcam']
    else:
        cam_method_zoo = cam_method
    # maxmin_flag_zoo = [True, False]  # intensity scaling
    # remove_minus_flag_zoo = [False, True]  # remove the part below zero, default: True in the original Grad CAM
    mm_rm_zoo =  [[True, False], [False, True],]
    im_selection_mode_zoo = ['all', 'diff_top']#, 'diff_top', 'reverse_diff_top']  # use feature selection or not -- relied on the importance matrices

    for method in cam_method_zoo:
        for um in mm_rm_zoo:
            mm, rm = um
            if mm:
                mm = 'tanh' if tan_flag else 'norm'
            else:
                mm = False
            for im in im_selection_mode_zoo:
                Agent = CAMAgent(model, target_layer, dataset,  
                                groups, ram,
                                # optional:
                                cam_method=method, name_str=f'{task}_{fold_order}',# cam method and im paths and cam output
                                batch_size=batch_size, select_category=target_category,  # info of the running process
                                rescale=mm,  remove_minus_flag=rm, scale_ratio=1,
                                feature_selection=im, feature_selection_ratio=im_selection_extra,  # feature selection
                                randomization=None,  # model randomization for sanity check
                                use_pred=use_pred,
                                rescaler=None,  # outer scaler
                                cam_type=None  # output 2D or 3D
                                )
                Agent.creator_main(cr_dataset=None, creator_target_category=target_category, eval_act=eval_flag, cam_save=cam_save,
                                   cluster=None, use_origin=use_origin, max_iter=max_iter)


def catsdog3d_runner(target_category:Union[None, int, str]=1, task:str='catsdogs3d', dataset_split:str='val'):    
    # -------------------------------- optional: -------------------------------- #
    batch_size:int=16
    target_category:Union[None, int, str]=target_category  # info of the running process
    # more functions
    im_selection_extra:float=0.05  # importance matrices attributes
    max_iter=None  # early stop
    groups:int=1
    ram:bool=False  # if it's a regression task
    use_pred:bool=False
    # -------------------------------- optional end -------------------------------- #


    # information needed:
    from predefined.multi_components.generators.catsdogs3d_generator import get_data_weight_output_path
    model, train_dataset, val_dataset, im_dir, cam_dir, target_layer, num_out_channel, num_classes =\
              get_data_weight_output_path(task_name=task)
    if dataset_split == 'val':
        dataset = val_dataset
    else:
        dataset = train_dataset
    dataset = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False,
                    num_workers=4, pin_memory=True)
    

    use_origin = False if task=='MNIST' else True
    fold_order:int=0

    # -------------------------------- start loop -------------------------------- #
    cam_method_zoo = ['fullcam', 'gradcam', 'gradcampp', 'xgradcam']
    maxmin_flag_zoo = ['norm', None]  # intensity scaling
    remove_minus_flag_zoo = [False, True]  # remove the part below zero, default: True in the original Grad CAM
    im_selection_mode_zoo = ['all', 'diff_top']  # use feature selection or not -- relied on the importance matrices

    for method in cam_method_zoo:
        for im in im_selection_mode_zoo:
            for mm in maxmin_flag_zoo:
                for rm in remove_minus_flag_zoo:
                    Agent = CAMAgent(model, target_layer, dataset,  
                                groups, ram,
                                # optional:
                                cam_method=method, name_str=f'{task}_{fold_order}',# cam method and im paths and cam output
                                batch_size=batch_size, select_category=target_category,  # info of the running process
                                rescale=mm,  remove_minus_flag=rm, scale_ratio=1.0,
                                feature_selection=im, feature_selection_ratio=im_selection_extra,  # feature selection
                                randomization=None,  # model randomization for sanity check
                                use_pred=use_pred,
                                rescaler=None,  # outer scaler
                                cam_type=None  # output 2D or 3D
                                )
                    Agent.creator_main(cr_dataset=None, creator_target_category='Default', eval_act='corr', cam_save=True,
                                   cluster=None, use_origin=use_origin, max_iter=max_iter)
    

def medical_runner(target_category:Union[None, int, str]=1, task:str='luna', dataset_split:str='val', cam_save:bool=True,
                   eval_flag:str='basic'):
    # -------------------------------- optional: -------------------------------- #
    batch_size:int=16
    target_category:Union[None, int, str]=target_category  # info of the running process
    # more functions
    im_selection_extra:float=0.2 if task=='siim' else 0.05  # importance matrices attributes
    max_iter=None  # early stop
    groups:int=1  # no group convolution here
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
        cam_method_zoo = ['fullcam', 'gradcam', 'gradcampp', 'xgradcam']
        # maxmin_flag_zoo = [True, False]  # intensity scaling
        # remove_minus_flag_zoo = [False, True]  # remove the part below zero, default: True in the original Grad CAM
        mm_rm_zoo =  [['norm', False], [None, True]]
        im_selection_mode_zoo = ['all', 'diff_top']  # use feature selection or not -- relied on the importance matrices

        for method in cam_method_zoo:
            for im in im_selection_mode_zoo:
                for um in mm_rm_zoo:
                    mm, rm = um
                    Agent = CAMAgent(model, target_layer, dataset,  
                                groups, ram,
                                # optional:
                                cam_method=method, name_str=f'{task}_{fold_order}',# cam method and im paths and cam output
                                batch_size=batch_size, select_category=target_category,  # info of the running process
                                rescale=mm,  remove_minus_flag=rm, scale_ratio=1.5,
                                feature_selection=im, feature_selection_ratio=im_selection_extra,  # feature selection
                                randomization=None,  # model randomization for sanity check
                                use_pred=use_pred,
                                rescaler=None,  # outer scaler
                                cam_type=None  # output 2D or 3D
                                )
                    Agent.creator_main(cr_dataset=None, creator_target_category='Default', eval_act=eval_flag, cam_save=cam_save,
                                   cluster=None, use_origin=True, max_iter=max_iter)


def esmira_runner(target_category:Union[None, int, str]=1, data_dir:str='D:\\ESMIRA\\ESMIRA_common',
                target_catename:list=['EAC','ATL'], target_site:list=['Wrist'], target_dirc:list=['TRA', 'COR'],
                cam_save:bool=True, eval_flag:str='basic'):
    # -------------------------------- optional: -------------------------------- #
    batch_size:int=5
    target_category:Union[None, int, str]=target_category  # info of the running process
    # more functions
    im_selection_extra:float=0.05  # importance matrices attributes
    max_iter=None  # early stop
    groups:int=len(target_dirc) * len(target_site)
    ram:bool=False  # if it's a regression task
    use_pred:bool=False
    # -------------------------------- optional end -------------------------------- #

    # information needed:
    from predefined.esmira_components.generators.dataset_class import ESMIRA_generator
    from predefined.esmira_components.model import ModelClass
    from predefined.esmira_components.weight_path import output_finder
    import torch

    dataset_generator = ESMIRA_generator(data_dir, target_catename, target_site, target_dirc)
    for fold_order in range(2,3):
        _, target_dataset = dataset_generator.returner(phase='train', fold_order=fold_order, mean_std=False)
        dataset = DataLoader(dataset=target_dataset, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True)
        # input: [N*5, 512, 512] + int(label)

        # Step. 2 get the model: (can be any nn.Module, make sure it fit your input size and output size)
        
        # --------------------------------------- model --------------------------------------- #
        in_channel = groups * 5
        model = ModelClass(in_channel, num_classes=2)
        weight_path = output_finder(target_catename, target_site, target_dirc, fold_order)
        weight_abs_path = os.path.join('D:\\ESMIRAcode\\RA_Class\\models\\weights\\modelclass_save', weight_path)
        if os.path.isfile(weight_abs_path):
            checkpoint = torch.load(weight_abs_path)
            model.load_state_dict(checkpoint)
        else:
            raise ValueError('weights not exisst')
        target_layer = [model.encoder_class.Conv4]
        # --------------------------------------- model --------------------------------------- #

        # -------------------------------- start loop -------------------------------- #
        cam_method_zoo = ['gradcam', 'fullcam', 'gradcampp', 'xgradcam']
        # maxmin_flag_zoo = [True, False]  # intensity scaling
        # remove_minus_flag_zoo = [False, True]  # remove the part below zero, default: True in the original Grad CAM
        mm_rm_zoo =  [['tanh', False], [None, True]]
        im_selection_mode_zoo = ['all', 'diff_top']  # use feature selection or not -- relied on the importance matrices

        for method in cam_method_zoo:
            for im in im_selection_mode_zoo:
                for um in mm_rm_zoo:
                    mm, rm = um
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
                    Agent.creator_main(cr_dataset=None, creator_target_category='Default', eval_act=eval_flag, cam_save=cam_save,
                                   cluster=None, use_origin=True, max_iter=max_iter)


def ramris_pred_runner(data_dir='', target_category=['EAC'], 
                 target_site=['Wrist'], target_dirc=['TRA', 'COR'],
                 target_biomarker=['SYN'],
                 target_reader=['Reader1', 'Reader2'], task_mode='clip', phase='train',
                 full_img:bool=True):
    # -------------------------------- optional: -------------------------------- #
    batch_size:int=5
    target_category:Union[None, int, str]=target_category  # info of the running process
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
    dataset_generator = ESMIRA_generator(data_dir, target_category, target_site, target_dirc, target_reader, target_biomarker, task_mode)

    for fold_order in range(1, 5):
        _, val_dataset = dataset_generator.returner(task_mode=task_mode, phase=phase, fold_order=fold_order,
                                                                material='img', monai=True, full_img=full_img)
        dataset = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True)
        # input: [N*5, 512, 512] + int(label)

        # --------------------------------------- model --------------------------------------- #
        if full_img:
            depth = 20
        else:
            depth = 5
        in_channel = len(target_site) * len(target_dirc) * depth   # input is a (5*site*dirc) * 512 * 512 img
        out_ch = 0
        output_matrix = [[15, 15, 3, 10],[8, 8, 4, 8],[10, 10, 5, 10]]
        site_order = {'Wrist':0, 'MCP':1, 'Foot':2}
        bio_order = {'ERO':0, 'BME':1, 'SYN':2, 'TSY':3}
        for site in target_site:
            if target_biomarker is None:
                target_biomarker = ['ERO', 'BME', 'SYN', 'TSY']
            for biomarker in target_biomarker:
                out_ch += output_matrix[site_order[site]][bio_order[biomarker]]
        model = ModelClip(in_channel, out_ch=out_ch, dimension=2, group_cap=depth) 

        weight_path = output_finder(target_category, target_site, target_dirc, fold_order) 
        weight_abs_path = os.path.join('D:\\ESMIRAcode\\RA_CLIP\\models\\weights', weight_path)
        if os.path.isfile():
            checkpoint = torch.load(weight_abs_path)
            model.load_state_dict(checkpoint)
        else:
            raise ValueError('weights not exisst')
 
        target_layer = [model.encoder_class.Conv4]
        # --------------------------------------- model --------------------------------------- #

        # -------------------------------- start loop -------------------------------- #
        cam_method_zoo = ['gradcam', 'fullcam', 'gradcampp', 'xgradcam']
        maxmin_flag_zoo = ['tanh', None]  # intensity scaling
        remove_minus_flag_zoo = [False, True]  # remove the part below zero, default: True in the original Grad CAM
        im_selection_mode_zoo = ['all', 'diff_top']  # use feature selection or not -- relied on the importance matrices

        for method in cam_method_zoo:
            for im in im_selection_mode_zoo:
                for mm in maxmin_flag_zoo:
                    for rm in remove_minus_flag_zoo:
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
                        Agent.creator_main(cr_dataset=None, creator_target_category='Default', eval_act='corr', cam_save=True,
                                   cluster=None, use_origin=True, max_iter=max_iter)




