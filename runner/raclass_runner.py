from typing import Union, Literal
from cam_components.camagent import CAMAgent
from torch.utils.data import DataLoader
import os
from torchsummary import summary


def raclass_pred_runner(data_dir='', target_category:Union[None, int, str, list]=['CSA'], 
                        target_site=['Wrist'], target_dirc=['TRA', 'COR'], phase='train',
                        model_counter='mobilevit', attn_type:Literal['normal', 'mobile', 'parr_normal', 'parr_mobile']='normal',
                        full_img:Union[bool, int]=5,
                        maxfold:int=5, tanh:bool=True):
    # -------------------------------- optional: -------------------------------- #
    batch_size:int=1
    target_category:Union[None, int, str, list]=target_category  # info of the running process
    # more functions
    im_selection_extra:float=0.05  # importance matrices attributes
    max_iter=None  # early stop
    groups:int=len(target_dirc) * len(target_site)
    ram:bool=True  # if it's a regression task
    use_pred:bool=False
    # -------------------------------- optional end -------------------------------- #

    # information needed:
    from predefined.raclass_components.models.model import ModelClass, Classifier11
    from torchvision.models import MobileNetV2
    from predefined.raclass_components.models.vit import ViT
    from predefined.raclass_components.models.model3d import ModelClass3D
    from predefined.raclass_components.models.mobilevit import mobilevit_s, mobilevit_xs, mobilevit_xxs
    from predefined.raclass_components.models.csv3d import make_csv3dmodel
    from predefined.raclass_components.models.convsharevit import make_csvmodel
    from predefined.raclass_components.generators.dataset_class import ESMIRA_generator
    from predefined.raclass_components.utils.output_finder import output_finder
    import torch

    # TODO prepare for the dataloader for synthetic segmentation
    dataset_generator = ESMIRA_generator(data_dir, target_category, target_site, target_dirc, maxfold=maxfold)
    for fold_order in range(0, maxfold):
        save_task = target_category[0] if len(target_category)==1 else (target_category[0]+'_'+target_category[1])
        save_site = target_site[0] if len(target_site)==1 else (target_site[0]+'_'+target_site[1])
        save_father_dir = os.path.join('./models/figs', f'{model_counter}_{save_site}_{save_task}')
        if not os.path.exists(save_father_dir):
            os.makedirs(save_father_dir)
        save_dir = os.path.join(save_father_dir, f'fold_{fold_order}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        dimenson = '3D' if '3d' in model_counter else '2D'
        train_dataset, val_dataset = dataset_generator.returner(phase=phase, fold_order=fold_order, 
                                                                mean_std=False, full_img=full_img, 
                                                                test_balance=False, dimension=dimenson)
        # input: [N*5, 512, 512] + int(label)
        
        dataset = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True)
        # input: [N*5, 512, 512] + int(label)

        # Step. 2 get the model: (can be any nn.Module, make sure it fit your input size and output size)
        in_channel = len(target_site) * len(target_dirc) * full_img if isinstance(full_img, int) else len(target_site) * len(target_dirc) * 20
        # model = ModelClass(in_channel, num_classes=2)
        if model_counter == 'mobilenet':
            model = MobileNetV2(num_classes=2, inch=in_channel)
            target_layer = [model.features[-1]]
        elif model_counter == 'mobilevit':
            model = mobilevit_s(img_2dsize=(512, 512), inch=in_channel, num_classes=2, patch_size=(4,4))
            target_layer = [model.conv2]
        elif model_counter == 'vit':
            model = ViT(image_size=(512, 512), patch_size=(16, 16), num_classes=2, 
                        dim=256, depth=12, heads=8, mlp_dim=512, pool='mean', channels=in_channel, 
                        dropout=0.2, emb_dropout=0.2)
            target_layer = [model.transformer[-1]]
        elif model_counter == 'modelclass':
            model = ModelClass(in_channel, group_num=len(target_site) * len(target_dirc), num_classes=2)
            target_layer = [model.encoder_class.Conv4]
        elif model_counter == 'modelclass11':
            model = ModelClass(in_channel, group_num=len(target_site) * len(target_dirc), num_classes=2, classifier=Classifier11)
            target_layer = [model.encoder_class.Conv4]
        elif model_counter == 'convsharevit':
            model = make_csvmodel(img_2dsize=(512, 512), inch=in_channel, num_classes=2, num_features=43, extension=57, 
                  groups=(len(target_site) * len(target_dirc)), width=1, dsconv=False, attn_type=attn_type, patch_size=(2,2), 
                  mode_feature=False, dropout=True, init=False)
            target_layer = [model.features[-1]]
        elif model_counter == 'modelclass3d':
            in_ch=len(target_site)*len(target_dirc)
            if in_ch > 2:
                poolsize = 1
            else:
                poolsize = 3
            model = ModelClass3D(in_ch=in_ch, depth=in_channel//in_ch, group_num=len(target_site) * len(target_dirc), 
                                 num_classes=2, poolsize=poolsize)
            target_layer = [model.encoder_class.Conv4]
        elif model_counter == 'csv3d':
            in_ch=len(target_site)*len(target_dirc)
            model = make_csv3dmodel(img_2dsize=(in_channel//in_ch, 512, 512), inch=in_ch, num_classes=2, num_features=43, extension=57, 
                  groups=(len(target_site) * len(target_dirc)), width=1, dsconv=False, attn_type=attn_type, patch_size=(1,2,2), 
                  mode_feature=False, dropout=True, init=False)
            target_layer = [model.features[-2]]
        else:
            raise ValueError('not supported model')


        weight_path = output_finder(model_counter, target_category, target_site, target_dirc, fold_order)
        weight_path = weight_path.replace('./models/weights/', '')
        mid_path = 'un22_csv3d_sumscore_splitsites_20240514'  # TODO
        weight_abs_path = os.path.join(f'D:\\ESMIRAcode\\RA_Class\\models\\weights\\{mid_path}', weight_path)
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
                                batch_size=batch_size, select_category=1,  # info of the running process
                                rescale=mm,  remove_minus_flag=rm, scale_ratio=2,
                                feature_selection=im, feature_selection_ratio=im_selection_extra,  # feature selection
                                randomization=None,  # model randomization for sanity check
                                use_pred=use_pred,
                                rescaler=None,  # outer scaler
                                cam_type='3D'  # output 2D or 3D
                                )
                        Agent.creator_main(cr_dataset=None, creator_target_category=[1], eval_act='corr', cam_save=True,
                                    cluster=None, use_origin=False, max_iter=max_iter)