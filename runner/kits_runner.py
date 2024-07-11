from predefined.kits_components.kits_generator import kits_intialization, KitsDataset
from predefined.raclass_components.models.convsharevit import make_csvmodel
from predefined.raclass_components.models.model import ModelClass, Classifier, Classifier11
from cam_components.camagent import CAMAgent
from torch.utils.data import DataLoader
import os
import torch


def kits_runner(num_samples:int=10000, tanh:bool=True, weight_abs_path=None, model_name:str='csv'):
    trainset, valset = kits_intialization(datacsv=f'D:\\ESMIRAcode\\ACAM\\dataprepare\\kits\\splitdatapath_{num_samples}.npy')
    # traindataset = KitsDataset(stacked_list=trainset, transform=None, val_flag=False, repeat=1, maskout=False)
    valdataset = KitsDataset(stacked_list=valset, transform=None, val_flag=True, repeat=1, maskout=False)
    dataset = DataLoader(dataset=valdataset, batch_size=1, shuffle=False,
            num_workers=4, pin_memory=True)

    weight_abs_path:str = r'D:\ESMIRAcode\ACAM\model\logs\bestmodelkits.model' if weight_abs_path==None else weight_abs_path
    
    if model_name=='csv':
        model = make_csvmodel(img_2dsize=(512, 512), inch=1, num_classes=2, num_features=50, extension=50, 
                    groups=1, width=1, dsconv=False, attn_type='normal', patch_size=(2,2), 
                    mode_feature=False, dropout=True, init=False)
        target_layer = [model.features[-1]]
    else:
        model = ModelClass(img_ch=1, group_num=1, num_classes=2, classifier=Classifier)
        target_layer = [model.encoder_class.Conv4]
    
    if os.path.isfile(weight_abs_path):
            checkpoint = torch.load(weight_abs_path)
            model.load_state_dict(checkpoint)
    else:
        raise ValueError(f'weights not exist: {weight_abs_path}')
    
    # -------------------------------- start loop -------------------------------- #
    cam_method_zoo = ['gradcam', 'fullcam']#, 'fullcam', 'gradcampp', 'xgradcam']
    maxmin_flag_zoo = [True, False]  # intensity scaling
    remove_minus_flag_zoo = [False, True]  # remove the part below zero, default: True in the original Grad CAM
    im_selection_mode_zoo = ['all']#, 'diff_top']  # use feature selection or not -- relied on the importance matrices
    mmrm_zoo = [[True, False], [True, True], [False, True]]

    for method in cam_method_zoo:
        for im in im_selection_mode_zoo:
            for mmrm in mmrm_zoo:
                mm = mmrm[0]
                rm = mmrm[1]
                if (mm and tanh):
                    mm = 'tanh'
                elif mm:
                    mm = 'norm'
                else:
                    mm = False
                Agent = CAMAgent(model, target_layer, dataset,  
                        groups=1, ram=False,
                        # optional:
                        cam_method=method, name_str=f'kits',# cam method and im paths and cam output
                        batch_size=1, select_category=1,  # info of the running process
                        rescale=mm,  remove_minus_flag=rm, scale_ratio=2,
                        feature_selection=im, feature_selection_ratio=0.05,  # feature selection
                        randomization=None,  # model randomization for sanity check
                        use_pred=False,
                        rescaler=None,  # outer scaler
                        cam_type='2D'  # output 2D or 3D
                        )
                Agent.creator_main(cr_dataset=None, creator_target_category=[1], eval_act='corr', cam_save=True,
                            cluster=None, use_origin=False, max_iter=None)
