from typing import Union, Literal
# from cam_components.agent_main import CAMAgent
from cam_components.camagent import CAMAgent
from torch.utils.data import DataLoader
import torch



def indiv_runner(target_category:Union[None, int, str]=None, 
                 model_flag:Literal['vgg', 'resnet', 'scratch', 'scratch_mnist']='resnet',
                 task:Literal['CatsDogs', 'MNIST', 'Imagenet']='CatsDogs', dataset_split:str='val',
                 max_iter=None, randomization=False, random_severity=0,
                 eval_flag:str='basic', tan_flag:bool=False,
                 cam_method:Union[list, None]=None,
                 cam_save:bool=True):
    # -------------------------------- optional: -------------------------------- #
    batch_size:int=16
    target_category:Union[None, int, str]=target_category  # info of the running process
    # more functions
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
    
    fold_order:int=0

    # -------------------------------- start loop -------------------------------- #
    if cam_method==None:
        cam_method_zoo = ['fullcam', 'gradcam', 'gradcampp', 'xgradcam']
    else:
        cam_method_zoo = cam_method

    for method in cam_method_zoo:
        Agent = CAMAgent(model, target_layer, dataset,  
                        groups, ram,
                        # optional:
                        cam_method=method, name_str=f'{task}_{fold_order}',# cam method and im paths and cam output
                        batch_size=batch_size, select_category=target_category,  # info of the running process
                        rescale='norm',  remove_minus_flag=False, scale_ratio=1,
                        feature_selection='all', feature_selection_ratio=1.0,  # feature selection
                        randomization=None,  # model randomization for sanity check
                        use_pred=use_pred,
                        rescaler=None,  # outer scaler
                        cam_type=None  # output 2D or 3D
                        )
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for x,y in dataset:
            x = x.to(dtype=torch.float32).to(device)
            y = y.to(device)

            indiv_cam = Agent.indiv_return(x, target_category, None)
            print(indiv_cam.shape)
            # [batch, groups, cluster/tc, (D), L, W]
            print('1')