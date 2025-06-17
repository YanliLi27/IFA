from quantus import FaithfulnessCorrelation, AOPC, SensitivityN, Robustness, MaxSensitivity, Continuity, Sparsity, Stability
from typing import Literal 
from ..cam_components import CAMAgent


def saliency_callable(model, inputs, targets, **kwargs):
    return explaination(model, inputs, targets)








class WorkSpace:
    def __init__(self, method:Literal['gradcam', 'fullcam', 'gradcampp', 'xgradcam'],
                       task:Literal['Imagenet'],
                       ):
        # initialize agent, dataset, model at here
        # call different quantus_calculation for different purposes of evaluation
        model, target_layer, dataset,groups, ram = task_generator()
        self.name_str = f'{task}_{method}'

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
        
        self.build_callable_saliency(Agent)
        
        self.score_to_be_calculate:list[str] = []
        self.score_dict:dict[str:float] = {}


    def quantus_calculation(self, metric_name:Literal['faith', 'sensitivity'], name_str:str=''):
    
        metric = FaithfulnessCorrelation()
        scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, device=device, explain_func=explain_func)

        self.score_dict[metric_name] = scores

        if metric_name==self.score_to_be_calculate[-1]:
            self.summary()


    
    def summary(self):
        pass


