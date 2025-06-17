from quantus import FaithfulnessCorrelation, SensitivityN, MaxSensitivity, Continuity, Selectivity, RelativeInputStability
from typing import Any, Literal 
from ..cam_components import CAMAgent


class WorkSpace:
    def __init__(self, method:Literal['gradcam', 'fullcam', 'gradcampp', 'xgradcam'],
                       task:Literal['Imagenet'],
                       ):
        # initialize agent, dataset, model at here
        # call different quantus_calculation for different purposes of evaluation
        model, target_layer, dataset,groups, ram, cam_type = task_generator()
        self.name_str = f'{task}_{method}'  # for metric need another name

        self.agent = CAMAgent(model, target_layer, dataset,  
                                groups, ram,
                                # optional:
                                cam_method=method, name_str=f'{self.name_str}',# cam method and im paths and cam output
                                batch_size=1, select_category=1,  # info of the running process
                                rescale='norm',  remove_minus_flag=False, scale_ratio=2,
                                feature_selection='all', feature_selection_ratio=1.,  # feature selection
                                randomization=None,  # model randomization for sanity check
                                use_pred=False,
                                rescaler=None,  # outer scaler
                                cam_type=cam_type  # output 2D or 3D
                                )
        
        self.agent.indiv_return(x) # make sure the input is a 4-dimension tensor [batch, channel, W, H]
        
        
        self.score_to_be_calculate:list[str] = []
        self.score_dict:dict = {}

    def _build_callable_saliency(self, Agent):
        pass


    def quantus_calculation(self, metric_name:Literal['faith', 'sensitivity'], name_str:str=''):
    
        metric = FaithfulnessCorrelation()
        scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, device=device, explain_func=explain_func)

        self.score_dict[metric_name] = scores

        if metric_name==self.score_to_be_calculate[-1]:
            self.summary()


    
    def summary(self):
        pass


