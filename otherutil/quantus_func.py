from quantus import FaithfulnessCorrelation, SensitivityN, MaxSensitivity, Continuity, Selectivity, RelativeInputStability
from typing import Any, Literal 
import torch
import quantus
from torch.utils.data import DataLoader, Dataset
from quantus_metric import obtain_metrics_results

from ..cam_components import CAMAgent


class WorkSpace:
    def __init__(self, task:Literal['Imagenet'],
                       method:Literal['gradcam', 'fullcam', 'gradcampp', 'xgradcam'],
                       
                       ):
        # call different quantus_calculation for different purposes of evaluation
        self.task = task
        self.method = method
        # ------------------------------- model, dataset initialization ------------------------------- #
        model, target_layer, dataset, groups, ram, cam_type = self._task_generator(task, method)
        self.name_str = f'{task}_{method}'  # for metric need another name


        # ------------------------------- explaination agent initialization ------------------------------- # 
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
        
        self.model, self.dataset = model, dataset

        self.score_to_be_calculate:list[str] = []
        self.score_dict:dict = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    def _task_generator(self, task:Literal['Imagenet'],
                       method:Literal['gradcam', 'fullcam', 'gradcampp', 'xgradcam']):
        return model, target_layer, dataset,groups, ram, cam_type


    def _explain_func(self, x:torch.Tensor):
        # [batch, channel, (D), L, W]
        cam = self.agent.indiv_return(x, pred_flag=False) # make sure the input is a 4-dimension tensor [batch, channel, W, H]
        # [batch, group(1), cluster/tc, (D), L, W]
        cam = cam.squeeze(axis=1).squeeze(axis=2)
        if len(cam.shape)<len(x.shape):
            cam = torch.unsqueeze(cam, dim=1)  # [batch, (D), L, W] -> [batch, 1, (D), L, W] 
        cam = cam.numpy() if cam.device=='cpu' else cam.cpu().numpy()
        return cam  # [batch, channel, (D), L, W] -> [batch, 1, (D), L, W] 


    def quantus_calculation(self, metric_name:Literal['faith', 'sensitivity'], name_str:str=''):
        # get dataloader
        dataloader = DataLoader(self.dataset, batch_size=16, shuffle=True)
        # metric setting
        metric = FaithfulnessCorrelation(model=model,
                                        explain_func=explain_func,  # 你的 Grad-CAM 或其他解释方法
                                        task="classification",
                                        device="cuda",
                                        return_instance_score=True)
        temp_score = []
        with torch.no_grad():
            for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
                print(f"Batch {batch_idx + 1}/{len(dataloader)}")
                # x_batch torch.Tensor [b, c, l, w]
                cam_batch = self._explain_func(x_batch)  # [b, 1, l, w]

                scores = metric(model=self.model, x_batch=x_batch, y_batch=y_batch, a_batch=cam_batch, device=self.device, 
                                explain_func=quantus.explain, explain_func_kwargs={"method": "Saliency"},)
                temp_score.extend(scores)

        assert len(temp_score)>0
        self.score_dict[metric_name] = sum(temp_score)/len(temp_score)

        if metric_name==self.score_to_be_calculate[-1]:
            self.summary()

    
    def summary(self):
        pass


