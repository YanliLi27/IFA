from typing import Literal 
import gc
import os
import numpy as np
import pandas as pd

import torch
import quantus
from torch.utils.data import DataLoader
from otherutil.quantus_metric import obtain_metrics_results

from otherutil.quantus_prepare import imagenet_generator, catsdogs_generator, mnist_generator, \
                            luna_generator, rsna_generator, siim_generator, us_generator, ddsm_generator, \
                            esmira_generator
from cam_components import CAMAgent

from tqdm import tqdm


class WorkSpace:
    def __init__(self, task:Literal['imagenet', 'catsdogs', 'mnist', 'luna', 'rsna', 'siim', 'us', 'esmira', 'ddsm'],
                       method:Literal['gradcam', 'fullcam', 'gradcampp', 'xgradcam'],
                       apply_norm:bool = True,
                       num_classes:int=2, 
                       subset_size:int=32
                       ):
        # call different quantus_calculation for different purposes of evaluation
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.task = task
        self.method = method
        # ------------------------------- model, dataset initialization ------------------------------- #
        model, target_layer, dataset, groups, ram, cam_type = self._task_generator(task)
        self.name_str = f'quantus_cam/{task}_{method}_norm{apply_norm}'  # for metric need another name

        self.model = model.to(self.device)
        self.dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

        # ------------------------------- explaination agent initialization ------------------------------- # 
        self.norm = 'norm' if apply_norm else False
        self.agent = CAMAgent(model, target_layer, self.dataloader,  
                            groups, ram,
                            # optional:
                            cam_method=method, name_str=f'{self.name_str}',# cam method and im paths and cam output
                            batch_size=1, select_category=None,  # info of the running process
                            rescale=self.norm,  remove_minus_flag=False, scale_ratio=2,
                            feature_selection='all', feature_selection_ratio=1.,  # feature selection
                            randomization=None,  # model randomization for sanity check
                            use_pred=False,
                            rescaler=None,  # outer scaler
                            cam_type=cam_type  # output 2D or 3D
                            )

        self.metrics = obtain_metrics_results(num_classes, subset_size)
        self.results = {self.method:{}}

        self.score_to_be_calculate:list[str] = [metric for metric, _ in self.metrics.items()]
        self.save_dir = r'./output/quantus'
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.save_file = f'{task}_{method}_norm{apply_norm}.txt'

    
    def _task_generator(self, task:Literal['imagenet', 'catsdogs', 'mnist', 'luna', 'rsna', 'siim', 'us', 'esmira', 'ddsm']):
        if task=='imagenet': return imagenet_generator()
        if task=='catsdogs': return catsdogs_generator()
        if task=='mnist': return mnist_generator()
        if task=='luna': return luna_generator()
        if task=='rsna': return rsna_generator()
        if task=='siim': return siim_generator()
        if task=='us': return us_generator()
        if task=='esmira': return esmira_generator() 
        if task=='ddsm': return ddsm_generator()    
        # return model, target_layer, dataset, groups, ram, cam_type


    def _explain_func(self, model, inputs, targets, **kwargs) -> np.ndarray:
        # [batch, channel, (D), L, W]
        targets = int(targets[0])
        if isinstance(inputs, np.ndarray): inputs = torch.from_numpy(inputs)
        cam = self.agent.indiv_return(inputs, creator_target_category=targets, pred_flag=False) # make sure the input is a 4-dimension tensor [batch, channel, W, H]
        # [batch, group(1), cluster/tc, (D), L, W]
        cam = np.squeeze(cam, axis=(1, 2))
        if len(cam.shape)<len(inputs.shape):
            cam = np.expand_dims(cam, axis=1)  # [batch, (D), L, W] -> [batch, 1, (D), L, W] 
        if np.max(cam)<0: cam[cam==np.max(cam)] = 0
        return cam  # [batch, channel, (D), L, W] -> [batch, 1, (D), L, W] 
    

    def run_quantus(self, max_iter:int=5000):
        for metric, metric_func in self.metrics.items():
            print(f"Evaluating {metric} of {self.method} method with norm{self.norm}.")
            gc.collect()
            torch.cuda.empty_cache()
            self.quantus_calculation(metric_name=metric, metric_func=metric_func, max_iter=max_iter)
            gc.collect()
            torch.cuda.empty_cache()


    def quantus_calculation(self, metric_name:str, metric_func, max_iter:int=5000):
        temp_score = []
        for batch_idx, (x_batch, y_batch) in tqdm(enumerate(self.dataloader)):
            # print(f"Batch {batch_idx + 1}/{len(self.dataloader)}")
            # x_batch torch.Tensor [b, c, l, w]
            # cam_batch = self._explain_func(x_batch)  # [b, 1, l, w]
            x_batch = x_batch.cpu().numpy()
            y_batch = y_batch.cpu().numpy()
            #scores = metric_func(model=self.model, x_batch=x_batch, y_batch=y_batch, a_batch=cam_batch, device=self.device)
            try:
                scores = metric_func(model=self.model,
                                    x_batch=x_batch,
                                    y_batch=y_batch,
                                    device=self.device,
                                    explain_func=self._explain_func,
                                    )
                if scores[0]>10 or scores[0]<-10 or (not isinstance(scores[0], np.float64)): 
                    continue
                temp_score.extend(scores)
            except:
                continue
            if max_iter and batch_idx>max_iter: break
            

        assert len(temp_score)>0
        temp_score = [x for x in temp_score if not np.isnan(x)]
        self.results[self.method][metric_name] = sum(temp_score)/len(temp_score)

        self.summary()

    
    def summary(self):
        # take from self.results[self.method][metric]
        df = pd.DataFrame.from_dict(self.results, orient="index")
        df.index.name = "method"
        df.to_csv(os.path.join(self.save_dir, self.save_file))




