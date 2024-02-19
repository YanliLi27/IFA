import numpy as np
from typing import Union

class Rescaler():
    def __init__(self, value_max:Union[None, float]=None, value_min:Union[None, float]=None, 
                 remove_minus_flag:bool=False, t_max:float=0.95, t_min:float=0.05, rescale_func:str='norm',
                 scale_ratio:float=1.5):
        self.value_max = value_max * scale_ratio if value_max else None
        self.value_min = value_min * scale_ratio if value_min else None
        self.rmf = remove_minus_flag
        self.t_max = t_max
        self.t_min = t_min
        if value_max and value_min:
            self.para_k = (np.arctanh(t_max) - np.arctanh(t_min))/(value_max-value_min)
            self.para_b = (np.arctanh(t_max)*value_min-np.arctanh(t_min)*value_max)/(value_min-value_max)
        else:
            self.para_b, self.para_k = None, None
        if rescale_func  in ['norm', 'tanh', 'norm_multi', 'tanh_multi', 'self_determin']:
            self.func = rescale_func.replace('_multi', '') if 'multi' in rescale_func else rescale_func
        else:
            self.func = 'norm'
            print('rescale func should be one of the defined')
      
    
    def _scale_cam_image(self, img):
        if self.value_max and self.value_min:
            value_max = self.value_max
            value_min = 0 if (self.rmf and self.value_max>0) else self.value_min
        else:
            value_max = np.max(img) + 1e-7
            value_min = np.min(img)
        img = (img - value_min) / (value_max - value_min)        
        return img


    def _tanh_scale_cam_image(self, img):
        return np.tanh(self.para_k*img+self.para_b)
    

    def _tanh_unscale(self, img):
        return (np.arctanh(img+1e-10) - self.para_b)/(self.para_k + 1e-10)

        
    def rescale(self, cam):
        if self.rmf:
            cam = np.maximum(cam, 0)
        return self._tanh_scale_cam_image(cam) if self.func=='tanh' else self._scale_cam_image(cam) 

    
    def unscale(self, cam):
        return self._tanh_unscale(cam) if self.func=='tanh' else cam