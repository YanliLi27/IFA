import numpy as np
from cam_components.core.sharedcam import SharedCAM


class GradCAM(SharedCAM):
    def __init__(self, 
                 model,  # model, mandantory
                 target_layers,  # selected_layers for gradient/feature trace, mandantory
                 ram:bool,  # if regression task, mandantory
                 use_cuda:bool=False,  # if use GPU, optional
                 reshape_transform=None,  # for transformer, mlp or other networks that change the spatial placement
                 compute_input_gradient:bool=False,  # if compute the gradients of input, only used when use input as the feature
                 uses_gradients:bool=True,  # calculate the gradients, only false when use pertubation methods
                 ifaoperation:bool=True,  # if it's in the process of ifa process
                 # only for creator
                 groups:int=1,  # if use group conv, need to seperate them
                 importance_matrix=None,  # if input from outside, then no need to rebuild an importance matrix
                 value_max=None,  # if input max and min from outside, then no need to use importance matrix
                 value_min=None,   
                 remove_minus_flag:bool=False,  # whether remove the values below 0
                 out_logit:bool=False,  # use logit as output instead of confidence as the score
                 tanh_flag:bool=False,  # use hyperbolic tangent function to rescale
                 t_max:float=0.95,
                 t_min:float=0.05):
        super(GradCAM, self).__init__(
                model,
                target_layers,
                ram,
                use_cuda,
                reshape_transform,
                compute_input_gradient,
                uses_gradients,
                ifaoperation,
                groups,
                importance_matrix,
                value_max=value_max,
                value_min=value_min,
                remove_minus_flag=remove_minus_flag,
                out_logit=out_logit,
                tanh_flag=tanh_flag,
                t_max=t_max,
                t_min=t_min)

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads,
                        ):
        if len(input_tensor.shape)==4:
            return np.mean(grads, axis=(2, 3)) # for 2D: [batch, channel, y, x], for 3D: [batch, channel, z, y, x]
        elif len(input_tensor.shape)==5:
            return np.mean(grads, axis=(2, 3, 4))
        else:
            raise ValueError(f'the shape is not supported: {input_tensor.shape}')


