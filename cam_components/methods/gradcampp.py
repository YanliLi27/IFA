import numpy as np
from cam_components.indiv.sharedcam import SharedCAM


class GradCAMPP(SharedCAM):
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
        super(GradCAMPP, self).__init__(
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
        grads_power_2 = grads**2
        grads_power_3 = grads_power_2 * grads
        # Equation 19 in https://arxiv.org/abs/1710.11063
        if len(input_tensor.shape)==4:
            sum_activations = np.sum(activations, axis=(2, 3))
            eps = 0.000001
            aij = grads_power_2 / (2 * grads_power_2 +
                                sum_activations[:, :, None, None] * grads_power_3 + eps)
            # Now bring back the ReLU from eq.7 in the paper,
            # And zero out aijs where the activations are 0
            aij = np.where(grads != 0, aij, 0)
            weights = np.maximum(grads, 0) * aij
            return np.sum(weights, axis=(2, 3))
        elif len(input_tensor.shape)==5:
            sum_activations = np.sum(activations, axis=(2, 3, 4))
            eps = 0.000001
            aij = grads_power_2 / (2 * grads_power_2 +
                                sum_activations[:, :, None, None, None] * grads_power_3 + eps)
            # Now bring back the ReLU from eq.7 in the paper,
            # And zero out aijs where the activations are 0
            aij = np.where(grads != 0, aij, 0)
            weights = np.maximum(grads, 0) * aij
            return np.sum(weights, axis=(2, 3, 4))
        else:
            raise ValueError(f'the shape is not supported: {input_tensor.shape}')
