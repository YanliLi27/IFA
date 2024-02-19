import numpy as np
from cam_components.core.sharedcam import SharedCAM


class GradCAMPP(SharedCAM):
    def __init__(self, 
                 model,  # model, mandantory
                 target_layers,  # selected_layers for gradient/feature trace, mandantory
                 ram:bool,  # if regression task, mandantory
                 use_cuda:bool=False,  # if use GPU, optional
                 reshape_transform=None,  # for transformer, mlp or other networks that change the spatial placement
                 compute_input_gradient:bool=False,  # if compute the gradients of input, only used when use input as the feature
                 uses_gradients:bool=True,  # calculate the gradients, only false when use pertubation methods
                 # only for creator
                 groups:int=1,  # if use group conv, need to seperate them
                 importance_matrix=None,  # used for selecting features, could be designed instead of created
                 out_logit:bool=False,  # use logit as output instead of confidence as the score
                 ):
        super(GradCAMPP, self).__init__(
                                        model,
                                        target_layers,
                                        ram=ram,
                                        use_cuda=use_cuda,
                                        reshape_transform=reshape_transform,
                                        compute_input_gradient=compute_input_gradient,
                                        uses_gradients=uses_gradients,
                                        groups=groups,
                                        importance_matrix=importance_matrix,
                                        out_logit=out_logit
                                        )

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
