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
                 # only for creator
                 groups:int=1,  # if use group conv, need to seperate them
                 importance_matrix=None,  # used for selecting features, could be designed instead of created
                 out_logit:bool=False,  # use logit as output instead of confidence as the score
                 ):
                 
        super(GradCAM, self).__init__(
                model,
                target_layers,
                ram=ram,
                use_cuda=use_cuda,
                reshape_transform=reshape_transform,
                compute_input_gradient=compute_input_gradient,
                uses_gradients=uses_gradients,
                groups=groups,
                importance_matrix=importance_matrix,
                out_logit=out_logit,
                )

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


