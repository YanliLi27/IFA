import torch
from tqdm import tqdm
from cam_components.indiv.sharedcam import SharedCAM


class ScoreCAM(SharedCAM):
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
        super(ScoreCAM, self).__init__(
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
        with torch.no_grad():
            upsample = torch.nn.UpsamplingBilinear2d(
                size=input_tensor.shape[-2:])
            activation_tensor = torch.from_numpy(activations)
            if self.cuda:
                activation_tensor = activation_tensor.cuda()

            upsampled = upsample(activation_tensor)

            maxs = upsampled.view(upsampled.size(0),
                                  upsampled.size(1), -1).max(dim=-1)[0]
            mins = upsampled.view(upsampled.size(0),
                                  upsampled.size(1), -1).min(dim=-1)[0]
            if len(input_tensor.shape)==4:
                maxs, mins = maxs[:, :, None, None], mins[:, :, None, None]
            elif len(input_tensor.shape)==5:
                maxs, mins = maxs[:, :, None, None, None], mins[:, :, None, None, None]
            upsampled = (upsampled - mins) / (maxs - mins)

            if len(input_tensor.shape)==4:
                input_tensors = input_tensor[:, None,
                                            :, :] * upsampled[:, :, None, :, :]
            elif len(input_tensor.shape)==5:
                input_tensors = input_tensor[:, None, :,
                                            :, :] * upsampled[:, :, None, :, :, :]

            if hasattr(self, "batch_size"):
                BATCH_SIZE = self.batch_size
            else:
                BATCH_SIZE = 16

            scores = []
            for batch_index, tensor in enumerate(input_tensors):
                category = target_category[batch_index]
                for i in tqdm.tqdm(range(0, tensor.size(0), BATCH_SIZE)):
                    batch = tensor[i: i + BATCH_SIZE, :]
                    outputs = self.model(batch).cpu().numpy()[:, category]
                    scores.extend(outputs)
            scores = torch.Tensor(scores)
            scores = scores.view(activations.shape[0], activations.shape[1])

            weights = torch.nn.Softmax(dim=-1)(scores).numpy()
            return weights


