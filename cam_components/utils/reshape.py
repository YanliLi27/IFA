import torch

def reshape_transform(tensor:torch.Tensor, height=14, width=14):
    # TODO NEED TO CONSIDER THE SIZE AT THAT LAYER
    if len(tensor.shape)==4:
        # B, C, L, W --> B, (ph*pw), L/ph*W/pw, C
        result = tensor[:, 1 :  , :].reshape(tensor.size(0), height, width, tensor.size(2))

        # Bring the channels to the first dimension,
        # like in CNNs.
        result = result.transpose(2, 3).transpose(1, 2)
        return result
    elif len(tensor.shape)==5:
        result = tensor[:, 1: , ]