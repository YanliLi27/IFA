import torch

def reshape_transformer(tensor:torch.Tensor, height=16, width=16):
    # TODO NEED TO CONSIDER THE SIZE AT THAT LAYER
    if len(tensor.shape)==3:
        # B, C, L, W --> B, L/ph*W/pw, (ph*pw*C)  was applied in transformer
        # In mobile transformer:
        # B, C, L, W --> B, (ph*pw), L/ph*W/pw, C
        # therefore needs to be reshaped back differently
        if tensor.size(1)!=height*width:
            result = tensor[:, 1 :  , :].reshape(tensor.size(0), height, width, tensor.size(2))
            # this is for [B, L/ph*W/pw + 1, (ph*pw*C)], start from 1:, because the first one is positional embedding
        elif tensor.size(1)==height*width:
            result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
        else:
            raise ValueError(f'size of tensor {tensor.size(1)} should equal to height {height} * width {width}')
        # [B, L/ph*W/pw + 1, (ph*pw*C)] -> [B, L/ph*W/pw, (ph*pw*C)] -> [B, L/ph, W/pw, (ph*pw*C)]

        # Bring the channels to the first dimension,
        # like in CNNs.
    elif len(tensor.shape)==4:
        # In mobile transformer:
        # B, C, L, W --> B, (ph*pw), L/ph*W/pw, C
        # therefore needs to be reshaped back differently
        tensor = tensor.transpose(1, 2)  
        # [B, (ph*pw), L/ph*W/pw, C] -> [B, L/ph*W/pw, (ph*pw), C]
        if tensor.size(1)!=height*width:
            result = tensor[:, 1 :  , :].reshape(tensor.size(0), height, width, tensor.size(2)*tensor.size(3))
            # this is for [B, L/ph*W/pw + 1, (ph*pw), C], start from 1:, because the first one is positional embedding
        elif tensor.size(1)==height*width:
            result = tensor.reshape(tensor.size(0), height, width, tensor.size(2)*tensor.size(3))
        # [B, L/ph, W/pw, ph*pw*C]
        else:
            raise ValueError(f'size of tensor {tensor.size(1)} should equal to height {height} * width {width}')
    result = result.transpose(2, 3).transpose(1, 2)  # [B, L/ph, W/pw, (ph*pw*C)] -> [B, (ph*pw*C), L/ph, W/pw]
    return result