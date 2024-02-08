# 需要以下功能：
# 1. IMG -> Words 的转换。 参数包括图片大小，patch大小。输出为一个经过切块的[b c (h p1) (w p2) -> b (h w) (p1 p2 c)]
# 此操作中，相当于在第一层将C变成了一个较大的数
# 2. Attention 中的多头注意力机制和换位机制
# 3. cls的repeat功能\
import torch
import numpy as np
import torch.nn as nn
from typing import Union, Tuple


def cls_repeat(cls_token:torch.Tensor, mode:int=0, batch_size:int=8)->torch.Tensor:
    size_tensor = len(cls_token.shape)
    size_mode = np.ones(size_tensor, dtype=np.int16)
    size_mode[mode] = batch_size
    print(size_mode)
    return cls_token.repeat(size_mode)


def rearrange(query:torch.Tensor, mode:str='getheads', heads:int=8)->torch.Tensor:
    assert mode in ['getheads', 'mergeheads']
    if mode == 'getheads':
        # 'b n (h d) -> b h n d', heads
        # [b, 1+cmm, dim_heads * heads] -> [b, heads, 1+cmm, dim_heads]
        b, n, h = query.shape
        query = query.reshape(b, n, heads, h/heads) # [b, 1+cmm, dim_heads * heads] -> [b, 1+cmm, heads, dim_heads]
        return torch.transpose(query, 1, 2)  # [b, 1+cmm, heads, dim_heads] -> [b, heads, 1+cmm, dim_heads]
    elif mode == 'mergeheads':
        # 'b h n d -> b n (h d)'
        # [b, heads, 1+cmm, dim_heads] -> [b, 1+cmm, dim_heads*heads]
        b, h, n, d = query.shape
        query = torch.transpose(query, 1, 2) # [b, heads, 1+cmm, dim_heads] -> [b, 1+cmm, heads, dim_heads]
        return query.reshape(b, n, h*d) # [b, 1+cmm, heads, dim_heads] -> [b, 1+cmm, heads* dim_heads]
    else:
        raise ValueError('not supported mode')


class WindowCreator(nn.Module):
    def __init__(self, dim_mode:float=2, p1:int=32, p2:int=32, p3:Union[None, int]=None):
        super().__init__()
        # 2D: 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width
        # 2.5D: 'b (c p3) (h p1) (w p2) -> b (c h w) (p1 p2 p3)', p1 = patch_height, p2 = patch_width, p3 = frame_patch_size
        # 3D: 'b c (f pf) (h p1) (w p2) -> b (f h w) (p1 p2 pf c)', p1 = patch_height, p2 = patch_width, p3 = frame_patch_size)
        assert dim_mode in [2, 3, 2.5]
        self.dim_mode = dim_mode
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        

    def forward(self, x:torch.Tensor):
        if self.dim_mode == 2:
            b, c, h, w = x.shape
            x = x.reshape((b, c, h/self.p1, self.p1, w/self.p2, w))  # 'b c (h p1) (w p2) -> b c h p1 w p2'
            x = torch.permute(x, (0, 2, 4, 3, 5, 1))  # 'b c h p1 w p2 -> b h w p1 p2 c' from (0 1 2 3 4 5) to (0, 2, 4, 3, 5, 1)
            return x.reshape((b, (h/self.p1 * w/self.p2), c*self.p1*self.p2)) # 'b h w p1 p2 c -> b (h w) (p1 p2 c)'
        elif self.dim_mode == 2.5:
            b, c, h, w = x.shape
            x = x.reshape((b, c/self.p3, self.p3, h/self.p1, self.p1, w/self.p2, self.p2))  # 'b (c p3) (h p1) (w p2) -> b c p3 h p1 w p2'
            x = torch.permute(x, (0, 1, 3, 5, 2, 4, 6)) # 'b c p3 h p1 w p2 -> b c h w p3 p1 p2' from (0 1 2 3 4 5 6) to (0, 1, 3, 5, 2, 4, 6)
            return x.reshape((b, (c/self.p3 *h/self.p1 * w/self.p2), self.p1*self.p2*self.p3)) # 'b c h w p3 p1 p2 -> b (c h w) (p1 p2 p3)'
        elif self.dim_mode == 3:
            b, c, l, h, w = x.shape
            x = x.reshape((b, c, l/self.p3, self.p3, h/self.p1, self.p1, w/self.p2, self.p2))  # 'b c (l p3) (h p1) (w p2) -> b c l p3 h p1 w p2'
            x = torch.permute(x, (0, 2, 4, 6, 3, 5, 7, 1)) # 'b c l p3 h p1 w p2 -> b l h w p3 p1 p2 c ' from (0 1 2 3 4 5 6 7) to (0,2,4,6,3,5,7,1)
            return x.reshape((b, (l/self.p3 *h/self.p1 * w/self.p2), self.p1*self.p2*self.p3*c)) # 'b c h w p1 p2 p3 -> b (c h w) (p1 p2 p3)'
        else:
            raise ValueError('not valid dim of input')