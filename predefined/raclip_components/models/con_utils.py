import torch.nn as nn
# ViTBlock: simplified ViT block, merging channel and dim
# dim:(channels of input), 
# depth:(num of transformer block)[2,4,3],
# kernel_size:(kernel size of convlutional neural networks)
# patch_size:(patch size of transformer)
# heads:(heads number/kernel number)
# att_dim:(nodes of mlp in attention module)
# mlp_dim:(nodes of mlp in feedfward module)
# groups:(groups for convolution)
# dropout
from typing import Optional

class NormCNN(nn.Module):
    def __init__(self, inp, oup, kernal_size=3, stride=1, groups=4):
        super().__init__()
        self.cnn = nn.Sequential(
        nn.Conv2d(inp, oup, kernal_size, stride, 1, groups=groups, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )
    
    def forward(self, x):
        return self.cnn(x)


class DSConvBlock(nn.Module):
    def __init__(self, inp, oup, stride=1, groups:int=4, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup  # 必须要输入输出size一致才能进行残差加和

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, groups=groups, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
            # dw
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=groups*hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, groups=groups, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
        

class NormalConvBlock(nn.Module):
    '''
    A 2-layer Normal ConvBlock with expansion
    inp: input channel
    oup: output channel, default -- None
    kernel_size: the size of conv kernel
    stride: the stride of the conv
    expansion: inner expansion -- the width

    * if oup is None, the oup== inp * expansion
    '''
    def __init__(self, inp, oup, kernal_size=3, stride=1, groups:int=4, expansion:int=0):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1  and inp == oup  # 必须要输入输出size一致才能进行残差加和

        if expansion==0:
            self.conv = nn.Sequential(
                nn.Conv2d(inp, oup, kernal_size, stride, 1, bias=False, groups=groups),
                nn.BatchNorm2d(oup),
                nn.SiLU(),
            )
        else:
            hidden_dim = int(inp*expansion)
            self.conv = nn.Sequential(
                nn.Conv2d(inp, hidden_dim, 3, 1, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 3, 1, 1, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
        

def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v