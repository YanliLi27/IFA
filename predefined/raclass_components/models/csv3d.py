import torch
import torch.nn as nn
from models.csv_utils import Transformer
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
from einops import rearrange
from typing import Callable, Any, Optional, List, Union, Literal


def conv_nxn_bn_group3d(inp:int, oup:int, kernal_size:Union[int, tuple]=3, stride:Union[int, tuple]=1, groups:int=4):
    return nn.Sequential(
        nn.Conv3d(inp, oup, (1, kernal_size, kernal_size), (1, stride, stride), (0, 1, 1), groups=groups, bias=False),
        nn.BatchNorm3d(oup),
        nn.SiLU()
    )


class NormCNN3d(nn.Module):
    def __init__(self, inp, oup, kernal_size=3, stride=1, groups=4):
        super().__init__()
        self.cnn = nn.Sequential(
        nn.Conv3d(inp, oup, (1, kernal_size, kernal_size), (1, stride, stride), (0, 1, 1), groups=groups, bias=False),
        nn.BatchNorm3d(oup),
        nn.SiLU()
    )
    
    def forward(self, x):
        return self.cnn(x)


class DSConvBlock3d(nn.Module):
    def __init__(self, inp, oup, stride=1, groups:int=4, expansion=4):
        super().__init__()
        assert stride in [1, 2]

        hidden_dim = int(inp * expansion)
        self.use_res_connect = stride == 1 and inp == oup  # 必须要输入输出size一致才能进行残差加和

        self.conv = nn.Sequential(
            # pw
            nn.Conv3d(inp, hidden_dim, 1, 1, 0, groups=groups, bias=False),
            nn.BatchNorm3d(hidden_dim),
            nn.SiLU(),
            # dw
            nn.Conv3d(hidden_dim, hidden_dim, (1, 3, 3), (1, stride, stride), 1, groups=groups*hidden_dim, bias=False),
            nn.BatchNorm3d(hidden_dim),
            nn.SiLU(),
            # pw-linear
            nn.Conv3d(hidden_dim, oup, 1, 1, 0, groups=groups, bias=False),
            nn.BatchNorm3d(oup),
        )

    def forward(self, x):  # [b, c0, d, l, w] -> [b, c1, d, l, w]
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
        

class NormalConvBlock3d(nn.Module):
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
        assert stride in [1, 2]

        self.use_res_connect = stride == 1  and inp == oup  # 必须要输入输出size一致才能进行残差加和

        if expansion==0:
            self.conv = nn.Sequential(
                nn.Conv3d(inp, oup, (1, kernal_size, kernal_size), (1, stride, stride), (0,1,1), bias=False, groups=groups),
                nn.BatchNorm3d(oup),
                nn.SiLU(),
            )
        else:
            hidden_dim = int(inp*expansion)
            self.conv = nn.Sequential(
                nn.Conv3d(inp, hidden_dim, (1, 3, 3), 1, (0,1,1), bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.SiLU(),
                # dw
                nn.Conv3d(hidden_dim, hidden_dim, (1, 3, 3), (1, stride, stride), (0,1,1), groups=hidden_dim, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv3d(hidden_dim, oup, (1, 3, 3), 1, (0,1,1), bias=False),
                nn.BatchNorm3d(oup),
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


class ViTBlock3d(nn.Module):
    def __init__(self, channel:int, kernel_size:Union[int, tuple], patch_size:Union[int, tuple], 
                 groups:int, depth:int, mlp_dim:int, dropout:float=0.3, 
                 attn_type:Literal['normal', 'mobile', 'parr_normal', 'parr_mobile']='mobile',
                 out_channel:Union[int, None]=None):
        super().__init__()
        '''
        ViTBlock: simplified ViT block, merging channel and dim
        channel:(channels of input), 
        depth:(num of transformer block)[2,4,3],
        kernel_size:(kernel size of convlutional neural networks)
        patch_size:(patch size of transformer)
        heads:(heads number/kernel number)
        att_dim:(nodes of mlp in attention module)
        mlp_dim:(nodes of mlp in feedfward module)
        groups:(groups for convolution)
        dropout
        '''
        out_ch = out_channel if out_channel is not None else channel
        self.attn_type = attn_type
        if type(patch_size)==list or type(patch_size)==tuple:
            self.ph, self.pw = patch_size  
        else:
            self.ph, self.pw = patch_size, patch_size
        tfdim = channel*self.ph*self.pw if 'normal' in attn_type else channel
        self.transformer = Transformer(tfdim, depth, 4, 8, mlp_dim, dropout, attn_type=attn_type)
        # Transformer(dim(channels of input), depth(num of transformer block)[2,4,3], 
        #             4(heads number/kernel number), 8(length of mlp in attention),
        #             mlp_dim(nodes of mlp, extension), dropout)
        self.merge_conv = conv_nxn_bn_group3d(2 * channel, out_ch, kernel_size, stride=1, groups=groups)
    
    def forward(self, x):
        # input size: B, C, D, L, W 
        B, C, D, h, w = x.shape
        x = torch.reshape(x, (B, C*D, h, w)) # [B, C, D, L, W] -> [B, CD, L, W]
        y = x.clone()  # [B, CD, L, W]
        
        # Global representations        
        x = rearrange(x, 'b (d pd) (h ph) (w pw) -> b (ph pw) (h w pd) d', pd=D, ph=self.ph, pw=self.pw)
        # B, CD, L, W --> B, (ph*pw), L/ph*W/pw*D, C
        x = self.transformer(x)
        x = rearrange(x, 'b (ph pw) (h w pd) d -> b (d pd) (h ph) (w pw)', h=h//self.ph, w=w//self.pw, pd=D, ph=self.ph, pw=self.pw)
        # B, (ph*pw), L/ph*W/pw*D, C  -->  B, CD, L, W
        # x: B, CD, L, W 

        # Fusion
        x = torch.cat((x, y), 1)
        # x: B, 2CD, L, W
        x = torch.reshape(x, (B, 2*C, D, h, w))  # B, 2CD, L, W --> B, 2C, D, L, W
        x = self.merge_conv(x)
        # x: B, C, D, L, W
        
        return x


class CSViT3d(nn.Module):
    def __init__(self, image_size, in_ch:int, num_classes:int=2,  # dataset related
                 num_features:int=43, extension:int=0,  # intermediate related
                 groups:int=4, width:int=1,# basic
                 dsconv:bool=False, attn_type:Literal['normal', 'mobile', 'parr_normal', 'parr_mobile']='normal', # module type
                 block_setting:Optional[List[List]] = None,
                 patch_size:Union[list,tuple,int]=(2, 2),  # vit
                 mode_feature:bool=True, #
                 dropout:bool=True,
                 initialization:bool=False
                 ):
        super().__init__()
        '''
        # General attributes
        image_size: the D, L, W size of input image, only 3D images
        in_ch: the input channel of the images
        num_classes: the output classes
        groups: multiple input sources
        width: the scale of the model

        # Module type attributes
        dsconv: use depthwise seperable conv or not
        parallel: use simplified vit or not

        # Block setting: list of lists, for model building
        block('c' for conv), out_channels, kernal_size, stride, groups, num of blocks, expansion(only for dsconv)
        block('t' for vit), out_channels, kernel_size, patch_size, groups, depth (num of blcoks), mlp_dim(like the expansion)
        [b, c, k, s, g, d, e] for each layer

        _ expansion: the width of conv layers, if >1: In_C->Mid_C->expansion*Mid_C->Mid_C->Out_C, else: In_C->Mid_C->Out_C

        # patch size:
        used for vit rearrange
        
        # initializtion:
        used for model initialization

        # ViT inside
        channel:(channels of input), 
        depth:(num of transformer block)[2,4,3],
        kernel_size:(kernel size of convlutional neural networks)
        patch_size:(patch size of transformer)
        heads:(heads number/kernel number)
        att_dim:(nodes of mlp in attention module)
        mlp_dim:(nodes of mlp in feedfward module)
        groups:(groups for end convolution)
        dropout
        '''
        self.mode = mode_feature
        self.num_features = num_features
        # vit setting: --------------------------------------------------------------------------------------------#
        id, ih, iw = image_size  # [d, l, w]
        if type(patch_size)==list or type(patch_size)==tuple:
            ph, pw = patch_size  
        else:
            ph, pw = patch_size, patch_size
        assert ih % ph == 0 and iw % pw == 0
        # vit setting: --------------------------------------------------------------------------------------------#
        # block setting: ------------------------------------------------------------------------------------------#
        # out_channels, kernal_size, stride, groups, num of blocks, expansion (only for dsconv, 0 if normal)
        if block_setting is None:
            block_setting = [
                # block('c' for conv), out_channels, kernal_size, stride, groups, num of blocks, expansion(only for dsconv)
                # block('t' for vit), out_channels, kernel_size, patch_size, groups, depth, mlp_dim(like the expansion)
                # b,  c,  k, s, g, d, e  
                ['c', 32, 3, 1, groups, 1, 0],  # [B, g*C, D, L/2(256), W/2(256)] -> [B, g*C, D, L/2(256), W/2(256)]
                ['c', 64, 3, 2, groups, 3, 0],  # downsample + 3 conv # [B, g*C, D, L/2(256), W/2(256)] -> [B, g*C, D, L/4(128), W/4(128)]
                ['c', 96, 3, 2, groups, 1, 0],  # downsample + conv # [B, g*C, D, L/4(128), W/4(128)] -> [B, g*C, D, L/8(64), W/8(64)]
                ['c', 160, 3, 2, groups, 1, 0],  # downsample + conv # [B, g*C, D, L/8(64), W/8(64)] -> [B, g*C, D, L/16(32), W/16(32)]
                ['t', 160, 3, pw, groups, 3, 640],  # vit # [B, g*C, D, L/16(32), W/16(32)] -> [B, g*C, D, L/16(32), W/16(32)]
            ]
        if dsconv:
            convblock = DSConvBlock3d
        else:
            convblock = NormalConvBlock3d

        vitblock = ViTBlock3d  # only 2d input accepted
        # blcok setting: ------------------------------------------------------------------------------------------#


        # block building -----------------------------------------------------------------------------------------#
        input_channel = _make_divisible(16* groups * width, 8)
        last_channel = _make_divisible(160 * groups* width, 160)
        features:List[nn.Module] = [
            NormCNN3d(in_ch, input_channel, stride=2, groups=groups)
        ]  # [B, g, D(7), L(512), W(512)]-> # [B, g*C, D(7), L/2(256), W/2(256)]  降维是有必要的，滤掉大多数的特异点,减少FLOPs

        for b, c, k, s, g, d, e in block_setting:
            output_channel = _make_divisible(c * width, 8)
            if b == 't':
                patch_size = [s, s]
                features.append(vitblock(output_channel, k, patch_size, g, d, e, attn_type='mobile'))
                # transformer doesnt change the output channel
            elif b=='c':
                for i in range(d):
                    stride = s if i == 0 else 1
                    features.append(convblock(input_channel, output_channel, k, stride, g, e))
                    input_channel = output_channel

        features.append(
            NormalConvBlock3d(inp=input_channel, oup=last_channel, kernal_size=1, stride=1, groups=groups, expansion=4)
        )  # [B, g*C0, D, L/16(32), W/16(32)] -> [B, g*C1, D, L/16(32), W/16(32)]
        self.features = nn.Sequential(*features)

        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))  # B, g*C, 1, 1, 1

        # not sure
        self.protofeature = nn.Linear(last_channel, num_features+extension, bias=False)

        if dropout:
            self.fc = nn.Sequential(
                nn.SiLU(True),
                nn.Dropout(0.3),
                nn.Linear(num_features+extension, num_classes, bias=False)
            )
        else:
            self.fc = nn.Sequential(
                nn.SiLU(True),
                nn.Linear(num_features+extension, num_classes, bias=False)
            )
        

        if initialization:
                    # weight initialization
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.zeros_(m.bias)

            
    def forward(self, x):
        x = self.features(x)  # [B, G, D, L, W] --> [B, g*C1, D, L/16(32), W/16(32)]
        x = self.pool(x).view(-1, x.shape[1])  # [B, g*C1,  D, L/16(32), W/16(32)] -> [B, C, 1, 1, 1] -> [B, C]
        x = self.protofeature(x)  # [B, C] --> [B, feature+extension]
        if self.mode:
            return x[:, :self.num_features]
        else:
            x = self.fc(x)  # [B, feature+extension] --> [B, num_classes]
            return x


def make_csv3dmodel(img_2dsize, inch, num_classes=2, 
                  num_features=43, extension=157,
                  groups=4, width=1, dsconv=False, 
                  attn_type='normal', patch_size=(2,2), 
                  mode_feature:bool=False, dropout:bool=True, init:bool=False):
    block_setting = [
                # block('c' for conv), out_channels, kernal_size, stride, groups, num of blocks, expansion(only for dsconv)
                # block('t' for vit), out_channels, kernel_size, patch_size, groups, depth, mlp_dim(like the expansion)
                # b,  c,  k, s, g, d, e  
                ['c', 32, 3, 1, groups, 1, 0],  # [B, g*C, D, L/2(256), W/2(256)] -> [B, g*C, D, L/2(256), W/2(256)]
                ['c', 64, 3, 2, groups, 3, 0],  # downsample + 3 conv # [B, g*C, D, L/2(256), W/2(256)] -> [B, g*C, D, L/4(128), W/4(128)]
                ['c', 96, 3, 2, groups, 1, 0],  # downsample + conv # [B, g*C, D, L/4(128), W/4(128)] -> [B, g*C, D, L/8(64), W/8(64)]
                ['c', 160, 3, 2, groups, 1, 0],  # downsample + conv # [B, g*C, D, L/8(64), W/8(64)] -> [B, g*C, D, L/16(32), W/16(32)]
                ['t', 160, 3, patch_size[0], groups, 3, 640],  # vit # [B, g*C, D, L/16(32), W/16(32)] -> [B, g*C, D, L/16(32), W/16(32)]
            ]
    return CSViT3d(img_2dsize, inch, num_classes=num_classes, 
                    num_features=num_features, extension=extension,
                    groups=groups, width=width,# basic
                    dsconv=dsconv, attn_type=attn_type, # module type
                    block_setting=block_setting,
                    patch_size=patch_size,  # vit
                    mode_feature=mode_feature,
                    dropout=dropout,
                    initialization=init
                    )


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    img = torch.randn(1, 20, 512, 512)

    vit = make_csv3dmodel()
    out = vit(img)
    print(out.shape)
    print(count_parameters(vit))
