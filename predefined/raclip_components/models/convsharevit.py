import torch
import torch.nn as nn
from predefined.raclip_components.models.csv_utils import ViTBlock
from predefined.raclip_components.models.con_utils import DSConvBlock, NormalConvBlock, NormCNN, _make_divisible
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


class ConvShareViT(nn.Module):
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
        image_size: the L, W size of input image, only 2D images
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
        ih, iw = image_size
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
                ['c', 32, 3, 1, groups, 1, 0],  # one layer of normal conv # B, 4*C, L/2(256), W/2(256) 不变
                ['c', 64, 3, 2, groups, 3, 0],  # downsample + three layers of conv # B, 4*C, L/2(256), W/2(256) -> B, 4*C, L/4(128), W/4(128)
                ['c', 96, 3, 2, groups, 1, 0],  # downsample + one layer of conv # B, 4*C, L/4(128), W/4(128) -> B, 4*C, L/8(64), W/8(64)
                ['t', 96, 3, 2, groups, 6, 240],  # vit # B, 4*C, L/8(64), W/8(64) -> B, 4*C, L/8(64), W/8(64)
                # vit信息转换，跨输入进行信息交互
                ['c', 160, 3, 2, groups, 1, 0],  # downsample + one layer of conv # B, 4*C, L/8(64), W/8(64) -> B, 4*C, L/16(32), W/16(32)
                # 用于中间信息转换的cnn，主要进行降维
                # B, 4*C, L/8(64), W/8(64) -> B, 4*C, L/16(32), W/16(32)
                # 此层也用作于RAMRIS输出与CAM生成
                # CAM分辨率和特征提取感受野的矛盾：CAM分辨率要求尽量降低downsample的个数，
                # 但是提高感受野要求尽可能多的卷积（不会降低分辨率，但FLOPs很高）和尽可能downsample（降低分辨率，FLOPs也降低）
                # We need a trade-off between the resolution of CAMs and the high-level features -> performance
                # 或者用多尺度的CAM来提升分辨率--multiplication instead of accumulation. -->保证精准度然后提升分辨率--未必可行
                # 隔层的gradient会不准确，底层的分辨率太低
                # FLOPs和分辨率一定有一个trade-off。 --# 32 还是 64 要看加不加后面这一段的效果怎么样
                ['t', 160, 3, 1, groups, 3, 640],  # vit # B, 4*C, L/16(32), W/16(32) -> B, 4*C, L/16(32), W/16(32)
                ['c', 160, 3, 1, groups, 1, 0],  # one layer of conv
            ]
        if dsconv:
            convblock = DSConvBlock
        else:
            convblock = NormalConvBlock

        vitblock = ViTBlock
        # blcok setting: ------------------------------------------------------------------------------------------#


        # block building -----------------------------------------------------------------------------------------#
        input_channel = _make_divisible(16* groups * width, 8)
        last_channel = _make_divisible(160 * groups* width, 8)
        features:List[nn.Module] = [
            NormCNN(in_ch, input_channel, stride=2, groups=groups)
        ]  # B, 4*D, L(512), W(512) -> # B, 4*C, L/2(256), W/2(256)  降维是有必要的，滤掉大多数的特异点,减少FLOPs

        for b, c, k, s, g, d, e in block_setting:
            output_channel = _make_divisible(c * width, 8)
            if b == 't':
                patch_size = [s, s]
                features.append(vitblock(output_channel, k, patch_size, g, d, e, attn_type=attn_type))
                # transformer doesnt change the output channel
            elif b=='c':
                for i in range(d):
                    stride = s if i == 0 else 1
                    features.append(convblock(input_channel, output_channel, k, stride, g, e))
                    input_channel = output_channel

        features.append(
            NormalConvBlock(inp=input_channel, oup=last_channel, kernal_size=1, stride=1, groups=groups, expansion=4)
        )
        self.features = nn.Sequential(*features)

        self.pool = nn.AdaptiveAvgPool2d((1,1))  # B, 4*C, 1, 1

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
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.zeros_(m.bias)

            
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).view(-1, x.shape[1])
        x = self.protofeature(x)
        if self.mode:
            return x[:self.num_features]
        else:
            x = self.fc(x)
            return x


def make_csvmodel(img_2dsize=(512, 512), inch=20, num_classes=2, 
                  num_features=43, extension=157,
                  groups=4, width=1, dsconv=False, 
                  attn_type='normal', patch_size=(2,2), 
                  mode_feature:bool=False, dropout:bool=True, init:bool=False):
    block_setting = [
                # block('c' for conv), out_channels, kernal_size, stride, groups, num of blocks, expansion(only for dsconv)
                # block('t' for vit), out_channels, kernel_size, patch_size, groups, depth, mlp_dim(like the expansion)
                # b,  c,  k, s, g, d, e  
                ['c', 32, 3, 1, groups, 1, 0],  # one layer of normal conv # B, 4*C, L/2(256), W/2(256) 不变
                ['c', 64, 3, 2, groups, 3, 0],  # downsample + three layers of conv # B, 4*C, L/2(256), W/2(256) -> B, 4*C, L/4(128), W/4(128)
                ['c', 96, 3, 2, groups, 1, 0],  # downsample + one layer of conv # B, 4*C, L/4(128), W/4(128) -> B, 4*C, L/8(64), W/8(64)
                ['t', 96, 3, 2, groups, 2, 240],  # vit # B, 4*C, L/8(64), W/8(64) -> B, 4*C, L/8(64), W/8(64)
                ['c', 160, 3, 2, groups, 1, 0],  # downsample + one layer of conv # B, 4*C, L/8(64), W/8(64) -> B, 4*C, L/16(32), W/16(32)
                ['t', 160, 3, 1, groups, 3, 640],  # vit # B, 4*C, L/16(32), W/16(32) -> B, 4*C, L/16(32), W/16(32)
                ['c', 160, 3, 1, groups, 1, 0],  # one layer of conv
            ]
    return ConvShareViT(img_2dsize, inch, num_classes=num_classes, 
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

    vit = make_csvmodel()
    out = vit(img)
    print(out.shape)
    print(count_parameters(vit))
