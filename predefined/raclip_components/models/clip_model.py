import torch
import torch.nn as nn


class conv_block_group(nn.Module):
    def __init__(self, ch_in, ch_out, group_num=2):
        super(conv_block_group, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, groups=group_num),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, groups=group_num),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(self, img_ch=5, group_cap=5, width=2):  # 6 is 3 TRA + 3 COR
        super(Encoder, self).__init__()
        group_num = img_ch // group_cap

        self.Maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.Conv1 = conv_block_group(ch_in=img_ch, ch_out=32*group_num*width, group_num=group_num)
        self.Conv2 = conv_block_group(ch_in=32*group_num*width, ch_out=64*group_num*width, group_num=group_num)
        self.Conv3 = conv_block_group(ch_in=64*group_num*width, ch_out=128*group_num*width, group_num=group_num)
        self.Conv4 = conv_block_group(ch_in=128*group_num*width, ch_out=256*group_num*width, group_num=group_num)
    def forward(self, x):
        # encoding path
        x = self.Conv1(x)
        x = self.Maxpool(x)
        x = self.Conv2(x)
        x = self.Maxpool(x)
        x = self.Conv3(x)
        x = self.Maxpool(x)
        x = self.Conv4(x)
        return x
    

class conv_block_group3d(nn.Module):
    def __init__(self, ch_in, ch_out, group_num=2):
        super(conv_block_group3d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False, groups=group_num),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch_out, ch_out, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False, groups=group_num),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.conv(x)
        return x


class Decoder(nn.Module):
    def __init__(self, out_ch=2, group_num=2, width=2, extension:int=0):
        super(Decoder, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier_fc = nn.Sequential(
            nn.Linear(256*group_num*width * 16, 4096),
            nn.LeakyReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.LeakyReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(4096, out_ch+extension),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier_fc(x)
        return x


class Encoder3d(nn.Module):
    def __init__(self, in_ch:int=2, group_num:int=2, width:int=1):  # 6 is 3 TRA + 3 COR
        super(Encoder3d, self).__init__()
        self.Maxpool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.Conv1 = conv_block_group3d(ch_in=in_ch, ch_out=16*group_num*width, group_num=group_num)
        self.Conv2 = conv_block_group3d(ch_in=16*group_num*width, ch_out=32*group_num*width, group_num=group_num)
        self.Conv3 = conv_block_group3d(ch_in=32*group_num*width, ch_out=64*group_num*width, group_num=group_num)
        self.Conv4 = conv_block_group3d(ch_in=64*group_num*width, ch_out=128*group_num*width, group_num=group_num)
        # [256*2, 7, 64, 64]

    def forward(self, x):
        # encoding path
        x = self.Conv1(x)
        x = self.Maxpool(x)
        x = self.Conv2(x)
        x = self.Maxpool(x)
        x = self.Conv3(x)
        x = self.Maxpool(x)
        x = self.Conv4(x)
        return x
    

class Decoder3d(nn.Module):
    def __init__(self, out_ch=2, depth:int=7, group_num:int=1, extension:int=0, poolsize:int=1):
        super(Decoder3d, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool3d((depth, poolsize, poolsize))  # compressed to 1, 4, 4
        self.classifier_fc = nn.Sequential(
            nn.Linear(128*depth*group_num*poolsize*poolsize, 2048),
            nn.LeakyReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.LeakyReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(2048, out_ch+extension),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier_fc(x)
        return x


class ModelClip(nn.Module):
    def __init__(self, group_num, group_cap:int=5, out_ch=2, dimension:int=2, width:int=2, 
                 extension:int=0,
                 init_weights: bool = True):
        super(ModelClip, self).__init__()
        
        if dimension==3:
            self.encoder_class = Encoder3d(in_ch=group_num, group_num=group_num, width=width)
            self.decoder = Decoder3d(out_ch=out_ch, depth=group_cap, group_num=group_num*width, extension=extension, poolsize=1)
        else:
            img_ch= group_num * group_cap
            self.encoder_class = Encoder(img_ch=img_ch, group_cap=group_cap, width=width)
            self.decoder = Decoder(out_ch=out_ch, group_num=group_num, width=width)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # encoding path
        x = self.encoder_class(x)
        # decoding + concat path
        d = self.decoder(x)
        return d

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
