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


class conv_block_group3d(nn.Module):
    def __init__(self, ch_in, ch_out, group_num=2):
        super(conv_block_group, self).__init__()
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
    

class Encoder(nn.Module):
    def __init__(self, img_ch=5, group_cap=5, width=2, out_ch:int=512):  # 6 is 3 TRA + 3 COR
        super(Encoder, self).__init__()
        group_num = img_ch // group_cap

        self.Maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.Conv1 = conv_block_group(ch_in=img_ch, ch_out=32*group_num*width, group_num=group_num)
        self.Conv2 = conv_block_group(ch_in=32*group_num*width, ch_out=64*group_num*width, group_num=group_num)
        self.Conv3 = conv_block_group(ch_in=64*group_num*width, ch_out=128*group_num*width, group_num=group_num)
        self.Conv4 = conv_block_group(ch_in=128*group_num*width, ch_out=256*group_num*width, group_num=group_num)
        self.Convlast = conv_block_group(ch_in=256*group_num*width, ch_out=out_ch, group_num=group_num)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
    def forward(self, x):
        # encoding path
        x = self.Conv1(x)
        x = self.Maxpool(x)
        x = self.Conv2(x)
        x = self.Maxpool(x)
        x = self.Conv3(x)
        x = self.Maxpool(x)
        x = self.Conv4(x)
        x = self.Maxpool(x)
        x = self.Convlast(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class Encoder3d(nn.Module):
    def __init__(self, img_ch=5, group_cap=5, width=2, out_ch:int=512):  # 6 is 3 TRA + 3 COR
        super(Encoder, self).__init__()
        group_num = img_ch // group_cap

        self.Maxpool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.Conv1 = conv_block_group3d(ch_in=img_ch, ch_out=32*group_num*width, group_num=group_num)
        self.Conv2 = conv_block_group3d(ch_in=32*group_num*width, ch_out=64*group_num*width, group_num=group_num)
        self.Conv3 = conv_block_group3d(ch_in=64*group_num*width, ch_out=128*group_num*width, group_num=group_num)
        self.Conv4 = conv_block_group3d(ch_in=128*group_num*width, ch_out=256*group_num*width, group_num=group_num)
        self.Convlast = conv_block_group3d(ch_in=256*group_num*width, ch_out=out_ch, group_num=group_num)
        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))
    def forward(self, x):
        # encoding path
        x = self.Conv1(x)
        x = self.Maxpool(x)
        x = self.Conv2(x)
        x = self.Maxpool(x)
        x = self.Conv3(x)
        x = self.Maxpool(x)
        x = self.Conv4(x)
        x = self.Maxpool(x)
        x = self.Convlast(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x
    

class MLP(nn.Module):
    def __init__(self, num_classes=2, in_channel:int=43):
        super(MLP, self).__init__()
        self.classifier_fc = nn.Sequential(
            nn.Linear(in_channel, 200),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(200, 200),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(200, num_classes),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, 1)
        x = self.classifier_fc(x)
        x = self.softmax(x)
        return x


class ModelMulti(nn.Module):
    def __init__(self, img_ch:int=5, out_ch:int=2, num_classes:int=2, dimension:int=2,
                 group_cap:int=5, width:int=2, extension:int=0, init_weights:bool=True):
        super(ModelMulti, self).__init__()
        if dimension==2:
            self.encoder_class = Encoder(img_ch=img_ch, group_cap=group_cap, width=width, out_ch=out_ch+extension)
        else:
            self.encoder_class = Encoder3d(img_ch=img_ch, group_cap=group_cap, width=width)
        self.classfier = MLP(num_classes=num_classes, in_channel=out_ch+extension)
        self.extension = extension

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # encoding path
        x = self.encoder_class(x)
        # x -- [batch, channel]
        if self.extension>0:
            c = self.classfier(x)     
            return x[:, :-self.extension], c       
        else:
            c = self.classfier(x)
            return x, c

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)



class ModelMulti_detach(nn.Module):
    def __init__(self, img_ch:int=5, out_ch:int=2, num_classes:int=2, dimension:int=2,
                 group_cap:int=5, width:int=2, extension:int=0, init_weights:bool=True):
        super(ModelMulti_detach, self).__init__()
        if dimension==2:
            self.encoder_class = Encoder(img_ch=img_ch, group_cap=group_cap, width=width, out_ch=out_ch+extension)
        else:
            self.encoder_class = Encoder3d(img_ch=img_ch, group_cap=group_cap, width=width)
        self.classfier = MLP(num_classes=num_classes, in_channel=out_ch+extension)
        self.extension = extension

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # encoding path
        x = self.encoder_class(x)
        # x -- [batch, channel]
        if self.extension>0:
            c = self.classfier(torch.cat((x[:, :-self.extension].detach(), x[:, -self.extension:]), dim=1))     
            return x[:, :-self.extension], c       
        else:
            c = self.classfier(x.detach())
            return x, c

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)