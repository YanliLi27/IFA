import torch
import torch.nn as nn


class conv_block_group2d(nn.Module):
    def __init__(self, ch_in, ch_out, group_num=2):
        super(conv_block_group2d, self).__init__()
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


class conv_block_group(nn.Module):
    def __init__(self, ch_in, ch_out, group_num=2):
        super(conv_block_group, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=(1, 3, 3), stride=(1, 1), padding=(1, 1), bias=False, groups=group_num),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=(1, 3, 3), stride=(1, 1), padding=(1, 1), bias=False, groups=group_num),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.conv(x)
        return x
    

class conv_block_group3d(nn.Module):
    def __init__(self, ch_in, ch_out, group_num=2):
        super(conv_block_group3d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False, groups=group_num),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch_out, ch_out, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False, groups=group_num),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.conv(x)
        return x
    

class EncoderCOR(nn.Module):
    def __init__(self, img_ch=2, width=2):  # 6 is 3 TRA + 3 COR
        super(EncoderCOR, self).__init__()
        group_num = img_ch

        self.Maxpool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.DownMaxpool = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.Conv1 = conv_block_group(ch_in=img_ch, ch_out=32*group_num*width, group_num=group_num)
        self.Conv2 = conv_block_group(ch_in=32*group_num*width, ch_out=64*group_num*width, group_num=group_num)
        self.Conv3 = conv_block_group(ch_in=64*group_num*width, ch_out=128*group_num*width, group_num=group_num)
        self.Conv4 = conv_block_group(ch_in=128*group_num*width, ch_out=256*group_num*width, group_num=group_num)
        self.Conv5 = conv_block_group3d(ch_in=256*group_num*width, ch_out=256*group_num*width, group_num=group_num)
        self.Conv6 = conv_block_group3d(ch_in=256*group_num*width, ch_out=256*group_num*width, group_num=group_num)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 7, 7))
    def forward(self, x):
        # encoding path
        x = self.Conv1(x)
        x = self.Maxpool(x)
        x = self.Conv2(x)
        x = self.Maxpool(x)
        x = self.Conv3(x)
        x = self.Maxpool(x)
        x = self.Conv4(x)
        x = self.DownMaxpool(x)
        x = self.Conv5(x)
        x = self.DownMaxpool(x)
        x = self.Conv6(x)
        x = self.avgpool(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)
    

class EncoderTRA(nn.Module):
    def __init__(self, img_ch=2, width=2):  # 6 is 3 TRA + 3 COR
        super(EncoderCOR, self).__init__()
        group_num = img_ch

        self.Maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.Conv1 = conv_block_group2d(ch_in=img_ch, ch_out=32*group_num*width, group_num=group_num)
        self.Conv2 = conv_block_group2d(ch_in=32*group_num*width, ch_out=64*group_num*width, group_num=group_num)
        self.Conv3 = conv_block_group2d(ch_in=64*group_num*width, ch_out=128*group_num*width, group_num=group_num)
        self.Conv4 = conv_block_group2d(ch_in=128*group_num*width, ch_out=256*group_num*width, group_num=group_num)
        self.Conv5 = conv_block_group2d(ch_in=256*group_num*width, ch_out=256*group_num*width, group_num=group_num)
        self.Conv6 = conv_block_group2d(ch_in=256*group_num*width, ch_out=256*group_num*width, group_num=group_num)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
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
        x = self.Conv5(x)
        x = self.Maxpool(x)
        x = self.Conv6(x)
        return torch.flatten(x, 1)
    

class Classifier(nn.Module):
    def __init__(self, num_classes=2, num_scan=2):
        super(Classifier, self).__init__()
        
        self.classifier_fc = nn.Sequential(
            nn.Linear(256*num_scan * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.classifier_fc(x)
        x = self.softmax(x)
        return x
    


class ModelClass(nn.Module):
    def __init__(self, num_scan=2, img_ch=5, num_classes=2, classifier=Classifier, init_weights: bool = True):
        super(ModelClass, self).__init__()
        self.num_scan = num_scan
        self.encoder = []
        for i in range(self.num_scan):
            self.encoder.append(EncoderTRA)
            self.encoder.append(EncoderCOR)
        self.classifier = classifier(num_classes=num_classes, num_scan=self.num_scan)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # encoding path
        feature = torch.Tensor()
        for i in range(self.num_scan):
            feature1 = self.encoder[i](x[i])
            feature = torch.cat((feature, feature1), dim=1)
        # decoding + concat path
        d = self.classifier(feature)
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