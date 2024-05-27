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
    def __init__(self, img_ch=5, group_num:int=1):  # 6 is 3 TRA + 3 COR
        super(Encoder, self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.Conv1 = conv_block_group(ch_in=img_ch, ch_out=32*group_num, group_num=group_num)
        self.Conv2 = conv_block_group(ch_in=32*group_num, ch_out=64*group_num, group_num=group_num)
        self.Conv3 = conv_block_group(ch_in=64*group_num, ch_out=128*group_num, group_num=group_num)
        self.Conv4 = conv_block_group(ch_in=128*group_num, ch_out=256*group_num, group_num=group_num)
    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        return x4


class Classifier(nn.Module):
    def __init__(self, num_classes=2, group_num=2):
        super(Classifier, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier_fc = nn.Sequential(
            nn.Linear(256*group_num * 7 * 7, 4096),
            nn.SiLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.SiLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier_fc(x)
        # x = self.softmax(x)
        return x
    

class Classifier11(nn.Module):
    def __init__(self, num_classes=2, group_num=2):
        super(Classifier11, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier_fc = nn.Sequential(
            nn.Linear(256*group_num, 2048),
            nn.SiLU(True),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.SiLU(True),
            nn.Dropout(),
            nn.Linear(2048, num_classes),
        )
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier_fc(x)
        # x = self.softmax(x)
        return x


class ModelClass(nn.Module):
    def __init__(self, img_ch=5, group_num=2, num_classes=2, encoder=Encoder, classifier=Classifier, init_weights: bool = True):
        super(ModelClass, self).__init__()
        self.encoder_class = encoder(img_ch=img_ch, group_num=group_num)
        self.classifier = classifier(num_classes=num_classes, group_num=group_num)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # encoding path
        x = self.encoder_class(x)
        # decoding + concat path
        d = self.classifier(x)
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
