import torch
import torch.nn as nn


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(self, img_ch=5):
        super(Encoder, self).__init__()
        self.single_encoder = nn.Sequential(
                                conv_block(ch_in=img_ch, ch_out=32),
                                nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
                                conv_block(ch_in=32, ch_out=64),
                                nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
                                conv_block(ch_in=64, ch_out=128),
                                nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
                                conv_block(ch_in=128, ch_out=256))
    def forward(self, x):
        # encoding path
        x = self.single_encoder
        return x


class Classifier(nn.Module):
    def __init__(self, num_classes=2, group_num=2):
        super(Classifier, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier_fc = nn.Sequential(
            nn.Linear(256*group_num * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self.softmax = nn.Softmax()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier_fc(x)
        x = self.softmax(x)
        return x


class ModelClass(nn.Module):
    def __init__(self, img_ch=5, num_class=2, encoder=Encoder, classifier=Classifier, init_weights: bool = True):
        super(ModelClass, self).__init__()
        self.group_num = img_ch // 5
        self.encoder_list = []
        for order in range(self.group_num ):
            self.encoder_list.append(encoder(img_ch=5))
        self.classifier = classifier(num_class=num_class, group_num=self.group_num )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # encoding path
        for i in range(self.group_num):
            x[:, i*5:(i+1):5, :] = self.encoder_list[i](x[:, i*5:(i+1):5, :])
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
