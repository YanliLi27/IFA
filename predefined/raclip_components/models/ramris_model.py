import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, num_feature, num_classes=2):
        super(Classifier, self).__init__()
        self.classifier_fc = nn.Sequential(
            nn.Linear(num_feature, 4096),
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
    def __init__(self, img_ch=43, num_classes=2, classifier=Classifier, init_weights: bool = True):
        super(ModelClass, self).__init__()
        self.classifier = classifier(num_feature=img_ch,num_classes=num_classes)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # encoding path
        # decoding + concat pathx
        x = self.classifier(x)
        return x

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