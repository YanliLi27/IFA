import torch.nn as nn


## First model
class catsdogs_3d_nn(nn.Module):
    # VGG11 [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]
    # NUM- nn.Conv2d(in_channels, v, kernel_size=3, padding=1) + nn.ReLU(inplace=True)
    # m - nn.MaxPool2d(kernel_size=2, stride=2)
    def __init__(self, in_channel=3, num_classes=2, init_weights=False):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv3d(in_channels=in_channel, out_channels=64, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
            # nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),  # 10, 112, 112
            nn.Conv3d(64, 128, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
            # nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),  # 10, 56, 56
            nn.Conv3d(128, 256, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
            # nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
            # nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),  # 10, 28, 28
            nn.Conv3d(256, 512, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
            # nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
            # nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),  # 10, 14, 14
            nn.Conv3d(512, 512, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
            # nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
            # nn.BatchNorm3d(128),
            nn.ReLU(inplace=False)  # 10, 14, 14
        )
        self.classifier = nn.Sequential(
            # nn.MaxPool3d(kernel_size=2),  # 5, 8, 8 
            nn.AdaptiveAvgPool3d((1, 7, 7)),
            nn.Flatten(),
            nn.Linear(512*49, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
            nn.Softmax(dim=1)
        )
        if init_weights:
            self._initialize_weights()
        
    def forward(self,x):
        # print(x.shape)
        x = self.cnn(x)
        # print(x.shape)
        return self.classifier(x)
    

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
