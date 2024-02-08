import torch.nn as nn


## First model
class us_nn(nn.Module):
    def __init__(self, in_channel=1, num_classes=2, init_weights=False):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # 64, 64
            nn.Conv2d(32, 32, 3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # 32, 32
            nn.Conv2d(32, 64, 3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # 16, 16
            nn.Conv2d(64, 128, 3, stride=1, padding=0),
            nn.Conv2d(128, 128, 3, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2),  # 8, 8 
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
            nn.Softmax(dim=1)
        )
        if init_weights:
            self._initialize_weights()
        
    def forward(self,x):
        return self.cnn(x)
    

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
