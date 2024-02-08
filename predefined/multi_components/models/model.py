import torch
import torch.nn as nn

## First model
class scratch_nn(nn.Module):
    def __init__(self, in_channel=3, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=100, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(100, 200, 3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(200, 200, 3, stride=1, padding=0)
        self.mpool = nn.MaxPool2d(kernel_size=3)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(49*200,1024)
        self.linear2 = nn.Linear(1024,512)
        self.linear3 = nn.Linear(512,num_classes)
        # self.classifier = nn.Softmax(dim=1)
        
    def forward(self,x):
        x = self.mpool( self.relu(self.conv1(x)) )
        x = self.mpool( self.relu(self.conv2(x)) )
        x = self.mpool( self.relu(self.conv3(x)) )
        x = torch.flatten(x, start_dim=1)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        # x = self.classifier(x)
        return x


class scratch_mnist(nn.Module):
    def __init__(self, in_channel=1, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=100, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(100, 200, 3, stride=1, padding=0)
        self.mpool = nn.MaxPool2d(kernel_size=3)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(4*200,512)
        self.linear2 = nn.Linear(512,num_classes)
        # self.classifier = nn.Softmax(dim=1)
        
    def forward(self,x):
        x = self.mpool( self.relu(self.conv1(x)) )
        x = self.mpool( self.relu(self.conv2(x)) )
        x = torch.flatten(x, start_dim=1)
        x = self.linear1(x)
        x = self.linear2(x)
        # x = self.classifier(x)
        return x