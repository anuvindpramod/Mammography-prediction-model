import torch.nn as nn
import torchvision.models as models

class MammoNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(
            nn.Conv2d(3,32,3,padding=1,stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier=nn.Sequential(
            nn.Linear(128*28*28,512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512,2)
        )

    def forward(self,x):
        x=self.features(x)
        x=x.view(x.size(0),-1)
        return self.classifier(x)
    
class Pretrained_ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet=models.resnet18(pretrained=True,progress=True)
        self.resnet.conv1=nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.resnet.fc=nn.Linear(self.resnet.fc.in_features,2)
        
    def forward(self,x):
        return self.resnet(x)
    
