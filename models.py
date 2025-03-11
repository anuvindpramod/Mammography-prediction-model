import torch.nn as nn
import torchvision.models as models

class MammoNet(nn.Module):
    def __init__(self,dropout_rate=0.5):
        super().__init__()
        self.features=nn.Sequential(
            nn.Conv2d(3,32,3,padding=1,stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier=nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(256*14*14,512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,2)

        )

    def forward(self,x):
        x=self.features(x)
        x=x.view(x.size(0),-1)
        return self.classifier(x)
    
class Pretrained_ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet=models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet.fc=nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.resnet.fc.in_features,2)
        )
        
        
    def forward(self,x):
        return self.resnet(x)
    
