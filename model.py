import torch
import torch.nn as nn
import torchvision
from torchvision import models

class MyModel(nn.Module):
    def __init__(self, K):
        super(MyModel, self).__init__()
        self.pretrained_model = torchvision.models.resnet18(pretrained=True)
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        
        num_ftrs = self.pretrained_model.fc.in_features
        self.pretrained_model.fc = nn.Linear(num_ftrs, K)

    def forward(self, X):
        out = self.pretrained_model(X)
        return out